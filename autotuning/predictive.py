#!/usr/bin/env python3
"""Lightweight, architecture-specific predictive autotuner for MGXBench.

Calibration uses the unmodified measured benchmark path, and prediction uses
piecewise log-log interpolation/extrapolation for each (library, grid, BK).
No third-party ML dependency is required. Predictions are not measurements.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys

LIBRARIES = ('MMM', 'YYY', 'CCC', 'XXX', 'NNN', 'WWW', 'SSS')
NAMES = {'MMM':'MPI-SYNC','YYY':'MPI-CHUNKED-ASYNC','CCC':'CUDA-AWARE-SYNC',
         'XXX':'CUDA-AWARE-ASYNC','NNN':'NCCL-ASYNC','WWW':'NVSHMEM-CHUNKED-SYNC',
         'SSS':'NVSHMEM-ASYNC'}


def fingerprint(root, decomposition, ranks, build_dir='build'):
    """Stable compatibility key plus diagnostic allocation metadata."""
    exe = Path(root) / build_dir / ('mm_2d' if decomposition == '2d' else 'mm_1d')
    if not exe.is_file():
        raise ValueError(f'Executable not found: {exe}; build MGXBench first')
    try:
        output = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version',
                                 '--format=csv,noheader'], capture_output=True,
                                text=True, timeout=10, check=True).stdout.strip().splitlines()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        raise ValueError('Cannot identify GPU architecture (nvidia-smi unavailable)') from None
    if not output:
        raise ValueError('No GPU detected')
    gpu_names = sorted(set(line.split(',')[0].strip() for line in output))
    stable = {'gpu_models': gpu_names, 'ranks': ranks,
              'decomposition': decomposition,
              'binary_sha256': hashlib.sha256(exe.read_bytes()).hexdigest(),
              'slurm_nodes': os.environ.get('SLURM_NNODES', '1')}
    signature = dict(stable)
    signature['driver_versions'] = sorted(set(line.split(',')[-1].strip() for line in output))
    signature['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    signature['gpus_per_node'] = os.environ.get('SLURM_GPUS_ON_NODE', '')
    signature['id'] = hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:16]
    return signature


def compatible_signature(stored, actual):
    """Accept legacy signatures only if their substantive fields match."""
    fields = ('gpu_models', 'ranks', 'decomposition', 'binary_sha256', 'slurm_nodes')
    def normalized_gpu(x):
        return sorted(set(str(v).split(',')[0].strip() for v in x))
    return (normalized_gpu(stored.get('gpu_models', [])) == normalized_gpu(actual.get('gpu_models', []))
            and all(str(stored.get(k)) == str(actual.get(k)) for k in fields[1:]))


def model_filename(decomposition, ranks):
    return f'calibration-{decomposition}-{ranks}gpu.json'


def resolve_model(root, decomposition, ranks, supplied=None):
    if supplied:
        path = Path(supplied)
        return path if path.is_absolute() else root / path
    named = root / 'autotuning' / 'models' / model_filename(decomposition, ranks)
    legacy = root / 'autotuning' / 'models' / 'calibration.json'
    return named if named.exists() or not legacy.exists() else legacy


def factor_grids(ranks):
    return [(r, ranks // r) for r in range(1, ranks + 1) if ranks % r == 0]


def candidate_configs(decomposition, ranks, sizes, libraries, bks, grids=None):
    if decomposition == '1d':
        return [{'library': lib, 'grid_rows': None, 'grid_cols': None, 'summa_bk': None}
                for lib in libraries]
    if grids is None:
        grids = factor_grids(ranks)
    result = []
    for r, c in grids:
        if r * c != ranks or any(n % r or n % c for n in sizes):
            continue
        for bk in bks:
            if bk <= 0:
                continue
            for lib in libraries:
                result.append({'library': lib, 'grid_rows': r,
                               'grid_cols': c, 'summa_bk': bk})
    return result


def key(config):
    return (config['library'], config.get('grid_rows'), config.get('grid_cols'),
            config.get('summa_bk'))


def estimate(samples, target):
    """Log-log interpolation using nearest bracket, extrapolating at the ends.

    Returns predicted seconds and a marker for extrapolation. A minimum of two
    distinct sizes is required; no confidence interval is implied.
    """
    by_size = {}
    for item in samples:
        n, t = int(item['size']), float(item['median_s'])
        if n > 0 and math.isfinite(t) and t > 0:
            by_size.setdefault(n, []).append(t)
    points = sorted((n, statistics.median(times)) for n, times in by_size.items())
    if len(points) < 2:
        raise ValueError('At least two calibrated sizes required per candidate')
    for n, t in points:
        if n == target:
            return t, False
    if target < points[0][0]:
        (n1, t1), (n2, t2) = points[:2]
    elif target > points[-1][0]:
        (n1, t1), (n2, t2) = points[-2:]
    else:
        i = next(i for i in range(len(points)-1) if points[i][0] < target < points[i+1][0])
        (n1, t1), (n2, t2) = points[i:i+2]
    slope = math.log(t2/t1) / math.log(n2/n1)
    # Physical sanity check: avoid pathological extrapolation on noisy samples.
    slope = max(0.0, min(4.0, slope))
    return t1 * (target/n1)**slope, not (points[0][0] <= target <= points[-1][0])



def _points(samples):
    by_size = {}
    for m in samples:
        n, t = int(m['size']), float(m['median_s'])
        if n > 0 and math.isfinite(t) and t > 0:
            by_size.setdefault(n, []).append(t)
    return sorted((n, statistics.median(ts)) for n, ts in by_size.items())


def _physical_fit(points, target, method, alpha=0.1):
    import numpy as np
    if len(points) < 4:
        raise ValueError('Physical regression requires >=4 distinct sizes')
    sizes = np.asarray([n for n, _ in points], dtype=float)
    times = np.asarray([t for _, t in points], dtype=float)
    ref = float(max(sizes))
    def features(ns):
        x = np.asarray(ns, dtype=float) / ref
        return np.column_stack((x**2, x**3))
    x = features(sizes)
    if method == 'ml':
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled = scaler.fit_transform(x)
        reg = Ridge(alpha=alpha, fit_intercept=True)
        reg.fit(scaled, times)
        predicted = float(reg.predict(scaler.transform(features([target])))[0])
    else:
        design = np.column_stack((np.ones(len(sizes)), x))
        coef, *_ = np.linalg.lstsq(design, times, rcond=None)
        predicted = float(np.dot(np.r_[1., features([target])[0]], coef))
    if not math.isfinite(predicted) or predicted <= 0:
        raise ValueError('Nonpositive or nonfinite physical prediction')
    return predicted


def _predict_method(model, size, ranks, decomposition, requested_grid, method):
    if size <= 0 or ranks <= 0:
        raise ValueError('size and ranks must be positive')
    if (model['signature']['ranks'] != ranks or
            model['signature']['decomposition'] != decomposition):
        raise ValueError('Model is for a different rank count or decomposition')
    predictions = []
    for candidate in model['candidates']:
        if requested_grid and (candidate['grid_rows'], candidate['grid_cols']) != requested_grid:
            continue
        if decomposition == '2d' and (size % candidate['grid_rows'] or size % candidate['grid_cols']):
            continue
        if decomposition == '1d' and size % ranks:
            continue
        points = _points(candidate['measurements'])
        try:
            if len(points) >= 4:
                seconds = _physical_fit(points, size, method)
                algorithm = 'ridge-physical' if method == 'ml' else 'physical-polynomial'
            elif method == 'extrapolation':
                seconds, _ = estimate(candidate['measurements'], size)
                algorithm = 'piecewise-power-law-fallback'
            else:
                continue
        except (ValueError, ImportError, ArithmeticError):
            if method != 'extrapolation':
                continue
            try:
                seconds, _ = estimate(candidate['measurements'], size)
                algorithm = 'piecewise-power-law-fallback'
            except ValueError:
                continue
        predictions.append({**{k: candidate[k] for k in ('library','grid_rows','grid_cols','summa_bk')},
                            'predicted_s': seconds, 'algorithm': algorithm,
                            'calibration_sizes': len(points),
                            'extrapolated': not (points[0][0] <= size <= points[-1][0])})
    if not predictions:
        raise ValueError('No compatible candidates with sufficient valid calibration sizes')
    return sorted(predictions, key=lambda r: r['predicted_s'])


def predict_ml(model, size, ranks, decomposition, requested_grid=None):
    return _predict_method(model, size, ranks, decomposition, requested_grid, 'ml')


def select_predictions(model, size, ranks, decomposition, requested_grid, method):
    if method in ('extrapolation', 'ml'):
        return _predict_method(model, size, ranks, decomposition, requested_grid, method), method
    if method == 'auto':
        # Compare extrapolation error at the largest held-out size per candidate.
        votes = {'extrapolation': [], 'ml': []}
        for candidate in model['candidates']:
            points = _points(candidate['measurements'])
            if len(points) < 5:
                continue
            for candidate_method in votes:
                try:
                    predicted = _physical_fit(points[:-1], points[-1][0], candidate_method)
                    votes[candidate_method].append(abs(predicted - points[-1][1]) / points[-1][1])
                except (ValueError, ImportError):
                    pass
        choices = [(statistics.median(errors), method_name)
                   for method_name, errors in votes.items() if errors]
        chosen = min(choices)[1] if choices else 'extrapolation'
        return _predict_method(model, size, ranks, decomposition, requested_grid, chosen), chosen
    raise ValueError(f'Unknown prediction model: {method}')


def load_model(path):
    model = json.loads(Path(path).read_text())
    if model.get('schema') != 1:
        raise ValueError('Unsupported autotuning model schema')
    return model


def predict(model, size, ranks, decomposition, requested_grid=None):
    return _predict_method(model, size, ranks, decomposition, requested_grid, 'extrapolation')


def _matching_report(data, cfg, size, args):
    """Check the actual benchmark.py schema, not the filesystem timestamp."""
    a = data.get('command_args') or {}
    libs = a.get('libraries')
    if isinstance(libs, str):
        libs = [libs]
    if (a.get('label', data.get('label')) != 'auto-calibration'
            or a.get('size') != size or a.get('ranks') != args.ranks
            or a.get('decomposition') != args.decomposition
            or libs != [cfg['library']]):
        return False
    if args.decomposition == '2d':
        return (a.get('grid_rows') == cfg['grid_rows']
                and a.get('grid_cols') == cfg['grid_cols']
                and a.get('summa_bk') == cfg['summa_bk'])
    return True


def calibrate(args):
    root = Path(args.project_root).resolve()
    signature = fingerprint(root, args.decomposition, args.ranks, args.build_dir)
    sizes = sorted(set(args.sizes))
    if len(sizes) < 2 or any(n <= 0 for n in sizes):
        raise ValueError('Supply at least two distinct positive calibration sizes')
    if args.decomposition == '1d' and any(n % args.ranks for n in sizes):
        raise ValueError('1D calibration sizes must be divisible by ranks')
    grids = [tuple(map(int, item.lower().split('x'))) for item in args.grids] if args.grids else None
    if args.grid_rows is not None:
        grids = [(args.grid_rows, args.grid_cols)]
    configs = candidate_configs(args.decomposition, args.ranks, sizes,
                                args.libraries, args.bks, grids)
    if not configs:
        raise ValueError('No valid calibration candidates')

    model = {'schema': 1, 'signature': signature, 'sizes': sizes, 'candidates': [],
             'warning': 'Empirical calibration; predictions are estimates, not measured timings'}
    out = Path(args.model or ('autotuning/models/' + model_filename(args.decomposition, args.ranks)))
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.stem + '.partial.json')
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    runs_dir = output_dir / 'runs'
    log_dir = output_dir / 'calibration_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    stats = {'attempted': 0, 'valid_measurements': 0, 'failed_launches': 0,
             'missing_reports': 0, 'rejected_reports': 0, 'rejected_results': 0}

    for idx, cfg in enumerate(configs, 1):
        item = {**cfg, 'measurements': []}
        for n in sizes:
            stats['attempted'] += 1
            cmd = [sys.executable, str(root / 'benchmark.py'),
                   '--decomposition', args.decomposition, '--size', str(n),
                   '--ranks', str(args.ranks), '--libraries', cfg['library'],
                   '--warmup', str(args.warmup), '--repeats', str(args.repeats),
                   '--build-dir', args.build_dir, '--launcher', args.launcher,
                   '--launcher-args', args.launcher_args,
                   '--output', str(output_dir), '--label', 'auto-calibration']
            if args.decomposition == '2d':
                cmd += ['--grid-rows', str(cfg['grid_rows']),
                        '--grid-cols', str(cfg['grid_cols']), '--summa-bk', str(cfg['summa_bk'])]
            grid_label = (f"grid{cfg['grid_rows']}x{cfg['grid_cols']}_bk{cfg['summa_bk']}"
                          if args.decomposition == '2d' else '1d')
            log_name = f"{idx:03d}_{cfg['library']}_N{n}_{grid_label}"
            print(f'Calibrating {idx}/{len(configs)} {cfg} N={n}', flush=True)

            # Capture both newly created and overwritten benchmark.json files.
            before = {p.resolve(): (p.stat().st_mtime_ns, p.stat().st_size)
                      for p in runs_dir.glob('*/benchmark.json')}
            try:
                completed = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
            except OSError as exc:
                stats['failed_launches'] += 1
                (log_dir / (log_name + '.stderr')).write_text(str(exc) + '\n')
                print(f'  FAILED to launch: {exc}', file=sys.stderr, flush=True)
                continue
            (log_dir / (log_name + '.stdout')).write_text(completed.stdout)
            (log_dir / (log_name + '.stderr')).write_text(completed.stderr)
            (log_dir / (log_name + '.command.txt')).write_text(' '.join(cmd) + '\n')
            if completed.returncode:
                stats['failed_launches'] += 1
                print(f'  FAILED (exit {completed.returncode}); logs: {log_dir / log_name}',
                      file=sys.stderr, flush=True)
                if completed.stderr:
                    print('  stderr tail: ' + completed.stderr[-800:], file=sys.stderr, flush=True)
                continue

            reports = []
            for report in runs_dir.glob('*/benchmark.json'):
                try:
                    stat = report.stat()
                    if before.get(report.resolve()) == (stat.st_mtime_ns, stat.st_size):
                        continue
                    data = json.loads(report.read_text())
                    if _matching_report(data, cfg, n, args):
                        reports.append((stat.st_mtime_ns, report, data))
                except (OSError, ValueError, TypeError) as exc:
                    print(f'  WARNING: Cannot read {report}: {exc}', file=sys.stderr, flush=True)
            if not reports:
                stats['missing_reports'] += 1
                print(f'  WARNING: No matching new/updated benchmark.json for {log_name}; '
                      f'check {log_dir / (log_name + ".stdout")}', file=sys.stderr, flush=True)
                continue
            _, report_path, data = max(reports, key=lambda x: x[0])
            if data.get('failures'):
                stats['rejected_reports'] += 1
                print(f'  WARNING: Benchmark failures in {report_path}: {data["failures"]}',
                      file=sys.stderr, flush=True)
                continue
            accepted = False
            for row in data.get('results', []):
                if row.get('library') != cfg['library'] or row.get('validated') is not True:
                    continue
                median = row.get('median_s')
                if (not isinstance(median, (int, float)) or isinstance(median, bool)
                        or not math.isfinite(median) or median <= 0):
                    continue
                item['measurements'].append({'size': n, 'median_s': median})
                stats['valid_measurements'] += 1
                accepted = True
                print(f'  SUCCESS: N={n} median={median:.6f}s ({report_path})', flush=True)
                break
            if not accepted:
                stats['rejected_results'] += 1
                print(f'  WARNING: No validated {cfg["library"]} median_s in {report_path}; '
                      f'results={data.get("results", [])}', file=sys.stderr, flush=True)

        count = len({x['size'] for x in item['measurements']})
        if count >= 2:
            model['candidates'].append(item)
            print(f'  ACCEPTED candidate: {count} distinct sizes', flush=True)
        else:
            print(f'  REJECTED candidate: only {count} distinct valid sizes',
                  file=sys.stderr, flush=True)
        # Preserve progress without destroying a previously valid model.
        partial.write_text(json.dumps(model, indent=2) + '\n')

    print(f'Calibration summary: {stats}; eligible candidates='
          f'{len(model["candidates"])}/{len(configs)}', flush=True)
    if not model['candidates']:
        raise ValueError(f'Calibration produced no candidate with two valid measurements; '
                         f'partial model: {partial}; logs: {log_dir}')
    if out.exists():
        import shutil
        shutil.copy2(out, out.with_suffix(out.suffix + '.bak'))
    partial.replace(out)
    print(f'Calibration saved: {out} ({len(model["candidates"])} candidates)')



def recover_calibration(args):
    """Rebuild a calibration model from existing benchmark.json reports.

    The report schema must contain command_args and validated result rows.
    Historical reports do not necessarily record GPU/binary provenance, so
    writing a model requires explicit --assume-compatible confirmation.
    """
    root = Path(args.project_root).resolve()
    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        input_dir = root / input_dir
    reports = sorted(input_dir.glob('*/benchmark.json'))
    if not reports:
        raise ValueError(f'No benchmark.json files found under {input_dir}')
    if not args.assume_compatible:
        raise ValueError('Cannot verify GPU/binary provenance of historical reports. '
                         'Review the source runs and rerun with --assume-compatible '
                         'only if they used this same GPU architecture, allocation, and binary.')

    signature = fingerprint(root, args.decomposition, args.ranks, args.build_dir)
    grouped = {}
    reasons = {}
    def reject(reason):
        reasons[reason] = reasons.get(reason, 0) + 1

    for report in reports:
        try:
            data = json.loads(report.read_text())
        except (OSError, ValueError) as exc:
            reject(f'unreadable JSON: {type(exc).__name__}')
            continue
        a = data.get('command_args') or {}
        if not isinstance(a, dict):
            reject('command_args is not an object')
            continue
        if a.get('label', data.get('label')) != 'auto-calibration':
            reject('label differs from auto-calibration')
            continue
        try:
            size = int(a['size'])
            ranks = int(a['ranks'])
        except (KeyError, ValueError, TypeError):
            reject('missing/invalid size or ranks')
            continue
        if ranks != args.ranks or a.get('decomposition') != args.decomposition:
            reject('different rank count or decomposition')
            continue
        if size <= 0 or (args.decomposition == '1d' and size % ranks):
            reject('invalid matrix size')
            continue
        libs = a.get('libraries')
        if isinstance(libs, str):
            libs = [libs]
        if not isinstance(libs, list) or len(libs) != 1 or libs[0] not in LIBRARIES:
            reject('missing/ambiguous library')
            continue
        lib = libs[0]
        if args.decomposition == '2d':
            try:
                gr, gc, bk = (int(a[k]) for k in ('grid_rows', 'grid_cols', 'summa_bk'))
            except (KeyError, ValueError, TypeError):
                reject('missing/invalid grid or BK')
                continue
            if gr <= 0 or gc <= 0 or gr * gc != ranks or bk <= 0 or size % gr or size % gc:
                reject('incompatible grid/BK')
                continue
        else:
            gr = gc = bk = None
        if data.get('failures'):
            reject('report has failures')
            continue
        rows = data.get('results') or []
        if not isinstance(rows, list):
            reject('results is not a list')
            continue
        times = []
        for row in rows:
            if not isinstance(row, dict) or row.get('library') != lib or row.get('validated') is not True:
                continue
            try:
                t = float(row['median_s'])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(t) and t > 0:
                times.append(t)
        if not times:
            reject('no validated positive median_s')
            continue
        k = (lib, gr, gc, bk)
        grouped.setdefault(k, {}).setdefault(size, []).append(statistics.median(times))

    candidates = []
    for (lib, gr, gc, bk), by_size in sorted(grouped.items(), key=lambda kv: str(kv[0])):
        if len(by_size) < 2:
            reject(f'candidate {lib}/{gr}x{gc}/BK{bk} has only {len(by_size)} size(s)')
            continue
        candidates.append({'library': lib, 'grid_rows': gr, 'grid_cols': gc,
                           'summa_bk': bk,
                           'measurements': [{'size': n, 'median_s': statistics.median(ts)}
                                            for n, ts in sorted(by_size.items())]})
    print(f'Reports scanned: {len(reports)}; eligible candidates: {len(candidates)}', flush=True)
    for reason, count in sorted(reasons.items()):
        print(f'  REJECTED {count}: {reason}', flush=True)
    if not candidates:
        raise ValueError('Recovery found no candidate with >=2 validated sizes. '
                         'Inspect command_args, results, validated, median_s and failures.')
    sizes = sorted({m['size'] for c in candidates for m in c['measurements']})
    model = {'schema': 1, 'signature': signature, 'sizes': sizes,
             'candidates': candidates,
             'warning': ('Recovered empirical calibration; predictions are estimates, '
                         'not measured timings. Historical hardware/binary compatibility '
                         'was confirmed manually via --assume-compatible.')}
    out = Path(args.model or ('autotuning/models/' + model_filename(args.decomposition, args.ranks)))
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        import shutil
        backup = out.with_suffix(out.suffix + '.bak')
        shutil.copy2(out, backup)
        print(f'Previous model backup: {backup}', flush=True)
    tmp = out.with_name(out.name + '.tmp')
    tmp.write_text(json.dumps(model, indent=2) + '\n')
    tmp.replace(out)
    print(f'Recovered model saved: {out}; candidates={len(candidates)}', flush=True)


def validate_and_feedback(args, best, report, model, model_path, root):
    """Execute the selected configuration, then optionally append a validated sample."""
    import shutil
    import datetime
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    validation_dir = output / 'validation'
    command = [sys.executable, str(root / 'benchmark.py'),
               '--decomposition', args.decomposition, '--size', str(args.size),
               '--ranks', str(args.ranks), '--libraries', best['library'],
               '--warmup', str(args.warmup), '--repeats', str(args.repeats),
               '--build-dir', args.build_dir, '--launcher', args.launcher,
               '--launcher-args', args.launcher_args, '--timeout', str(args.timeout),
               '--output', str(validation_dir), '--label', 'auto-validation']
    if args.decomposition == '2d':
        command += ['--grid-rows', str(best['grid_rows']),
                    '--grid-cols', str(best['grid_cols']), '--summa-bk', str(best['summa_bk'])]
    print('AUTO validation: ' + ' '.join(command), flush=True)
    completed = subprocess.run(command, cwd=root)
    if completed.returncode:
        report['validation'] = {'status': 'FAILED', 'exit_code': completed.returncode}
        return 2
    candidates = sorted((p for p in (validation_dir / 'runs').glob('*/benchmark.json')),
                        key=lambda p: p.stat().st_mtime_ns, reverse=True)
    matched = None
    for path in candidates:
        data = json.loads(path.read_text())
        a = data.get('command_args', {})
        if (a.get('label') != 'auto-validation' or a.get('size') != args.size
                or a.get('ranks') != args.ranks or a.get('decomposition') != args.decomposition
                or a.get('libraries') != [best['library']] or data.get('failures')):
            continue
        if args.decomposition == '2d' and any(a.get(k) != best[k] for k in ('grid_rows', 'grid_cols', 'summa_bk')):
            continue
        matched = (path, data)
        break
    if matched is None:
        report['validation'] = {'status': 'FAILED', 'reason': 'No matching benchmark report'}
        return 2
    path, data = matched
    rows = [r for r in data['results'] if r.get('library') == best['library'] and r.get('validated') is True]
    if len(rows) != 1:
        report['validation'] = {'status': 'FAILED', 'reason': 'No unique validated result'}
        return 2
    measured = float(rows[0]['median_s'])
    if not math.isfinite(measured) or measured <= 0:
        report['validation'] = {'status': 'FAILED', 'reason': 'Invalid measured median'}
        return 2
    predicted = best['predicted_s']
    report['validation'] = {'status': 'PASS', 'median_s': measured,
                            'predicted_s': predicted, 'absolute_error_s': abs(measured-predicted),
                            'relative_error': abs(measured-predicted)/measured,
                            'benchmark_report': str(path)}
    print(f'AUTO validation: predicted={predicted:.6f}s measured={measured:.6f}s '
          f'error={100*abs(measured-predicted)/measured:.2f}%', flush=True)
    if not args.auto_feedback:
        return 0
    candidate = next((c for c in model['candidates'] if key(c) == key(best)), None)
    if candidate is None:
        report['feedback'] = {'status': 'SKIPPED', 'reason': 'Candidate not found'}
        return 0
    # Preserve previous observations; aggregate repeated sizes at prediction time.
    candidate['measurements'].append({'size': args.size, 'median_s': measured,
                                      'source': str(path), 'timestamp_utc':
                                      datetime.datetime.now(datetime.timezone.utc).isoformat()})
    model['sizes'] = sorted(set(model.get('sizes', []) + [args.size]))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    backup = model_path.with_suffix('.json.bak')
    shutil.copy2(model_path, backup)
    temp = model_path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(model, indent=2) + '\n')
    temp.replace(model_path)
    report['feedback'] = {'status': 'SAVED', 'model': str(model_path), 'backup': str(backup)}
    return 0


def auto(args):
    root = Path(__file__).resolve().parents[1]
    model_path = resolve_model(root, args.decomposition, args.ranks, getattr(args, 'auto_model', None))
    if not model_path.is_file():
        print('AUTO model missing. Run: python3 -m autotuning.predictive calibrate --help', file=sys.stderr)
        return 2
    try:
        model = load_model(model_path)
        actual = fingerprint(root, args.decomposition, args.ranks, args.build_dir)
        if not compatible_signature(model['signature'], actual):
            raise ValueError('Model does not match this GPU architecture/binary/allocation; recalibrate')
        grid = (args.grid_rows, args.grid_cols) if args.grid_rows and args.grid_cols else None
        ranking, chosen_model = select_predictions(
            model, args.size, args.ranks, args.decomposition, grid,
            getattr(args, 'prediction_model', 'extrapolation'))
    except (ValueError, ImportError) as exc:
        print(f'AUTO: {exc}', file=sys.stderr)
        return 2
    best = ranking[0]
    mpi = next((r for r in ranking if r['library'] == 'MMM' and
                (r['grid_rows'], r['grid_cols'], r['summa_bk']) ==
                (best['grid_rows'], best['grid_cols'], best['summa_bk'])), None)
    speedup = mpi['predicted_s']/best['predicted_s'] if mpi else None
    report = {'size': args.size, 'ranks': args.ranks, 'decomposition': args.decomposition,
              'architecture_id': actual['id'], 'best': best, 'ranking': ranking,
              'speedup_vs_MPI_predicted': speedup,
              'prediction_model': chosen_model, 'calibration_file': str(model_path),
              'status': 'PREDICTED', 'measured': False}
    out = Path(args.output) / 'autotuning_prediction.json'
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    if getattr(args, 'auto_validate', False):
        status = validate_and_feedback(args, best, report, model, model_path, root)
        report['status'] = 'VALIDATED' if status == 0 else 'VALIDATION_FAILED'
    else:
        status = 0
    out.write_text(json.dumps(report, indent=2)+'\n')
    print('\n'+'='*48+'\n         MGXBench - AUTOTUNING REPORT\n'+'='*48)
    print(f'Prediction Model  : {chosen_model}\nConfiguration:\n  Matrix Size       : {args.size}\n  GPUs              : {args.ranks}'
          f'\n  Decomposition     : {args.decomposition.upper()}')
    if args.decomposition == '2d':
        print(f'  Grid              : {best["grid_rows"]} x {best["grid_cols"]}')
    print(f'\nBest Predicted Configuration:\n  Communication API : {NAMES[best["library"]]}'
          f'\n  Implementation    : {best["library"]}'
          f'\n  SUMMA BK          : {best["summa_bk"] if args.decomposition == "2d" else "N/A"}'
          f'\n  Predicted Time    : {best["predicted_s"]:.6f} s'
          f'\n  Speedup vs MPI    : {f"{speedup:.2f}x" if speedup else "N/A"}'
          f'\n  Improvement       : {f"{(1-1/speedup)*100:.2f}%" if speedup else "N/A"}'
          f'\n  Extrapolation     : {"YES" if best["extrapolated"] else "NO"}'
          f'\n\nStatus: PREDICTED (not measured)\nReport: {out}\n'+'='*48)
    if getattr(args, 'auto_validate', False):
        print('Validation status:', report.get('validation', {}).get('status', 'UNKNOWN'))
        if report.get('feedback'):
            print('Feedback status:', report['feedback']['status'])
    return status


def validate_grid_args(args, parser):
    """Validate explicit SUMMA grid options without changing auto-grid defaults."""
    if getattr(args, 'ranks', 0) <= 0:
        parser.error('--ranks must be a positive integer')
    rows = getattr(args, 'grid_rows', None)
    cols = getattr(args, 'grid_cols', None)
    if (rows is None) != (cols is None):
        parser.error('--grid-rows and --grid-cols must be supplied together')
    if rows is not None:
        if args.decomposition != '2d':
            parser.error('--grid-rows/--grid-cols require --decomposition 2d')
        if rows <= 0 or cols <= 0 or rows * cols != args.ranks:
            parser.error('--grid-rows * --grid-cols must equal --ranks (positive integers)')
        if args.command == 'calibrate' and args.grids:
            parser.error('Use either --grids or --grid-rows/--grid-cols, not both')
        if args.command == 'auto' and (args.size % rows or args.size % cols):
            parser.error('--size must be divisible by both grid dimensions')
    if args.command == 'calibrate' and args.grids:
        for item in args.grids:
            try:
                parts = item.lower().split('x')
                if len(parts) != 2:
                    raise ValueError
                r, c = map(int, parts)
                if r <= 0 or c <= 0 or r * c != args.ranks:
                    raise ValueError
            except ValueError:
                parser.error(f'Invalid --grids entry {item!r}; expected RxC with R*C=--ranks')


def main():
    p = argparse.ArgumentParser(description='Calibrate MGXBench AUTO using short measured runs')
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('calibrate')
    c.add_argument('--project-root', default='.')
    c.add_argument('--decomposition', choices=('1d','2d'), required=True)
    c.add_argument('--ranks', type=int, required=True)
    c.add_argument('--sizes', type=int, nargs='+', default=[2048,4096,8192])
    c.add_argument('--libraries', nargs='+', choices=LIBRARIES, default=list(LIBRARIES))
    c.add_argument('--bks', type=int, nargs='+', default=[128,256])
    c.add_argument('--grids', nargs='+', default=None, help='e.g. 2x4 4x2')
    c.add_argument('--grid-rows', type=int, default=None, help='Explicit 2D SUMMA grid rows')
    c.add_argument('--grid-cols', type=int, default=None, help='Explicit 2D SUMMA grid columns')
    c.add_argument('--warmup', type=int, default=1)
    c.add_argument('--repeats', type=int, default=3)
    c.add_argument('--launcher', default='mpirun')
    c.add_argument('--launcher-args', default='')
    c.add_argument('--build-dir', default='build')
    c.add_argument('--output', default='results/auto_calibration')
    c.add_argument('--model', default=None)
    r = sub.add_parser('recover', help='Rebuild model from existing calibration reports')
    r.add_argument('--project-root', default='.')
    r.add_argument('--decomposition', choices=('1d', '2d'), required=True)
    r.add_argument('--ranks', type=int, required=True)
    r.add_argument('--input', default='results/auto_calibration/runs')
    r.add_argument('--model', default=None)
    r.add_argument('--build-dir', default='build')
    r.add_argument('--assume-compatible', action='store_true',
                   help='Confirm historical runs used the current GPU/binary/allocation')
    a_parser = sub.add_parser('auto', help='Predict the fastest calibrated configuration')
    a_parser.add_argument('--decomposition', choices=('1d', '2d'), required=True)
    a_parser.add_argument('--ranks', type=int, required=True)
    a_parser.add_argument('--size', type=int, required=True)
    a_parser.add_argument('--grid-rows', type=int, default=None, help='Filter to a specific SUMMA grid')
    a_parser.add_argument('--grid-cols', type=int, default=None, help='Filter to a specific SUMMA grid')
    a_parser.add_argument('--prediction-model', choices=('extrapolation', 'ml', 'auto'),
                          default='extrapolation')
    a_parser.add_argument('--build-dir', default='build')
    a_parser.add_argument('--auto-model', default=None)
    a_parser.add_argument('--auto-validate', action='store_true')
    a_parser.add_argument('--auto-feedback', action='store_true')
    a_parser.add_argument('--warmup', type=int, default=2)
    a_parser.add_argument('--repeats', type=int, default=5)
    a_parser.add_argument('--launcher', default='mpirun')
    a_parser.add_argument('--launcher-args', default='')
    a_parser.add_argument('--timeout', type=int, default=600)
    a_parser.add_argument('--output', default='results/auto_prediction')
    a = p.parse_args()
    validate_grid_args(a, p)
    if a.command == 'auto' and a.auto_feedback and not a.auto_validate:
        p.error('--auto-feedback requires --auto-validate')
    if a.command == 'auto' and a.size <= 0:
        p.error('--size must be positive')
    if a.command == 'calibrate':
        calibrate(a)
    elif a.command == 'recover':
        recover_calibration(a)
    elif a.command == 'auto':
        return auto(a)

if __name__ == '__main__':
    main()



