#!/usr/bin/env python3
"""MGXBench — Multi-GPU Execution Benchmark. Preserves the CUDA/MPI application."""
import argparse,csv,json,math,os,pathlib,re,shlex,statistics,subprocess,sys,time
from configuration.experiment import collect_environment, new_run, sample_statistics
LIBRARY_NAMES = {
    'MMM': 'MPI-SYNC',
    'YYY': 'MPI-CHUNKED-ASYNC',
    'CCC': 'CUDA-AWARE-SYNC',
    'XXX': 'CUDA-AWARE-ASYNC',
    'NNN': 'NCCL-ASYNC',
    'WWW': 'NVSHMEM-CHUNKED-SYNC',
    'SSS': 'NVSHMEM-ASYNC',
}


def print_summary(results, failures, args, dest):
    width = 91
    print('\n' + '=' * width)
    print('MGXBench | Multi-GPU Execution Benchmark'.center(width))
    print('=' * width)
    print(f'Workload: Distributed Matrix Multiplication | Decomposition: {args.decomposition.upper()}')
    print(f'Matrix: {args.size} x {args.size} | MPI ranks / GPUs: {args.ranks} | Warmups: {args.warmup} | Repeats: {args.repeats}')
    print('-' * width)
    print(f'{"Rank":>4}  {"Code":<4}  {"Communication backend":<22} {"Mean (s)":>10} {"GFLOP/s":>12} {"Speedup":>9} {"Status":>8}')
    print('-' * width)
    ranked = sorted(results, key=lambda r: r['mean_s'])
    for rank, r in enumerate(ranked, 1):
        speedup = f"{r['speedup_vs_MMM']:.2f}x" if r['speedup_vs_MMM'] is not None else 'N/A'
        print(f"{rank:>4}  {r['library']:<4}  {LIBRARY_NAMES.get(r['library'], 'Unknown'):<22} "
              f"{r['mean_s']:>10.6f} {r['gflops']:>12.2f} {speedup:>9} {'PASS':>8}")
    if not results:
        print('No successful measurements.')
    print('-' * width)
    if ranked:
        best = ranked[0]
        print(f"Fastest backend: {best['library']} ({LIBRARY_NAMES.get(best['library'], 'Unknown')}) "
              f"| {best['mean_s']:.6f} s | {best['gflops']:.2f} GFLOP/s")
    print(f'Validated: {len(results)}/{len(args.libraries)} | Failed: {len(failures)}')
    print('Metric: application-reported computation/communication time (MPI_Wtime).')
    print('GFLOP/s: theoretical 2*N^3 / time; not a hardware counter.')
    print(f'Results: {dest / "benchmark.csv"} | {dest / "benchmark.json"}')
    print('=' * width)

PARSER=re.compile(r'BENCHMARK_RESULT\s+([^\r\n]+)')
def parse_result(output):
    matches=PARSER.findall(output)
    if len(matches)!=1: raise ValueError('Expected exactly one BENCHMARK_RESULT record; got '+str(len(matches)))
    values=dict(x.split('=',1) for x in matches[0].split())
    return values

def main():
    ap=argparse.ArgumentParser(description='MGXBench — Multi-GPU Execution Benchmark')
    ap.add_argument('--decomposition',choices=['1d','2d'],required=True)
    ap.add_argument('--size',type=int,required=True)
    ap.add_argument('--ranks',type=int,required=True)
    ap.add_argument('--libraries',nargs='+',default=['MMM','YYY','CCC','XXX','NNN','WWW','SSS'])
    ap.add_argument('--warmup',type=int,default=2)
    ap.add_argument('--warmup-mode', choices=['external','internal'], default='external', help='Internal warmup is supported by 1D only')
    ap.add_argument('--retries', type=int, default=0, help='Retries for transient launch/timeout failures (not validation failures)')
    ap.add_argument('--repeats',type=int,default=5)
    ap.add_argument('--launcher',default='mpirun')
    ap.add_argument('--launcher-args',default='')
    ap.add_argument('--build-dir',default='build')
    ap.add_argument('--output',default='results')
    ap.add_argument('--timeout',type=int,default=600)
    ap.add_argument('--plot', action='store_true', help='Generate PNG/PDF and HTML report after measurement')
    ap.add_argument('--inner-loops', type=int, default=1, help='1D internal iterations; 2D supports 1 only')
    ap.add_argument('--environment', help='JSON hardware/topology/software metadata shared across a campaign')
    ap.add_argument('--label', default='default', help='Experiment/campaign label')
    ap.add_argument('--device-map', choices=['local','explicit'], default='local')
    args=ap.parse_args()
    if args.retries < 0: ap.error('retries must be nonnegative')
    if args.warmup_mode == 'internal' and args.decomposition != '1d': ap.error('internal warmup currently supports 1D only')
    if args.inner_loops < 1 or (args.decomposition == '2d' and args.inner_loops != 1):
        ap.error('inner-loops must be positive; 2D supports 1')
    if any(lib not in LIBRARY_NAMES for lib in args.libraries) or len(set(args.libraries)) != len(args.libraries):
        ap.error('select distinct supported library codes')
    if args.size<=0 or args.ranks<=0 or args.repeats<=0 or args.warmup<0:ap.error('invalid size/ranks/repeats/warmup')
    if args.size%args.ranks and args.decomposition=='1d':ap.error('1D requires size divisible by ranks')
    if args.decomposition=='2d' and math.isqrt(args.ranks)**2!=args.ranks:ap.error('2D SUMMA requires perfect square rank count')
    if args.decomposition=='2d' and args.size%math.isqrt(args.ranks):ap.error('2D requires size divisible by sqrt(ranks)')
    for lib in args.libraries:
        if len(lib)!=3 or any(c not in 'MYCXNSW' for c in lib):ap.error('invalid three-character library code: '+lib)
    exe=pathlib.Path(args.build_dir).resolve()/('mm_1d' if args.decomposition=='1d' else 'mm_2d')
    if not exe.is_file():ap.error('missing executable: '+str(exe)+'; first run cmake -S . -B build and cmake --build build -j 4, and check the build output')
    environment = collect_environment(args.environment, args.build_dir)
    dest, run_id = new_run(args.output)
    child_env = os.environ.copy()
    child_env['MGXBENCH_INNER_LOOPS'] = str(args.inner_loops)
    child_env['MGXBENCH_INTERNAL_WARMUP'] = str(args.warmup if args.warmup_mode == 'internal' else 0)
    results=[]; failures=[]
    print('\nMGXBench — Multi-GPU Execution Benchmark')
    print(f'Running {args.decomposition.upper()} matrix multiplication, N={args.size}, GPUs={args.ranks}\n', flush=True)
    for lib in args.libraries:
        samples=[]; runs=[]
        for rep in range(-args.warmup if args.warmup_mode == 'external' else 0,args.repeats):
            # Open MPI MPMD syntax preserves explicit rank->GPU mapping in original code.
            command=shlex.split(args.launcher)+shlex.split(args.launcher_args)
            for rank in range(args.ranks):
                if rank:command+=[':']
                command+=['-np','1',str(exe),('auto' if args.device_map == 'local' else str(rank)),str(args.size),lib]
            stage = f'warmup {rep + args.warmup + 1}/{args.warmup}' if rep < 0 else f'run {rep + 1}/{args.repeats}'
            print(f'[{lib:3}] {LIBRARY_NAMES.get(lib, lib):<22} {stage:<15}', flush=True)
            output = ''
            try:
                for attempt in range(args.retries + 1):
                    try:
                        proc=subprocess.run(command,capture_output=True,text=True,timeout=args.timeout, env=child_env)
                    except subprocess.TimeoutExpired:
                        if attempt == args.retries: raise
                        print(f'  timeout; retry {attempt + 1}/{args.retries}', file=sys.stderr)
                        continue
                    output=proc.stdout+'\n'+proc.stderr
                    if proc.returncode != 0 and 'BENCHMARK_RESULT' not in output and attempt < args.retries:
                        print(f'  launcher failure; retry {attempt + 1}/{args.retries}', file=sys.stderr)
                        continue
                    break
                record=parse_result(output)
                (dest / f'{lib}_{rep}.log').write_text(output)
                valid=proc.returncode==0 and record['valid']=='1'
                if not valid:raise RuntimeError(f'validation/exit failed: exit={proc.returncode}, valid={record["valid"]}')
                if int(record['size'])!=args.size or int(record['ranks'])!=args.ranks or record['library']!=lib:
                    raise RuntimeError('metadata mismatch')
                if record.get('decomposition', '').lower() != args.decomposition or int(record.get('loops', 0)) != args.inner_loops:
                    raise RuntimeError('decomposition/inner-loop metadata mismatch')
                elapsed=float(record['seconds'])
                if not math.isfinite(elapsed) or elapsed<=0:raise RuntimeError('invalid time')
                if rep>=0:
                    samples.append(elapsed)
                    runs.append({'iteration':rep+1,'seconds':elapsed,'validated':True})
            except Exception as exc:
                log=dest/f'{args.decomposition}_{lib}_failed_{rep}.log'
                log.write_text((' '.join(command)+'\n'+locals().get('output','')+'\n'+str(exc))[-100000:])
                failures.append({'library':lib,'iteration':rep,'error':str(exc),'log':str(log)})
                print('ERROR:',exc,file=sys.stderr)
                break
        if len(samples)!=args.repeats:continue
        avg=statistics.mean(samples)
        sorted_samples=sorted(samples)
        def pct(q):
            pos=(len(sorted_samples)-1)*q; low=int(pos);high=min(low+1,len(sorted_samples)-1)
            return sorted_samples[low]+(sorted_samples[high]-sorted_samples[low])*(pos-low)
        # 2*N^3 FLOPs globally for one dense matrix multiplication.
        gflops=(2.0*args.size**3)/(avg*1e9)
        results.append({'decomposition':args.decomposition,'library':lib,'size':args.size,
            'ranks':args.ranks,'mean_s':avg,'median_s':statistics.median(samples),
            'p95_s':pct(.95),'stddev_s':statistics.pstdev(samples),
            'gflops':gflops,'speedup_vs_MMM':None,'validated':True,
            'samples_s':samples,'runs':runs,'original_inner_loops':args.inner_loops, 'seconds_semantics':'mean_seconds_per_multiplication', **sample_statistics(samples)})
    baseline=next((r['mean_s'] for r in results if r['library']=='MMM'),None)
    for r in results:
        if baseline:r['speedup_vs_MMM']=baseline/r['mean_s']
    metadata={'run_id':run_id, 'label':args.label, 'environment':environment, 'schema_version':2,'application':'distributed-multi-gpu-mm-domain-decomposition',
        'measurement':'original MPI_Wtime computation/communication region; not process startup',
        'note':'Independent measured process launches; warmups excluded. 1D inner loops configurable; 2D one iteration. seconds is mean time per multiplication.',
        'bandwidth':'not reported: actual communication bytes vary with backend and decomposition',
        'timestamp_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'benchmark_name':'MGXBench', 'benchmark_full_name':'Multi-GPU Execution Benchmark',
        'command_args':vars(args),'results':results,'failures':failures}
    for row in results:
        row.update(run_id=run_id, label=args.label, environment_id=environment['id'])
    (dest/'benchmark.json').write_text(json.dumps(metadata,indent=2))
    cols=['run_id','label','environment_id','cv_pct','ci95_low_s','ci95_high_s','decomposition','library','size','ranks','mean_s','median_s','p95_s','stddev_s','gflops','speedup_vs_MMM','validated','original_inner_loops','seconds_semantics']
    with (dest/'benchmark.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:r[k] for k in cols} for r in results)
    print_summary(results, failures, args, dest)
    if args.plot and results:
        plotted = subprocess.run([sys.executable, str(pathlib.Path(__file__).parent / 'configuration' / 'plot_benchmark.py'), '--input', str(dest), '--output', str(dest/'plots')])
        if plotted.returncode:
            print('Measurements saved; visualization failed. Install configuration/requirements-viz.txt and retry configuration/plot_benchmark.py.', file=sys.stderr)
            return 3
    if failures or not results:return 2
    return 0
if __name__=='__main__':sys.exit(main())
