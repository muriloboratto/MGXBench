#!/usr/bin/env python3
"""Execute reproducible size sweeps, strong and dense-MM weak scaling campaigns."""
import argparse
import csv
from pathlib import Path
import math
import subprocess
import sys

def weak_scaling_size(base_size, ranks, decomposition='1d', size_alignment=4):
    """Return (theoretical_N, aligned_N) for compute-based weak scaling.

    Alignment satisfies both the user-selected multiple and the process grid.
    """
    if base_size <= 0 or ranks <= 0 or size_alignment <= 0:
        raise ValueError('Base size, ranks and alignment must be positive')
    if decomposition == '1d':
        grid_divisor = ranks
    elif decomposition == '2d':
        grid_divisor = math.isqrt(ranks)
        if grid_divisor ** 2 != ranks:
            raise ValueError('2D decomposition requires a square process grid')
    else:
        raise ValueError('Invalid decomposition')
    theoretical = base_size * ranks ** (1.0 / 3.0)
    actual = max(divisor, divisor * math.floor(theoretical / divisor + 0.5))
    return theoretical, actual


def configurations(decompositions, sizes, ranks, mode, base_size, size_alignment=4):
    configs=[]
    for d in decompositions:
        for p in ranks:
            if p<=0:raise ValueError('Ranks must be positive')
            if d=='2d' and math.isqrt(p)**2!=p:
                print(f'Skipping 2D p={p}: grid must be square',file=sys.stderr);continue
            requested = [base_size * p ** (1 / 3)] if mode == 'weak' else sizes
            for value in requested:
                if mode == 'weak':
                    _, n = weak_scaling_size(base_size, p, d, size_alignment)
                else:
                    n = max(divisor, divisor * math.floor(value / divisor + 0.5))
                if mode!='weak' and n!=value:
                    print(f'Skipping {d} N={value}, p={p}: divisibility',file=sys.stderr);continue
                if n<=0:raise ValueError('Matrix size must be positive')
                if mode=='weak' and abs(n**3/p/base_size**3-1)>.025:
                    raise ValueError('Weak-scaling rounding exceeds 2.5%; choose a larger base size')
                configs.append((d,n,p))
    if not configs:raise ValueError('No compatible configurations')
    return configs


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--decompositions',nargs='+',choices=['1d','2d'],default=['1d','2d'])
    ap.add_argument('--sizes',nargs='+',type=int,default=[2048,4096,8192,16384,32768])
    ap.add_argument('--ranks',nargs='+',type=int,default=[1,4])
    ap.add_argument('--mode',choices=['sweep','strong','weak'],default='sweep')
    ap.add_argument('--base-size',type=int,default=4096)
    ap.add_argument('--size-alignment',type=int,default=4,help='Minimum N alignment; combined with process-grid divisibility (default: 4)')
    ap.add_argument('--output',default='results/campaign')
    ap.add_argument('--label',default='campaign')
    args,extra=ap.parse_known_args()
    if any(x.split('=')[0] == '--dry-run' for x in extra):
        ap.error('--dry-run foi removido; campanhas executam os benchmarks imediatamente')
    forbidden=('--size','--ranks','--decomposition','--output','--label')
    if any(x.split('=')[0] in forbidden for x in extra):ap.error('Conflicting forwarded benchmark argument')
    if args.base_size<=0 or any(n<=0 for n in args.sizes) or args.size_alignment<=0:ap.error('Sizes and size alignment must be positive')
    if args.mode=='strong' and len(args.sizes)!=1:ap.error('Strong scaling requires exactly one --sizes value')
    try:configs=configurations(args.decompositions,args.sizes,args.ranks,args.mode,args.base_size,args.size_alignment)
    except ValueError as exc:ap.error(str(exc))
    # Campaign manifest records the theoretical vs. actual sizes independently
    # of the benchmark's own output schema.
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / f'{args.label}-size-manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            'mode', 'decomposition', 'ranks', 'N_theoretical',
            'N_actual', 'work_per_gpu_ratio', 'size_alignment'])
        writer.writeheader()
        for d, n, p in configs:
            theoretical = args.base_size * p ** (1.0 / 3.0) if args.mode == 'weak' else n
            writer.writerow({
                'mode': args.mode, 'decomposition': d, 'ranks': p,
                'N_theoretical': f'{theoretical:.6f}', 'N_actual': n,
                'work_per_gpu_ratio': f'{(n / args.base_size) ** 3 / p:.6f}' if args.mode == 'weak' else '',
                'size_alignment': args.size_alignment})
    print(f'Size manifest: {manifest}', flush=True)
    failures=0
    for d,n,p in configs:
        command=[sys.executable,str(Path(__file__).resolve().parents[1] / 'benchmark.py'),'--decomposition',d,'--size',str(n),'--ranks',str(p),'--output',args.output,'--label',('weak:' if args.mode=='weak' else '')+args.label]+extra
        print(subprocess.list2cmdline(command),flush=True)
        if subprocess.run(command).returncode:failures+=1
    proc=subprocess.run([sys.executable,str(Path(__file__).with_name('plot_benchmark.py')),'--input',args.output,'--output',str(Path(args.output)/'report')])
    if proc.returncode:failures+=1
    return 2 if failures else 0

if __name__=='__main__':sys.exit(main())
