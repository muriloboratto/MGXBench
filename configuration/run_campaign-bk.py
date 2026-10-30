#!/usr/bin/env python3
"""Execute reproducible size sweeps, strong and dense-MM weak scaling campaigns."""
import argparse
from pathlib import Path
import math
import subprocess
import sys

def configurations(decompositions, sizes, ranks, mode, base_size):
    configs=[]
    for d in decompositions:
        for p in ranks:
            if p<=0:raise ValueError('Ranks must be positive')
            if d=='2d' and math.isqrt(p)**2!=p:
                print(f'Skipping 2D p={p}: grid must be square',file=sys.stderr);continue
            divisor=p if d=='1d' else math.isqrt(p)
            requested=[base_size*p**(1/3)] if mode=='weak' else sizes
            for value in requested:
                n=max(divisor, round(value/divisor)*divisor)
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
    ap.add_argument('--output',default='results/campaign')
    ap.add_argument('--label',default='campaign')
    args,extra=ap.parse_known_args()
    if any(x.split('=')[0] == '--dry-run' for x in extra):
        ap.error('--dry-run foi removido; campanhas executam os benchmarks imediatamente')
    forbidden=('--size','--ranks','--decomposition','--output','--label')
    if any(x.split('=')[0] in forbidden for x in extra):ap.error('Conflicting forwarded benchmark argument')
    if args.base_size<=0 or any(n<=0 for n in args.sizes):ap.error('Sizes must be positive')
    if args.mode=='strong' and len(args.sizes)!=1:ap.error('Strong scaling requires exactly one --sizes value')
    try:configs=configurations(args.decompositions,args.sizes,args.ranks,args.mode,args.base_size)
    except ValueError as exc:ap.error(str(exc))
    failures=0
    for d,n,p in configs:
        command=[sys.executable,str(Path(__file__).resolve().parents[1] / 'benchmark.py'),'--decomposition',d,'--size',str(n),'--ranks',str(p),'--output',args.output,'--label',('weak:' if args.mode=='weak' else '')+args.label]+extra
        print(subprocess.list2cmdline(command),flush=True)
        if subprocess.run(command).returncode:failures+=1
    proc=subprocess.run([sys.executable,str(Path(__file__).with_name('plot_benchmark.py')),'--input',args.output,'--output',str(Path(args.output)/'report')])
    if proc.returncode:failures+=1
    return 2 if failures else 0

if __name__=='__main__':sys.exit(main())
