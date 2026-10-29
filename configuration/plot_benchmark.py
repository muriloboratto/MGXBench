#!/usr/bin/env python3
"""Offline scientific figures and self-contained performance report (no network)."""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import LIBRARY_NAMES
from configuration.experiment import sample_statistics


def load_results(inputs):
    rows, seen = [], set()
    for raw in inputs:
        path = Path(raw)
        files = sorted(path.rglob('benchmark.json')) if path.is_dir() else [path]
        if path.is_dir():
            files += [f for f in sorted(path.rglob('benchmark.csv')) if not f.with_suffix('.json').exists()]
        for f in files:
            if f.resolve() in seen:
                continue
            seen.add(f.resolve())
            if f.suffix == '.json':
                doc = json.loads(f.read_text())
                if not isinstance(doc, dict) or 'results' not in doc:
                    raise ValueError(f'{f}: expected benchmark JSON object')
                entries = doc['results']
                env = doc.get('environment', {}).get('id', 'unknown-' + hashlib.sha256(str(f.parent.resolve()).encode()).hexdigest()[:8])
                label = doc.get('label', 'legacy')
            else:
                with f.open() as stream:
                    entries = list(csv.DictReader(stream))
                env, label = 'unknown-' + hashlib.sha256(str(f.parent.resolve()).encode()).hexdigest()[:8], 'legacy'
            for entry in entries:
                r = dict(entry)
                if str(r.get('validated', '')).lower() not in ('true', '1'):
                    continue
                for key in ('mean_s', 'stddev_s'):
                    r[key] = float(r[key])
                for key in ('size', 'ranks', 'original_inner_loops'):
                    r[key] = int(r.get(key, 1))
                if r['size'] <= 0 or r['ranks'] <= 0 or r['original_inner_loops'] <= 0 or not math.isfinite(r['mean_s']) or r['mean_s'] <= 0 or not math.isfinite(r['stddev_s']) or r['stddev_s'] < 0:
                    raise ValueError(f'{f}: invalid numeric measurement')
                r['decomposition'] = r['decomposition'].lower()
                r['environment_id'] = r.get('environment_id') or env
                r['label'] = r.get('label') or label
                r['run_id'] = r.get('run_id') or f.parent.name
                r['samples_s'] = r.get('samples_s', [])
                if any(not math.isfinite(float(x)) or float(x) <= 0 for x in r['samples_s']):
                    raise ValueError(f'{f}: invalid samples')
                r['source'] = str(f)
                r['gflops'] = 2*r['size']**3/(r['mean_s']*1e9)
                rows.append(r)
    if not rows:
        raise ValueError('No validated measurements found')
    return pd.DataFrame(rows)


def consolidate(df):
    """Pool independent launches only inside an identical environment/configuration."""
    keys = ['environment_id','label','original_inner_loops','decomposition','library','size','ranks']
    rows = []
    for _, group in df.groupby(keys, dropna=False):
        r = group.iloc[0].to_dict()
        if len(group) > 1 and not all(len(x) for x in group.samples_s):
            raise ValueError('Repeated CSV summaries cannot be pooled without raw samples; select a single run')
        samples = [float(s) for sample in group.samples_s for s in sample]
        if samples:
            r.update(mean_s=statistics.mean(samples), stddev_s=statistics.pstdev(samples), samples_s=samples,
                     median_s=statistics.median(samples), p95_s=float(np.percentile(samples,95)), **sample_statistics(samples))
        r['gflops'] = 2*r['size']**3/(r['mean_s']*1e9)
        r['run_count'] = len(group)
        rows.append(r)
    result = pd.DataFrame(rows)
    result['speedup_vs_MMM'] = float('nan')
    for _, group in result.groupby(['environment_id','label','original_inner_loops','decomposition','size','ranks']):
        baseline = group[group.library == 'MMM']
        if not baseline.empty:
            result.loc[group.index,'speedup_vs_MMM'] = float(baseline.iloc[0].mean_s) / group.mean_s
    return result


def save(fig, out, name, files):
    fig.tight_layout()
    for suffix in ('png','pdf'):
        fig.savefig(out/f'{name}.{suffix}', dpi=220, bbox_inches='tight')
    files.append(name+'.png')
    plt.close(fig)


def generate(df, output):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    data = consolidate(df)
    figures = []
    notes = ['Only numerically validated measurements are plotted.',
             'Bars show means ± population standard deviation. CI95 is a seeded percentile bootstrap of independent launches (2,000 resamples); small samples give uncertain intervals.',
             'GFLOP/s is effective application throughput: 2 N³ / elapsed time. No hardware-counter or isolated-kernel claim.',
             'Identical environment, label and internal-loop count are required for comparisons. Repeated matching runs pool raw launch samples.',
             '1D and 2D timed regions retain their original algorithm-specific scope. Equal internal-loop count does not make their communication work identical.',
             'Matrix-size curves describe workload growth; strong scaling fixes N. Weak scaling for dense MM holds N³/p approximately constant, not N²/p.',
             'Kernel and application totals do not identify concurrent phases. This release provides benchmark summaries and kernel statistics only.']
    for identity, family in data.groupby(['environment_id','label','original_inner_loops']):
        tag = hashlib.sha256(str(identity).encode()).hexdigest()[:10]
        for (size,ranks), group in family.groupby(['size','ranks']):
            prefix=f'{tag}_N{size}_p{ranks}'
            libs=[x for x in LIBRARY_NAMES if x in set(group.library)]
            libs += sorted(set(group.library)-set(libs))
            decomps=sorted(group.decomposition.unique())
            means=group.pivot(index='library',columns='decomposition',values='mean_s').reindex(libs)
            stds=group.pivot(index='library',columns='decomposition',values='stddev_s').reindex(libs)
            width=.8/len(decomps)
            for metric in ('execution_time','throughput','speedup'):
                fig,ax=plt.subplots(figsize=(12,6))
                drawn=False
                for j,d in enumerate(decomps):
                    y=means[d].copy()
                    errors=None
                    if metric=='execution_time':
                        errors=stds[d]
                    elif metric=='throughput':
                        y=2*size**3/(y*1e9)
                    else:
                        if 'MMM' not in means.index or pd.isna(means.loc['MMM',d]):
                            notes.append(f'{prefix}/{d}: MPI-SYNC baseline absent; speedup omitted.')
                            continue
                        y=means.loc['MMM',d]/y
                    ax.bar(np.arange(len(libs))+(j-(len(decomps)-1)/2)*width,y,width,yerr=errors,capsize=3,label=d.upper())
                    drawn=True
                if not drawn:
                    plt.close(fig);continue
                if metric=='speedup':ax.axhline(1,color='black',ls='--',lw=1)
                ax.set_xticks(range(len(libs)));ax.set_xticklabels([LIBRARY_NAMES.get(x,x) for x in libs],rotation=25,ha='right')
                ax.set_ylabel({'execution_time':'Execution time (s)','throughput':'Effective GFLOP/s','speedup':'Speedup relative to MPI-SYNC'}[metric])
                ax.set_title(f'{metric.replace("_"," ").title()} | N={size}, GPUs={ranks}\nEnvironment {identity[0]} | {identity[1]} | inner loops={identity[2]}')
                ax.legend();ax.grid(axis='y',alpha=.25)
                save(fig,out,prefix+'_'+metric,figures)
            for d,g in group.groupby('decomposition'):
                available=g[g.samples_s.map(len)>0]
                if available.empty:continue
                fig,ax=plt.subplots(figsize=(12,6))
                ax.boxplot(available.samples_s.tolist(),showmeans=True)
                ax.set_xticks(range(1,len(available)+1));ax.set_xticklabels([LIBRARY_NAMES.get(x,x) for x in available.library])
                ax.tick_params(axis='x',rotation=25);ax.set_ylabel('Launch sample time (s)')
                ax.set_title(f'{d.upper()} time distribution | N={size}, GPUs={ranks}')
                save(fig,out,prefix+'_'+d+'_time_distribution',figures)
        for (decomp,ranks),g in family.groupby(['decomposition','ranks']):
            if g['size'].nunique()<2:continue
            fig,ax=plt.subplots(figsize=(10,6))
            for lib,v in g.groupby('library'):
                v=v.sort_values('size');ax.plot(v['size'],v.mean_s,'o-',label=LIBRARY_NAMES.get(lib,lib))
            ax.set(xlabel='Matrix dimension N',ylabel='Execution time (s)',title=f'{decomp.upper()} workload growth | GPUs={ranks}',yscale='log',xscale='log')
            ax.legend(fontsize=8);ax.grid(alpha=.25)
            save(fig,out,f'{tag}_{decomp}_p{ranks}_size_scalability',figures)
        for (decomp,size),g in family.groupby(['decomposition','size']):
            if g.ranks.nunique()<2:continue
            fig,axes=plt.subplots(1,2,figsize=(13,5));drawn=False
            for lib,v in g.groupby('library'):
                v=v.sort_values('ranks');base=v[v.ranks==1]
                if base.empty:
                    notes.append(f'{tag}/{decomp}/N{size}/{lib}: no one-GPU baseline; strong scaling omitted.');continue
                speed=float(base.iloc[0].mean_s)/v.mean_s
                axes[0].plot(v.ranks,speed,'o-',label=lib)
                axes[1].plot(v.ranks,100*speed/v.ranks,'o-',label=lib);drawn=True
            if drawn:
                axes[0].plot(sorted(g.ranks.unique()),sorted(g.ranks.unique()),'k--',label='Ideal')
                axes[1].axhline(100,color='black',ls='--')
                for ax in axes:ax.set_xlabel('GPUs / MPI ranks');ax.legend();ax.grid(alpha=.25)
                axes[0].set_ylabel('Strong scaling speedup');axes[1].set_ylabel('Efficiency (%)')
                fig.suptitle(f'{decomp.upper()} strong scaling | N={size}')
                save(fig,out,f'{tag}_{decomp}_N{size}_strong_scaling',figures)
            else:plt.close(fig)
        # Weak scaling campaigns explicitly opt in using label prefix, avoiding accidental fits.
        if str(identity[1]).startswith('weak:'):
            for (decomp,lib),g in family.groupby(['decomposition','library']):
                if g.groupby('ranks').size().max()!=1 or len(g)<2:
                    notes.append(f'{tag}/{decomp}/{lib}: ambiguous weak-scaling curve; one N per rank count required.');continue
                g=g.sort_values('ranks');work=g['size'].astype(float)**3/g.ranks
                if work.max()/work.min()>1.05:
                    notes.append(f'{tag}/{decomp}/{lib}: weak-scaling work per GPU varies by >5%; omitted.');continue
                base=g[g.ranks==1]
                if base.empty:continue
                fig,ax=plt.subplots(figsize=(9,5))
                ax.plot(g.ranks,100*float(base.iloc[0].mean_s)/g.mean_s,'o-');ax.axhline(100,color='black',ls='--')
                ax.set(xlabel='GPUs',ylabel='Weak scaling efficiency (%)',title=f'{decomp.upper()} {lib} | N³/p approximately constant')
                save(fig,out,f'{tag}_{decomp}_{lib}_weak_scaling',figures)
    table=data.drop(columns=['samples_s','runs'],errors='ignore')
    table.to_csv(out/'summary.csv',index=False)
    notes=list(dict.fromkeys(notes))
    (out/'report.md').write_text('# MGXBench Performance Report\n\n'+'\n'.join('- '+n for n in notes)+'\n\n'+'\n\n'.join(f'![{f}]({f})' for f in figures))
    body=''.join('<li>'+html.escape(n)+'</li>' for n in notes)
    imgs=''.join(f'<figure><img loading="lazy" src="{html.escape(f)}"><figcaption>{html.escape(f)}</figcaption><a href="{html.escape(f[:-4])}.pdf">PDF</a></figure>' for f in figures)
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>MGXBench Performance Report</title><style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:20px;color:#142b40}img{width:100%}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:6px}.table{overflow:auto}figure{margin:30px 0}</style><h1>MGXBench Performance Report</h1><ul>'+body+'</ul><div class="table">'+table.to_html(index=False,escape=True)+'</div>'+imgs)
    return out/'report.html'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',nargs='+',required=True)
    ap.add_argument('--output',default='results/comparison')
    ap.add_argument('--size',type=int);ap.add_argument('--ranks',type=int)
    ap.add_argument('--environment-id');ap.add_argument('--label')
    args=ap.parse_args()
    try:
        data=load_results(args.input)
        for key in ('size','ranks','environment_id','label'):
            value=getattr(args,key)
            if value is not None:data=data[data[key]==value]
        if data.empty:raise ValueError('No measurements match the selected filters')
        print(generate(data,args.output))
    except (ValueError,KeyError,OSError,json.JSONDecodeError) as exc:ap.exit(2,f'Cannot generate report: {exc}\n')

if __name__=='__main__':main()
