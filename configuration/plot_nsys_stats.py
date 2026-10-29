#!/usr/bin/env python3
"""MGXBench 5.0: headless nsys stats CSV export and Matplotlib kernel summaries."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

REPORTS = ('gpukernsum', 'cuda_gpu_kern_sum')


def columns(header):
    names = [str(c).strip() for c in header]
    name = next((c for c in names if c.lower() in ('name', 'kernel name')), None)
    percent = next((c for c in names if 'time' in c.lower() and '%' in c), None)
    return names, name, percent


def read_summary(path):
    """Find one kernel-summary CSV table, allowing leading Nsight console messages."""
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    for index, line in enumerate(lines):
        header = next(csv.reader([line]), [])
        names, name, percent = columns(header)
        if name and percent:
            break
    else:
        raise ValueError('Colunas Name/Kernel Name e Time (%) não encontradas. Exporte um resumo de kernels em CSV, não a tabela alinhada do terminal.')
    records=[]
    for record in csv.reader(io.StringIO('\n'.join(lines[index+1:]))):
        if not record or not any(x.strip() for x in record):
            if records:break
            continue
        if len(record)!=len(names):
            if records:break
            raise ValueError('Linha CSV com número de colunas inesperado')
        if columns(record)[1]:
            raise ValueError('Use apenas um relatório CSV por entrada')
        records.append(record)
    df=pd.DataFrame(records,columns=names)
    if df.empty:raise ValueError('Resumo sem kernels; verifique se a captura contém atividade CUDA')
    numeric=pd.to_numeric(df[percent].astype(str).str.replace('%','',regex=False).str.strip(),errors='coerce')
    valid=numeric.notna() & numeric.map(lambda x: math.isfinite(x) and 0<=x<=100)
    valid &= df[name].astype(str).str.strip().ne('')
    discarded=int((~valid).sum())
    df=df.loc[valid].copy();df[percent]=numeric.loc[valid]
    if df.empty:raise ValueError('O CSV não contém percentuais válidos de kernels')
    return df,name,percent,discarded


def export_stats(source, output, report='auto', nsys='nsys', timeout=600):
    executable=shutil.which(nsys)
    if not executable:raise ValueError(f'Executável {nsys!r} não encontrado. Instale/carregue Nsight CLI ou forneça um CSV já exportado.')
    attempts=REPORTS if report=='auto' else (report,)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    log=output/f'{source.stem}_nsys_stats.log'
    transcript=[]
    try:
        for selected in attempts:
            with tempfile.TemporaryDirectory(prefix='mgx-stats-') as tmp:
                prefix=Path(tmp)/source.stem
                command=[executable,'stats','--report',selected,'--format','csv','--output',str(prefix),str(source.resolve())]
                transcript.append('COMMAND: '+subprocess.list2cmdline(command))
                try:
                    proc=subprocess.run(command,capture_output=True,text=True,timeout=timeout)
                except subprocess.TimeoutExpired as exc:
                    raise ValueError(f'nsys stats excedeu {timeout}s; consulte {log}') from exc
                transcript.extend([proc.stdout,proc.stderr])
                candidates=list(Path(tmp).glob('*.csv'))
                if proc.returncode==0 and len(candidates)==1:
                    # Validate before installing output; no table is inferred from an empty report.
                    read_summary(candidates[0])
                    destination=output/f'{source.stem}_{selected}.csv'
                    shutil.copyfile(candidates[0],destination)
                    return destination,selected
                message=(proc.stdout+'\n'+proc.stderr).lower()
                unknown=any(x in message for x in ('not found','not be found','unknown','unrecognized','invalid report','does not exist','not exist','no such')) and any(x in message for x in ('report','gpukernsum','cuda_gpu_kern_sum'))
                if report=='auto' and unknown and selected!=attempts[-1]:continue
                raise ValueError(f'Falha ao exportar {selected}; consulte {log}. Confira nsys --version e nsys stats --help.')
        raise ValueError(f'Nenhum relatório de kernels compatível; consulte {log}')
    finally:
        log.write_text('\n'.join(transcript))


def plot_summary(path, top=15, output_dir=None, report=None):
    if top<1:raise ValueError('--top deve ser positivo')
    path=Path(path)
    df,name,percent,discarded=read_summary(path)
    selected=df.nlargest(top,percent).sort_values(percent)
    output=Path(output_dir) if output_dir else path.parent
    output.mkdir(parents=True,exist_ok=True)
    # Wrapping changes labels only; the source/normalized CSV retain complete names.
    labels=['\n'.join(textwrap.wrap(str(x),width=80,break_long_words=True)) for x in selected[name]]
    height=max(5,sum(max(1,label.count('\n')+1) for label in labels)*.35+2)
    fig,ax=plt.subplots(figsize=(14,height))
    positions=range(len(selected));bars=ax.barh(positions,selected[percent],color='#2878B5')
    ax.set_yticks(list(positions));ax.set_yticklabels(labels,fontsize=9)
    ax.bar_label(bars,fmt='%.2f%%',padding=4)
    ax.set_xlabel('Participação no tempo acumulado dos kernels (%)')
    ax.set_ylabel('Kernel');ax.set_title(f'Nsight Systems — {path.stem}')
    ax.grid(axis='x',linestyle='--',alpha=.3);ax.set_axisbelow(True);ax.margins(x=.15)
    fig.text(.5,.01,'Tempo acumulado dos kernels; não é tempo total da aplicação nem medida de overlap.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1))
    outputs=[]
    for extension in ('png','pdf'):
        target=output/f'{path.stem}_plot.{extension}'
        fig.savefig(target,dpi=300,bbox_inches='tight');outputs.append(target)
    plt.close(fig)
    df.to_csv(output/f'{path.stem}_normalized.csv',index=False)
    (output/f'{path.stem}_plot.json').write_text(json.dumps({'input':str(path.resolve()),'report':report,'valid_rows':len(df),'discarded_rows':discarded,'plotted_rows':len(selected),'displayed_percent_sum':float(selected[percent].sum()),'note':'Original Nsight percentages retained; top selection is not renormalized. Kernel sums may include concurrent work. This does not measure wall time, GPU utilization or overlap.'},indent=2))
    return outputs,discarded


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_file',help='Resumo CSV ou captura .qdrep/.nsys-rep/.sqlite')
    parser.add_argument('--top',type=int,default=15)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--report',choices=('auto',)+REPORTS,default='auto')
    parser.add_argument('--nsys',default='nsys',help='Executável Nsight CLI')
    parser.add_argument('--timeout',type=int,default=600)
    args=parser.parse_args()
    if args.top<1 or args.timeout<1:parser.error('--top e --timeout devem ser positivos')
    source=Path(args.input_file)
    if not source.is_file():parser.error(f'Arquivo não encontrado: {source}')
    selected=None
    try:
        if source.suffix.lower() in ('.qdrep','.nsys-rep','.sqlite'):
            source,selected=export_stats(source,args.output_dir or source.parent/'stats',args.report,args.nsys,args.timeout)
            print(f'CSV exportado: {source}')
        elif source.suffix.lower()!='.csv':raise ValueError('Use .csv, .qdrep, .nsys-rep ou .sqlite')
        outputs,discarded=plot_summary(source,args.top,args.output_dir,selected)
        if discarded:print(f'Aviso: {discarded} linha(s) sem percentual/nome válido foram descartadas.')
        for output in outputs:print(f'Gráfico salvo: {output}')
    except (ValueError,OSError,pd.errors.ParserError) as exc:parser.exit(2,f'Erro: {exc}\n')

if __name__=='__main__':main()
