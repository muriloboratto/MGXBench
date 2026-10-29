#!/usr/bin/env python3
"""Generate explicitly SYNTHETIC measurements to exercise every figure family."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from configuration.plot_benchmark import generate,load_results

out=Path(sys.argv[1] if len(sys.argv)>1 else 'examples/synthetic-demo')
out.mkdir(parents=True,exist_ok=True)
rows=[]
for label in ('SYNTHETIC-demo','weak:SYNTHETIC-demo'):
    for d in ('1d','2d'):
        configs=[(1024,1),(2048,1),(1024,4),(2048,4)] if not label.startswith('weak') else [(1024,1),(2048,8)]
        # For demo weak scaling use 1D only: p=8 is invalid for a square 2D grid.
        if label.startswith('weak') and d=='2d':continue
        for n,p in configs:
            for lib,factor in [('MMM',1),('NNN',.7)]:
                t=.1*(n/1024)**3/p*factor*(1 if d=='1d' else .9)*(1+.07*(p-1))
                rows.append(dict(label=label,decomposition=d,library=lib,size=n,ranks=p,mean_s=t,stddev_s=t*.01,original_inner_loops=1,validated=True,samples_s=[t*.99,t,t*1.01]))
(out/'benchmark.json').write_text(json.dumps({'schema_version':2,'label':'SYNTHETIC-demo','environment':{'id':'SYNTHETIC-not-hardware'},'measurement':'SYNTHETIC DATA — not experimental evidence','results':rows},indent=2))
print(generate(load_results([out/'benchmark.json']),out/'plots'))
