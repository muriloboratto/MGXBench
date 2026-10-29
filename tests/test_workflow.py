import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmark import parse_result
from configuration.experiment import sample_statistics,new_run
from configuration.plot_benchmark import load_results,consolidate,generate
from configuration.run_campaign import configurations

class Workflow(unittest.TestCase):
    def test_parser_rejects_duplicate_records(self):
        with self.assertRaises(ValueError):parse_result('BENCHMARK_RESULT valid=1\nBENCHMARK_RESULT valid=1')

    def test_statistics_and_history(self):
        self.assertEqual(sample_statistics([1,1,1])['ci95_low_s'],1)
        self.assertIsNone(sample_statistics([1])['ci95_low_s'])
        with tempfile.TemporaryDirectory() as d:
            a,_=new_run(d);b,_=new_run(d);self.assertNotEqual(a,b)

    def test_campaign_constraints(self):
        self.assertEqual(configurations(['2d'],[2048],[1,2,4],'strong',4096),[('2d',2048,1),('2d',2048,4)])
        curve=configurations(['1d'],[],[1,8],'weak',1024)
        self.assertEqual(curve,[('1d',1024,1),('1d',2048,8)])

    def test_loading_filtering_pooling_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            rows=[]
            for decomp in ['1d','2d']:
                for lib,t in [('MMM',2.),('NNN',1.)]:
                    rows.append(dict(decomposition=decomp,library=lib,size=16,ranks=1,mean_s=t,stddev_s=.1,validated=True,original_inner_loops=1,samples_s=[t-.1,t,t+.1]))
            rows.append(dict(rows[0],validated=False))
            (root/'benchmark.json').write_text(json.dumps(dict(environment={'id':'test'},label='fixture',results=rows)))
            data=load_results([root,root/'benchmark.json'])
            self.assertEqual(len(data),4)
            self.assertEqual(consolidate(data).iloc[0].run_count,1)
            report=generate(data,root/'plots')
            self.assertTrue(report.exists())
            self.assertEqual(len(list((root/'plots').glob('*.png'))),5)
            self.assertEqual(len(list((root/'plots').glob('*.pdf'))),5)
            doubled=consolidate(pd.concat([data,data]))
            self.assertEqual(len(doubled.iloc[0].samples_s),6)

    def test_benchmark_cli_with_simulated_launcher(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);build=root/'build';build.mkdir();(build/'mm_1d').touch()
            launcher=root/'launcher.py'
            launcher.write_text("import sys\nn=sys.argv[-2];lib=sys.argv[-1]\nprint(f'BENCHMARK_RESULT decomposition=1D library={lib} size={n} ranks=1 seconds=0.5 valid=1 loops=1')\n")
            command=[sys.executable,str(ROOT/'benchmark.py'),'--decomposition','1d','--size','16','--ranks','1','--warmup','1','--repeats','2','--libraries','MMM','NNN','--launcher',sys.executable,'--launcher-args',str(launcher),'--build-dir',str(build),'--output',str(root/'out')]
            for _ in range(2):
                proc=subprocess.run(command,capture_output=True,text=True)
                self.assertEqual(proc.returncode,0,proc.stderr)
            files=list((root/'out').rglob('benchmark.json'));self.assertEqual(len(files),2)
            doc=json.loads(files[0].read_text());self.assertEqual(doc['results'][0]['samples_s'],[.5,.5]);self.assertEqual(doc['results'][1]['speedup_vs_MMM'],1)
            launcher.write_text("print('BENCHMARK_RESULT decomposition=1D library=MMM size=16 ranks=1 seconds=0.5 valid=0 loops=1')\n")
            proc=subprocess.run(command,capture_output=True,text=True);self.assertEqual(proc.returncode,2)

if __name__=='__main__':unittest.main()
