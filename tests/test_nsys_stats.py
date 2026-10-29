import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from configuration.plot_nsys_stats import read_summary,plot_summary,export_stats

class NsysStats(unittest.TestCase):
    def test_preamble_bom_quoted_names_and_invalid_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'kernels.csv'
            path.write_text('\ufeffNsight export message\n\nTime (%),Total Time (ns),Name\n60%,600,"gemm<double, double>"\n40,400,ncclKernel\nnan,0,bad\n-1,0,negative\ninf,0,infinite\n')
            df,name,percent,discarded=read_summary(path)
            self.assertEqual(discarded,3);self.assertEqual(df[name].tolist(),['gemm<double, double>','ncclKernel'])
            self.assertEqual(df[percent].tolist(),[60,40])

    def test_plot_top_names_preserved_without_renormalizing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'kernels.csv';path.write_text('Time(%),Kernel Name\n70,same\n20,same\n10,other\n')
            outputs,_=plot_summary(path,top=2)
            self.assertTrue(all(x.stat().st_size>1000 for x in outputs))
            data=json.loads((Path(tmp)/'kernels_plot.json').read_text())
            self.assertEqual(data['displayed_percent_sum'],90)
            self.assertEqual(data['plotted_rows'],2)

    def test_missing_columns_empty_and_invalid_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.csv'
            for text in ('Name,Duration\nx,2\n','Time (%),Name\n','Time (%),Name\nno,x\n'):
                path.write_text(text)
                with self.assertRaises(ValueError):read_summary(path)

    def test_old_and_new_report_names_with_simulated_nsys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'rank-0.qdrep';source.touch();fake=root/'nsys'
            fake.write_text('#!'+sys.executable+'\nimport sys\nfrom pathlib import Path\na=sys.argv\nr=a[a.index("--report")+1]\nif r=="gpukernsum":\n print("Report gpukernsum not found",file=sys.stderr);sys.exit(1)\nprefix=a[a.index("--output")+1]\nPath(prefix+"_"+r+".csv").write_text("Time (%),Name\\n100,gemm\\n")\n')
            fake.chmod(0o755)
            path,report=export_stats(source,root/'out',nsys=str(fake))
            self.assertEqual(report,'cuda_gpu_kern_sum');self.assertTrue(path.exists())
            self.assertIn('gpukernsum',(root/'out'/'rank-0_nsys_stats.log').read_text())
            cli=subprocess.run([sys.executable,str(ROOT/'configuration'/'plot_nsys_stats.py'),str(source),'--nsys',str(fake),'--output-dir',str(root/'cli')],capture_output=True,text=True)
            self.assertEqual(cli.returncode,0,cli.stderr)
            self.assertTrue((root/'cli'/'rank-0_cuda_gpu_kern_sum_plot.png').exists())
            # Explicit old-report path supported when the CLI provides it.
            fake.write_text(fake.read_text().replace('if r=="gpukernsum":','if False:'))
            path,report=export_stats(source,root/'old',nsys=str(fake))
            self.assertEqual(report,'gpukernsum')

    def test_cli_csv_and_input_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'kernels.csv';path.write_text('Time (%),Name\n100,gemm\n')
            cli=[sys.executable,str(ROOT/'configuration'/'plot_nsys_stats.py')]
            result=subprocess.run(cli+[str(path),'--output-dir',str(root/'plots')],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue((root/'plots'/'kernels_plot.png').exists())
            for extra in ([str(path),'--top','0'],[str(root/'missing.csv')]):
                result=subprocess.run(cli+extra,capture_output=True,text=True)
                self.assertEqual(result.returncode,2)

    def test_nsys_failure_does_not_try_unrelated_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'rank-0.qdrep';source.touch();fake=root/'nsys'
            fake.write_text('#!'+sys.executable+'\nimport sys\nprint("Permission denied",file=sys.stderr)\nsys.exit(1)\n');fake.chmod(0o755)
            with self.assertRaises(ValueError):export_stats(source,root/'out',nsys=str(fake))
            log=(root/'out'/'rank-0_nsys_stats.log').read_text()
            self.assertEqual(log.count('COMMAND:'),1)

if __name__=='__main__':unittest.main()
