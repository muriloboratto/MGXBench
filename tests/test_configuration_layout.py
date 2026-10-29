"""Regression coverage for the v4 directory move and executable paths."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class ConfigurationLayout(unittest.TestCase):
    def test_files_moved_and_visualization_requirements(self):
        files=['plot_benchmark.py','run_campaign.py','experiment.py','plot_nsys_stats.py','requirements-viz.txt']
        for name in files:
            self.assertTrue((ROOT/'configuration'/name).is_file())
            self.assertFalse((ROOT/name).exists())
        self.assertIn('numexpr==2.8.6',(ROOT/'configuration/requirements-viz.txt').read_text().splitlines())

    def test_campaign_and_benchmark_plot_subprocess_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);build=root/'build';build.mkdir();(build/'mm_1d').touch()
            launcher=root/'launcher.py'
            launcher.write_text("import sys\nn=sys.argv[-2];lib=sys.argv[-1]\nprint(f'BENCHMARK_RESULT decomposition=1D library={lib} size={n} ranks=1 seconds=0.5 valid=1 loops=1')\n")
            command=[sys.executable,str(ROOT/'configuration/run_campaign.py'),'--decompositions','1d','--sizes','16','--ranks','1','--warmup','0','--repeats','2','--libraries','MMM','NNN','--launcher',sys.executable,'--launcher-args',str(launcher),'--build-dir',str(build),'--output',str(root/'out'),'--plot']
            result=subprocess.run(command,cwd=root,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            files=list((root/'out').rglob('benchmark.json'));self.assertEqual(len(files),1)
            self.assertTrue((files[0].parent/'plots'/'report.html').is_file())
            self.assertTrue((root/'out'/'report'/'report.html').is_file())
            self.assertEqual(len(json.loads(files[0].read_text())['results']),2)

if __name__=='__main__':unittest.main()
