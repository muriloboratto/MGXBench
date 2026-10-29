"""Regression checks for the removed v5 features and immediate campaign execution."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class V5Scope(unittest.TestCase):
    def test_removed_modules_and_examples_absent(self):
        for path in ('configuration/dashboard.py','configuration/requirements-dashboard.txt',
                     'configuration/analyze_overlap.py','examples/intervals_SYNTHETIC.csv','examples/synthetic-overlap'):
            self.assertFalse((ROOT/path).exists(),path)
        for file in (ROOT/'configuration').glob('*.py'):
            self.assertNotIn('import streamlit',file.read_text())
            self.assertNotIn('import plotly',file.read_text())

    def test_removed_flag_rejected_before_any_execution(self):
        script=ROOT/'configuration/run_campaign.py'
        help_result=subprocess.run([sys.executable,str(script),'--help'],capture_output=True,text=True)
        self.assertEqual(help_result.returncode,0)
        self.assertNotIn('--dry-run',help_result.stdout)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'not-created'
            for flag in ('--dry-run','--dry-run=true'):
                proc=subprocess.run([sys.executable,str(script),'--sizes','16','--ranks','1','--output',str(output),flag],capture_output=True,text=True)
                self.assertEqual(proc.returncode,2,proc.stderr)
                self.assertIn('foi removido',proc.stderr)
                self.assertNotIn('benchmark.py',proc.stdout)
                self.assertFalse(output.exists())

if __name__=='__main__':unittest.main()
