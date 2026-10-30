import unittest
from unittest.mock import patch
from configuration.experiment import _gpu_inventory, _library_versions

class TestEnvironment(unittest.TestCase):
    @patch('configuration.experiment.probe')
    def test_gpu_inventory(self, probe):
        probe.return_value = '0, NVIDIA V100, GPU-abc, 535.1, 32768, 0000:01:00.0'
        self.assertEqual(_gpu_inventory()[0]['name'], 'NVIDIA V100')

    @patch('configuration.experiment.probe', return_value=None)
    def test_no_gpu(self, probe):
        self.assertIsNone(_gpu_inventory())

    def test_library_versions_graceful(self):
        self.assertIn('nccl_header_version', _library_versions())

if __name__ == '__main__':
    unittest.main()
