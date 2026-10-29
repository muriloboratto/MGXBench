"""Persistent experiment identities, bounded environment probes, robust statistics."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import subprocess
import uuid


def probe(command):
    if not shutil.which(command[0]):
        return None
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=8)
        return (result.stdout + result.stderr).strip()[:16000] if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def collect_environment(metadata_path=None, build_dir=None):
    data = {
        'hostname': platform.node(), 'platform': platform.platform(),
        'python': platform.python_version(),
        'gpu': probe(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader']),
        'gpu_topology': probe(['nvidia-smi', 'topo', '-m']),
        'cuda': probe(['nvcc', '--version']), 'mpi': probe(['mpirun', '--version']),
        'interconnect': probe(['ibstat']),
        'modules': os.environ.get('LOADEDMODULES'),
        'runtime': {k: v for k, v in os.environ.items() if k.startswith(('MM_PIPELINE_', 'NCCL_', 'NVSHMEM_', 'UCX_')) and not any(x in k.upper() for x in ('TOKEN','KEY','PASSWORD','SECRET'))},
        'allocation': {k: os.environ[k] for k in ('SLURM_JOB_NODELIST','SLURM_NNODES','SLURM_GPUS_ON_NODE') if k in os.environ},
        'software_note': 'NCCL/NVSHMEM versions must be supplied in user metadata if not present in loaded modules.',
    }
    if metadata_path:
        supplied = json.loads(Path(metadata_path).read_text())
        if not isinstance(supplied, dict):
            raise ValueError('environment metadata must be a JSON object')
        data['user'] = supplied
    if build_dir:
        data['binaries'] = {name: hashlib.sha256((Path(build_dir)/name).read_bytes()).hexdigest() if (Path(build_dir)/name).is_file() else None for name in ('mm_1d','mm_2d')}
    # Identity is intentionally strict: never silently mix distinct hardware/runtime environments.
    data['id'] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]
    return data


def new_run(output):
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-' + uuid.uuid4().hex[:8]
    dest = Path(output) / 'runs' / run_id
    dest.mkdir(parents=True, exist_ok=False)
    return dest, run_id


def sample_statistics(samples, bootstrap=2000):
    mean = statistics.mean(samples)
    rng = random.Random(0)
    if len(samples) > 1:
        means = sorted(statistics.mean(rng.choices(samples, k=len(samples))) for _ in range(bootstrap))
        low, high = means[int(.025*bootstrap)], means[min(bootstrap-1, int(.975*bootstrap))]
    else:
        low = high = None
    return {'cv_pct': 100*statistics.pstdev(samples)/mean,
            'ci95_low_s': low, 'ci95_high_s': high}
