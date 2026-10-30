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
import re
import csv
import io


def probe(command):
    if not shutil.which(command[0]):
        return None
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=8)
        return (result.stdout + result.stderr).strip()[:16000] if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None



def _gpu_inventory():
    raw = probe(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id', '--format=csv,noheader,nounits'])
    if raw is None:
        return None
    keys = ('index', 'name', 'uuid', 'driver_version', 'memory_mib', 'pci_bus_id')
    return [dict(zip(keys, [v.strip() for v in row])) for row in csv.reader(io.StringIO(raw)) if len(row) == len(keys)]


def _version_from_header(header, pattern):
    if not header.is_file():
        return None
    text = header.read_text(errors='replace')
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1) if match else None


def _find_header(name, env_keys, candidates):
    paths = []
    for key in env_keys:
        if os.environ.get(key):
            paths.append(Path(os.environ[key]) / 'include' / name)
    paths.extend(Path(path) for path in candidates)
    return next((path for path in paths if path.is_file()), None)


def _library_versions():
    nccl_header = _find_header('nccl.h', ('NCCL_HOME', 'NCCL_ROOT'), ('/usr/include/nccl.h', '/usr/local/cuda/include/nccl.h'))
    nvshmem_header = _find_header('nvshmem.h', ('NVSHMEM_HOME', 'NVSHMEM_ROOT'), ('/usr/include/nvshmem.h',))
    nccl = None
    if nccl_header:
        text = nccl_header.read_text(errors='replace')
        vals = [re.search(r'^\s*#\s*define\s+NCCL_'+part+r'\s+([0-9]+)', text, re.MULTILINE) for part in ('MAJOR','MINOR','PATCH')]
        if all(vals):
            nccl = '.'.join(m.group(1) for m in vals)
    nvshmem = None
    if nvshmem_header:
        text = nvshmem_header.read_text(errors='replace')
        vals = [re.search(r'^\s*#\s*define\s+NVSHMEM_'+part+r'\s+([0-9]+)', text, re.MULTILINE) for part in ('MAJOR_VERSION','MINOR_VERSION','PATCH_VERSION')]
        if all(vals):
            nvshmem = '.'.join(m.group(1) for m in vals)
    return {'nccl_header_version': nccl, 'nccl_header_path': str(nccl_header) if nccl_header else None,
            'nvshmem_header_version': nvshmem, 'nvshmem_header_path': str(nvshmem_header) if nvshmem_header else None,
            'note': 'Header versions are compile-time hints, not proof of the dynamically loaded runtime versions.'}


def collect_environment(metadata_path=None, build_dir=None):
    data = {
        'hostname': platform.node(), 'platform': platform.platform(),
        'python': platform.python_version(),
        'gpu': probe(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader']),
        'gpu_topology': probe(['nvidia-smi', 'topo', '-m']),
        'gpu_inventory': _gpu_inventory(),
        'library_versions': _library_versions(),
        'cuda': probe(['nvcc', '--version']), 'mpi': probe(['mpirun', '--version']),
        'interconnect': probe(['ibstat']),
        'modules': os.environ.get('LOADEDMODULES'),
        'runtime': {k: v for k, v in os.environ.items() if k.startswith(('MM_PIPELINE_', 'NCCL_', 'NVSHMEM_', 'UCX_')) and not any(x in k.upper() for x in ('TOKEN','KEY','PASSWORD','SECRET'))},
        'allocation': {k: os.environ[k] for k in ('SLURM_JOB_NODELIST','SLURM_NNODES','SLURM_GPUS_ON_NODE') if k in os.environ},
        'software_note': 'NCCL/NVSHMEM header versions are hints; capture linked/runtime versions separately for publication.',
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
