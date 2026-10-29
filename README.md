<div align="center">
    <img src="img/logo.png" width="800" style="max-width: 100%;">
<!-- Technologies -->
<!-- HPC Features -->
    <p align="center">
  <img alt="Version" src="https://img.shields.io/badge/version-1-blue?style=flat-square">
  <img alt="C++ and CUDA" src="https://img.shields.io/badge/C++-CUDA-76B900?style=flat-square">
  <img alt="Communication" src="https://img.shields.io/badge/communication-MPI%20%7C%20CUDA--Aware%20MPI%20%7C%20NCCL%20%7C%20NVSHMEM-00599C?style=flat-square">
  <img alt="Visualization" src="https://img.shields.io/badge/plots-Python%20%7C%20Matplotlib-3776AB?style=flat-square">
  <img alt="HPC" src="https://img.shields.io/badge/computing-HPC-8A2BE2?style=flat-square">
  <img alt="Multi-GPU" src="https://img.shields.io/badge/architecture-Multi--GPU-76B900?style=flat-square">
  <img alt="Benchmark" src="https://img.shields.io/badge/performance-Benchmarking-DC143C?style=flat-square">
</p>
</p>

</div>


MGXBench is a C++/CUDA benchmark for evaluating the execution performance of distributed matrix multiplication on multiple NVIDIA GPUs. Its current workload compares **1D Row-Block** and **2D Block/SUMMA** decompositions using **MPI, CUDA-aware MPI, NCCL, and NVSHMEM** communication strategies. The benchmark investigates how data distribution, communication mechanisms, and communication–computation overlap affect execution time and throughput.

## Contents

1. [Benchmark design](#benchmark-design)
2. [Domain decomposition](#domain-decomposition)
3. [Communication models](#communication-models)
4. [Communication–computation overlap](#communicationcomputation-overlap)
5. [Repository structure](#repository-structure)
6. [Requirements and SDumont environment](#requirements-and-sdumont-environment)
7. [Installation and compilation](#installation-and-compilation)
8. [Running MGXBench](#running-mgxbench)
9. [Outputs and performance metrics](#outputs-and-performance-metrics)
10. [Validation and interpretation](#validation-and-interpretation)
11. [Cleaning the build](#cleaning-the-build)

## Benchmark design

MGXBench evaluates two complementary dimensions of distributed GPU execution:

- **Domain decomposition:** how matrices are partitioned and assigned to GPUs.
- **Communication model:** how matrix data is transferred and whether communication can overlap GPU computation.

The workload is dense matrix multiplication, $C = A × B$, with square matrices of order $N$. The Python runner (`benchmark.py`) orchestrates repeated runs of the CUDA executables, records timings and validation results, and exports CSV/JSON summaries.

## Domain decomposition

![1D and 2D domain decomposition](img/domain-decomposition.png)

## Communication models

![Communication models](img/communication-models.png)

The benchmark exposes the following library identifiers through `--libraries`:

| ID | Model | Main communication mechanism | Chunking | Intended overlap |
|:--:|---|---|---|---|
| `MMM` | **MPI-SYNC** | Blocking `MPI_Bcast` via host buffers | No | None |
| `YYY` | **MPI-ASYNC** | `MPI_Ibcast`, pinned host buffers, asynchronous host-to-device transfers | A and B | Communication / H2D / GEMM pipeline |
| `CCC` | **CUDA-Aware-SYNC** | Blocking `MPI_Bcast` directly on CUDA buffers | No | None |
| `XXX` | **CUDA-Aware-ASYNC** | `MPI_Ibcast` directly on GPU buffers | B | B communication / GEMM |
| `NNN` | **NCCL** | `ncclBroadcast` on CUDA streams | B | B communication / GEMM |
| `WWW` | **NVSHMEM-SYNC** | Blocking `nvshmem_getmem` | No | None |
| `SSS` | **NVSHMEM-ASYNC** | `nvshmemx_getmem_nbi_on_stream` and double buffering | B | NVSHMEM GET / GEMM |

The terms *asynchronous* and *overlap* describe the implementations' **intended behavior**. Actual overlap depends on the hardware, runtime, synchronization, and problem size, and should be verified experimentally.

##  Repository structure

```text
MGXBench/
├── 1D-row-block/
│   ├── mmb.cu
│   ├── mulmat_kernel.cu
│   └── README.md
│
├── 2D-block-matrix/
│   ├── mmb.cu
│   ├── mulmat_kernel.cu
│   └── README.md
│
├── configuration/
│   ├── experiment.py
│   ├── run_campaign.py
│   ├── plot_benchmark.py
│   ├── plot_nsys_stats.py
│   └── requirements-viz.txt
│
├── tools/
│   ├── profile_rank.sh
│   └── stats_rank.sh
│
├── tests/
│   ├── test_workflow.py
│   ├── test_configuration_layout.py
│   ├── test_nsys_stats.py
│   └── test_v5_scope.py
│
├── examples/
├── img/
├── benchmark.py
├── CMakeLists.txt
├── install.sh
├── README.md
└── GUIDE.md      
```

The `build/` and `results/` directories are created locally and are not required in the source distribution.

## 🚀 Get Started

### Initialization Modules

```bash
[username ~]$ module load anaconda3/2020.07_sequana 
[username ~]$ module load nvshmem/3.1.7_cuda-11.2_sequana
[username ~]$ module load cmake/3.30.3_sequana
[username ~]$ module load gcc/9.3_sequana
[username ~]$ module load cuda/11.2_sequana
[username ~]$ module load openmpi/gnu/4.1.4+cuda-11.2_sequana
[username ~]$ module load nccl/2.13_cuda-11.2_sequana
```

### 📦 Installation

```bash
[username ~]$ bash install.sh
```

### ▶️ Running 

**Run within a GPU allocation**, following the cluster scheduler's policies. For the examples below, request **four visible GPUs on one node**. The current execution setup associates ranks with GPU indices on the same node.

##### `1D Row-Block` 

```bash
[username ~]$ python3 benchmark.py \
              --decomposition 1d \
              --size 4096 \
              --ranks 4 \
              --libraries MMM YYY CCC XXX NNN WWW SSS \
              --warmup 2 --repeats 5 \
              --plot \
              --output results/1d
```

##### `2D Block/SUMMA`

```bash
[username ~]$ python3 benchmark.py \
              --decomposition 2d \
              --size 4096 \
              --ranks 4 \
              --libraries MMM YYY CCC XXX NNN WWW SSS \
              --warmup 2 --repeats 3 \
              --plot \
              --output results/2d
```

##### Main command-line arguments

| Argument | Purpose |
|---|---|
| `--decomposition` | Select `1d` or `2d` |
| `--size` | Matrix dimension **N** (for an N × N problem) |
| `--ranks` | Number of MPI ranks / participating GPUs |
| `--libraries` | Communication implementations to evaluate |
| `--warmup` | Warm-up runs before recorded measurements |
| `--repeats` | Number of recorded runs per implementation |
| `--output` | Destination directory for reports |
| `--build-dir` | Directory containing `mm_1d` and `mm_2d` |
| `--launcher` | MPI launcher executable (default: `mpirun`) |
| `--launcher-args` | Additional launcher arguments |
| `--plot` | Generate figure and table with the experimental results |
| `--timeout` | Execution timeout in seconds |

###  📊 Outputs and performance metrics

```text
results/sweep-4gpus/
├── runs/
│   └── <identificador>/
│       ├── benchmark.csv
│       ├── benchmark.json
│       └── <logs de execução>
└── report/
    ├── report.html
    ├── report.md
    ├── summary.csv
    └── <figures PNG and PDF>
```

### 🧩 Validation

```bash
[username ~]$ python3 -m unittest discover -s tests -v
```

### ✅ Cleaning the build

Remove compiled objects and binaries while retaining the CMake configuration:

```bash
[username ~]$ cmake --build build --target clean
```
