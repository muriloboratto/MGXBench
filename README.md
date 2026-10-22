# MGXBench — Multi-GPU Execution Benchmark

**Distributed matrix multiplication | GPU communication | Domain decomposition | Performance evaluation**

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

- **Data decomposition:** how matrices are partitioned and assigned to GPUs.
- **Communication model:** how matrix data is transferred and whether communication can overlap GPU computation.

The workload is dense matrix multiplication, **C = A × B**, with square matrices of order **N**. The Python runner (`benchmark.py`) orchestrates repeated runs of the CUDA executables, records timings and validation results, and exports CSV/JSON summaries.

## Domain decomposition

### Scheme 1 — 1D Row-Block

Matrix **A** is partitioned by rows across GPUs. Each GPU needs the complete matrix **B** and computes its corresponding block of **C**:

$$
C_i = A_i B
$$

For the current implementation, **N must be divisible by the number of MPI ranks**.

### Scheme 2 — 2D Block/SUMMA

Matrices are partitioned into two-dimensional blocks. Each output block is formed from contributions of the corresponding **A** and **B** blocks:

$$
C_{ij} = \sum_k A_{ik} B_{kj}
$$

The implementation follows a block-oriented organization based on **SUMMA (Scalable Universal Matrix Multiplication Algorithm)**. For the current implementation, the number of ranks must be a **perfect square**, and **N must be divisible by the square root of the rank count**.

![1D and 2D domain decomposition](img/domain-decomposition.png)

## Communication models

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

![Communication models](img/communication-models.png)

## Communication–computation overlap

| Synchronous approach | Asynchronous / pipelined approach |
|---|---|
| Communication completes before GEMM begins | Data may be transferred in chunks |
| Communication → GEMM | Communication ↔ GEMM |
| No intended overlap | Intended communication–computation overlap |

Potential overlap mechanisms include **chunking**, **CUDA streams**, **nonblocking communication**, and **double buffering**. The objective is to reduce exposed data-movement time by executing GPU computation while subsequent data is in transit.

## Repository structure

```text
benchmark/
├── install.sh                 # Configure and build both CUDA executables
├── CMakeLists.txt             # CMake build configuration
├── benchmark.py               # Experiment runner; CSV and JSON output
├── README.md                  # MGXBench documentation (this file)
├── 1D-row-block/
│   ├── mmb.cu
│   ├── mulmat_kernel.cu
│   ├── README.md
│   └── experimental-results/  # Historical measurements
├── 2D-block-matrix/
│   ├── mmb.cu
│   ├── mulmat_kernel.cu
│   ├── README.md
│   └── experimental-results/  # Historical measurements
└── img/
    ├── domain-decomposition.png
    └── communication-models.png
```

The `build/` and `results/` directories are created locally and are not required in the source distribution.

## Requirements and SDumont environment

The documented reference environment is the **Santos Dumont (SDumont) supercomputer**, using **NVIDIA V100** GPUs with CUDA compute capability **7.0 (`sm_70`)**.

- Linux and Python 3
- CMake 3.22 or newer (a version compatible with the selected compiler runtime)
- GCC 9.3 and CUDA 11.2
- CUDA-enabled OpenMPI 4.1.4
- NCCL 2.13 and NVSHMEM 3.1.7

Example SDumont modules inicialization:

```bash
module load gcc/9.3_sequana
module load cuda/11.2_sequana
module load openmpi/gnu/4.1.4+cuda-11.2_sequana
module load nccl/2.13_cuda-11.2_sequana
module load nvshmem/3.1.7_cuda-11.2_sequana
module load cmake/3.30.3_sequana
```

Ensure the selected CUDA toolkit, GCC compiler, MPI implementation, and CMake executable are mutually compatible.

## Installation and compilation

From the repository root, with the required modules already loaded:

```bash
bash install.sh
```

To remove the existing build directory before compiling:

```bash
bash install.sh --clean
```

## Running MGXBench

**Run within a GPU allocation**, following the cluster scheduler's policies. For the examples below, request **four visible GPUs on one node**, and verify GPU visibility:

```bash
nvidia-smi
```

The current execution setup associates ranks with GPU indices on the same node; it is **not a general multi-node launcher**.

### 1D Row-Block — 4 GPUs

```bash
python3 benchmark.py --decomposition 1d --size 4096 --ranks 4 \
  --libraries MMM YYY CCC XXX NNN WWW SSS \
  --warmup 2 --repeats 5 \
  --output results/1d
```

### 2D Block/SUMMA — 4 GPUs

```bash
python3 benchmark.py --decomposition 2d --size 4096 --ranks 4 \
  --libraries MMM YYY CCC XXX NNN WWW SSS \
  --warmup 2 --repeats 5 \
  --output results/2d
```

### Main command-line arguments

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
| `--timeout` | Execution timeout in seconds |

## Outputs and performance metrics

```text
results/
├── 1d/
│   ├── benchmark.csv
│   └── benchmark.json
└── 2d/
    ├── benchmark.csv
    └── benchmark.json
```

The CSV summarizes each implementation using fields such as `mean_s`, `median_s`, `p95_s`, `stddev_s`, `gflops`, `speedup_vs_MMM`, and `validated`. The JSON stores run parameters, individual timing samples, and recorded failures.

**Throughput** is computed from the conventional dense matrix multiplication operation count:

$$
\mathrm{GFLOP/s} = \frac{2N^3}{\mathrm{mean\_s}\times 10^9}
$$

**Speedup** uses the `MMM` (MPI-SYNC) result as the baseline:

$$
\mathrm{Speedup}_{X} = \frac{T_{\mathrm{MMM}}}{T_X}
$$

The reported time is measured by the C++/CUDA application's computation/communication region, **not the complete MPI process launch**.

## Validation and interpretation

- The original implementation validates numerical results using **constant-valued matrices**. This is useful for regression checks but is not equivalent to testing arbitrary matrix contents.
- The **1D implementation internally averages 10 iterations**, whereas the **2D implementation measures one execution**. Direct comparisons of absolute timings between decomposition schemes therefore require care.
- Results depend on matrix size, GPU model, number of ranks, communication stack, memory transfers, and synchronization behavior.
- The `validated` result indicates whether the implementation passed its current numerical check; it does not establish comprehensive numerical correctness for all inputs.
- Confirm successful compilation and run correctness on the target cluster before publishing comparative performance claims.

## Cleaning the build

Remove compiled objects and binaries while retaining the CMake configuration:

```bash
cmake --build build --target clean
```

Remove the entire CMake build directory:

```bash
rm -rf build
```

Neither command removes `results/`. Delete a specific result directory explicitly if you intend to discard its reports.

---

**MGXBench — Multi-GPU Execution Benchmark**  
*Current workload: Distributed Multi-GPU Matrix Multiplication*
