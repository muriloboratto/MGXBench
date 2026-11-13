# MGXBench — User Guide | v1.0.0

A practical guide to running MGXBench benchmarks in multi-GPU environments, comparing communication models, generating plots, and analyzing execution profiles from the command line. Adapt the commands shown to the resources and modules available on your cluster.

## Table of Contents

1. [Choose a Workflow](#workflow)
2. [Environment Setup and Installation](#installation)
3. [Run Your First Benchmark and Compare 1D vs. 2D](#first-benchmark)
4. [Output Files and Performance Metrics](#outputs)
5. [Benchmark Campaigns Across Matrix Sizes and Communication Models](#campaigns)
6. [Strong and Weak Scaling](#scaling)
7. [Running MGXBench with SLURM](#slurm)
8. [Reports and Visualizations](#plots)
9. [Profiling](#profiling)
10. [Methodology and Reproducibility](#methodology)
11. [Tool Validation](#validation)
12. [Predictive Autotuning](#autotuning)
13. [Cleaning the Build](#cleaning)

<a id="workflow"></a>
## 1. Choose a Workflow

MGXBench compares two matrix-multiplication decomposition strategies: **1D Row-Block** (row-wise partitioning) and **2D Block/SUMMA** (two-dimensional block partitioning). The documented communication models are:

| Code | Communication model |
|---|---|
| `MMM` | MPI-SYNC |
| `YYY` | MPI-CHUNKED-ASYNC |
| `CCC` | CUDA-AWARE-SYNC |
| `XXX` | CUDA-AWARE-ASYNC |
| `NNN` | NCCL-ASYNC |
| `WWW` | NVSHMEM-CHUNKED-SYNC |
| `SSS` | NVSHMEM-ASYNC |

The *asynchronous* labels describe the implemented strategy; they do not, by themselves, guarantee effective overlap between communication and computation. Such overlap must be verified through timeline-based profiling.

| Goal | Program or command | Main output |
|---|---|---|
| Benchmark a single configuration | `benchmark.py` | CSV, JSON, and logs |
| Benchmark multiple sizes/GPU counts | `configuration/run_campaign.py` | Run history and final report |
| Generate or regenerate plots | `configuration/plot_benchmark.py` | PNG, PDF, HTML, Markdown, and consolidated summary table |
| Predict the fastest calibrated configuration | `autotuning.predictive calibrate`, then `benchmark.py --libraries AUTO` | Calibration JSON and prediction ranking |
| Capture execution activity | `tools/profile_rank.sh`, via MPI | Per-rank Nsight capture |
| Inspect capture statistics | `nsys stats` | Terminal tables or CSV |
| Plot a kernel summary | `configuration/plot_nsys_stats.py` | Per-kernel bar charts in PNG/PDF |


<a id="installation"></a>
## 2. Environment Setup and Installation

### 2.1 Navigate to the project directory and command-line help

MGXBench provides a command-line help interface that lists all available arguments, configuration options, and their descriptions.

To display the available benchmark options, run:

```bash
[username ~]$ cd MGXBench

[username ~]$ python3 benchmark.py --help

```

It is recommended to consult these commands before running experiments to verify the options supported by the installed version of MGXBench.


### 2.2 Prepare the cluster build and runtime environment

Building requires compatible versions of CMake, a C++ compiler, CUDA, MPI, NCCL, and NVSHMEM. Check the following in the same environment:

```bash
[username ~]$ python3 --version
[username ~]$ cmake --version
[username ~]$ nvcc --version
[username ~]$ mpicxx --version
[username ~]$ mpirun --version
[username ~]$ nvidia-smi
```
Example module configuration for a specific environment (module names and versions may vary across clusters):

```bash
[username ~]$ module load anaconda3/2020.07_sequana
[username ~]$ module load nvshmem/3.1.7_cuda-11.2_sequana
[username ~]$ module load cmake/3.30.3_sequana
[username ~]$ module load gcc/9.3_sequana
[username ~]$ module load cuda/11.2_sequana
[username ~]$ module load openmpi/gnu/4.1.4+cuda-11.2_sequana
[username ~]$ module load nccl/2.13_cuda-11.2_sequana
```

After loading the required modules, build the benchmark:

```bash
[username ~]$ bash install.sh
```

The documented installer expects the static library `$NVSHMEM_HOME/lib/libnvshmem.a`. Before building, make sure `NVSHMEM_HOME` is set and the file exists. The installer's default CUDA architecture is `70`. For a different architecture, set `CUDA_ARCH` to match the target GPU. The `--clean` option removes the build directory before rebuilding; use it only when a clean rebuild is needed.


<a id="first-benchmark"></a>
## 3. Run Your First Benchmark and Compare 1D vs. 2D

Start with a smaller matrix and a few repetitions to verify the entire workflow. Before running a four-rank benchmark, request an allocation with four GPUs using your cluster scheduler.  The partition name, GPU limit, and allocation policy are site-specific; confirm that the allocation provides four distinct GPUs before running the commands below. 

### 3.1 Run the 1D benchmark

```bash
[username ~]$ python3 benchmark.py \
              --decomposition 1d \
              --size 4096 \
              --ranks 4 \
              --libraries MMM YYY CCC XXX NNN WWW SSS \
              --warmup 2 \
              --repeats 3 \
              --plot \
              --output results/1d
```

### 3.2 Run the 2D benchmark

```bash
[username ~]$ python3 benchmark.py \
              --decomposition 2d \
              --size 4096 \
              --ranks 4 \
              --libraries MMM YYY CCC XXX NNN WWW SSS \
              --warmup 2 \
              --repeats 3 \
              --plot \
              --output results/2d
```

### 3.3 Compare the decompositions

```bash
[username ~]$ python3 configuration/plot_benchmark.py \
              --input results/1d results/2d \
              --output results/comparison
```

Open `results/comparison/report.html` on your computer, keeping the images in the same directory.

| `benchmark.py` argument | Description |
|---|---|
| `--decomposition` | Selects `1d` or `2d` |
| `--size` | Matrix dimension N for an N × N matrix |
| `--ranks` | Number of MPI processes; typically one per GPU |
| `--libraries` | Models to run, or `AUTO` alone to predict |
| `--prediction-model` | `extrapolation`, `ml`, or `auto`, used with `AUTO` |
| `--warmup` | Initial runs excluded from statistics |
| `--repeats` | Number of measured runs per model |
| `--plot` | Generates plots for this configuration after measurement |
| `--output` | Base directory for run history |


<a id="outputs"></a>
## 4. Output Files and Performance Metrics

Each benchmark invocation creates a run identifier consisting of a UTC timestamp and a unique suffix:

```text
results/1d/
└── runs/
    └── <run-id>/
        ├── benchmark.csv
        ├── benchmark.json
        ├── MMM_0.log
        ├── ...
        └── plots/               
            ├── report.html
            ├── report.md
            ├── summary.csv
            ├── <figure>.png
            └── <figure>.pdf
```

Previous runs are preserved. Logs also record warm-up runs; failures are recorded in the logs and in the JSON `failures` list. The CSV contains summaries for successfully completed and numerically validated models. The JSON includes individual samples in `samples_s`, results, failures, and environment metadata.

| Field | Interpretation |
|---|---|
| `mean_s` | Mean execution time in seconds; lower is better |
| `median_s` | Median sample time |
| `p95_s` | 95th percentile of execution times |
| `stddev_s` | Population standard deviation |
| `cv_pct` | Relative dispersion: 100 × standard deviation / mean |
| `ci95_low_s`, `ci95_high_s` | Bounds of a bootstrap confidence interval for the mean |
| `gflops` | Effective throughput: 2N³ / (mean time × 10⁹) |
| `speedup_vs_MMM` | Mean execution time of MMM divided by that of the model |
| `validated` | Indicates successful numerical validation |
| `original_inner_loops` | Number of inner iterations per launch |

GFLOP/s throughput is an estimate of effective performance based on the theoretical operation count for dense matrix multiplication and the measured mean execution time. It reflects the application's timed region, including the computation and communication costs covered by the measurement. It is neither the hardware's theoretical peak performance nor an isolated kernel measurement.

<a id="campaigns"></a>
## 5. Benchmark Campaigns Across Matrix Sizes and Communication Models

### 5.1 Vary matrix size: `sweep` mode

This command runs the five specified matrix sizes, seven models, and two decompositions using four GPUs:

```bash
[username ~]$ python3 configuration/run_campaign.py \
              --mode sweep \
              --decompositions 1d 2d \
              --sizes 2048 4096 8192 16384 32768 \
              --ranks 4 \
              --libraries MMM YYY CCC XXX NNN WWW SSS \
              --warmup 2 \
              --repeats 3 \
              --output results/sweep-4gpus \
              --label sweep-4gpus
```

To benchmark only 1D, use `--decompositions 1d`. To select specific models, for example, use `--libraries MMM NNN SSS`. Include MMM when you need speedup relative to MPI-SYNC.

### 5.2 What happens when the campaign finishes

The campaign runner automatically generates a consolidated report:

```text
results/sweep-4gpus/
├── runs/
│   └── <run-id>/...
└── report/
    ├── report.html
    ├── report.md
    ├── summary.csv
    └── <figures>.png / <figures>.pdf
```

The `--plot` option is not required for the final campaign report. If supplied, it is forwarded to the benchmark and also generates individual reports for each configuration.

By default, a campaign includes both decompositions, rank counts of $1$ and $4$, and the five matrix sizes listed above. Specify these arguments explicitly to make the experiment unambiguous. Options such as `--libraries`, `--warmup`, `--repeats`, `--environment`, `--build-dir`, and `--launcher-args` are forwarded to the benchmark.

<a id="scaling"></a>
## 6. Strong and Weak Scaling

### 6.1 `Strong` scaling: fixed N, more GPUs

```bash
[username ~]$ python3 configuration/run_campaign.py \
              --mode strong \
              --sizes 8192 \
              --ranks 1 4 \
              --libraries MMM YYY CCC XXX NNN WWW SSS \
              --warmup 2 --repeats 3 \
              --output results/strong \
              --label strong-N8192
```

The **strong** mode requires exactly one matrix size. For each model and decomposition, compare measured execution times at the same matrix size. The updated 2D/SUMMA implementation supports rectangular `Pr × Pc` process grids, where `Pr * Pc = p` and `N` must be divisible by both dimensions. The default is the closest factor pair to a square (for example, 4 → 2×2, 8 → 2×4, 12 → 3×4, and 16 → 4×4).

$$
S(p)=\frac{T(1)}{T(p)},\qquad E(p)=\frac{S(p)}{p}\times100\%.
$$

### 6.2 `Weak` scaling: approximately constant computational work per GPU

This is **compute-based weak scaling**, rather than memory-based weak scaling. This example requires resources and appropriate mapping for up to nine GPUs. $N$ increases approximately as $N(p) = N(1) \times p^{1/3}$ and is rounded to satisfy decomposition constraints. GPU memory usage is not held constant.

The runner aligns the theoretical size to the nearest valid multiple using
`d = lcm(--size-alignment, process-grid divisor)`. For 1D the divisor is `p`; for rectangular 2D grids, `N` must be divisible by both `Pr` and `Pc`. Verify that the installed campaign runner supports the updated rectangular-grid constraints before executing weak-scaling campaigns. The default is `--size-alignment 4`.

```bash
[username ~]$ python3 configuration/run_campaign.py \
              --mode weak  \
              --base-size 4096  \
              --decompositions 1d  \
              --size-alignment 4 \
              --libraries MMM NNN SSS  \
              --warmup 1 --repeats 2 \
              --output results/weak \
              --label weak-N4096
```

Each campaign writes `<label>-size-manifest.csv` in the output directory,
with `N_theoretical`, `N_actual`, `work_per_gpu_ratio`, decomposition, and ranks.
The work ratio is `(N_actual/base_size)^3 / p`; values near 1 indicate
approximately constant floating-point work per rank. The alignment does not
make GPU memory usage constant. Confirm that the selected CUDA kernels accept
the aligned dimensions (including their tiling constraints) before long runs.

The runner rejects rounding choices that change the work per rank by more than $2.5%$. The plot allows at most a 5% variation between the largest and smallest $N^3 / p$ values, requires one configuration per $p$, a reference at $p = 1$, and a label beginning with `weak`. Weak-scaling efficiency is $E(p) = \frac{T(1)}{T(p)} \times 100\%$.

Configurations incompatible with the selected 2D process grid or divisibility requirements are reported and skipped. Missing data points are never fabricated.

<a id="slurm"></a>
## 7. Running MGXBench with SLURM

This example is based on specifical architecture and targets with NVIDIA GPUs. It requests **two nodes**, **four MPI ranks per node**, and **four GPUs per node**, for a total of **eight MPI ranks and eight GPUs**. The source configuration describes an inter-node Mellanox InfiniBand connection. Cluster account, partition, QoS, and module availability are site-specific and must be checked before submission.

### 7.1 Prepare the batch script

Save the following as `mgxbench-8gpu.slurm` in the MGXBench project root. The two loops run both 1D row-block and 2D SUMMA decompositions for `N = 4096, 8192, 16384`, evaluating all seven communication backends. For 2D, an explicit **2×4 rectangular MPI grid** and a SUMMA panel width of **256** are used.

```bash
#!/bin/bash
#SBATCH --job-name=MGXBench-8GPU
#SBATCH --partition=sequana_gpu_dev
#SBATCH --account=treinamento
#SBATCH --qos=normal

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Environment used in the source SLURM example.
module load nvshmem/3.1.7_cuda-11.2_sequana
module load anaconda3/2020.07_sequana
module load cmake/3.30.3_sequana

# 1D row-block, eight GPUs.
for SIZE in 4096 8192 16384; do
    echo "Running 1D: N=${SIZE}, GPUs=8"
    python3 benchmark.py \
        --decomposition 1d \
        --size "${SIZE}" \
        --ranks 8 \
        --libraries MMM YYY CCC XXX NNN WWW SSS \
        --warmup 1 \
        --repeats 2 \
        --plot \
        --output results/1d
done

# 2D SUMMA, explicit rectangular 2 x 4 process grid.
for SIZE in 4096 8192 16384; do
    echo "Running 2D: N=${SIZE}, GPUs=8, grid=2x4"
    python3 benchmark.py \
        --decomposition 2d \
        --size "${SIZE}" \
        --ranks 8 \
        --grid-rows 2 \
        --grid-cols 4 \
        --summa-bk 256 \
        --libraries MMM YYY CCC XXX NNN WWW SSS \
        --warmup 1 \
        --repeats 2 \
        --plot \
        --output results/2d
done
```

The script assumes that the MGXBench binaries have already been built and that the loaded modules expose the CUDA, MPI, NCCL, and NVSHMEM runtime dependencies required by your build. Check the environment as described in Section 2.

### 7.2 Submit and monitor the job

```bash
[username ~]$ sbatch script_mgxbench-8gpu.slurm
```

SLURM allocates resources, while `benchmark.py` launches the MPI processes for each experiment. Do not add an extra `srun` around `benchmark.py` unless the launcher configuration has been explicitly adapted to avoid nested launches.

### 7.3 Rectangular SUMMA grid parameters

The updated 2D implementation supports a general `Pr × Pc` grid, where `Pr × Pc = --ranks` and the matrix dimension `N` is divisible by both `Pr` and `Pc`. The default grid is the factor pair closest to a square, with `Pr ≤ Pc`.

| MPI ranks / GPUs | Default 2D grid | Default `--summa-bk` |
|---:|:---:|---:|
| 1 | 1×1 | 256 |
| 2 | 1×2 | 256 |
| 4 | 2×2 | 256 |
| 6 | 2×3 | 256 |
| 8 | 2×4 | 256 |
| 12 | 3×4 | 256 |
| 16 | 4×4 | 256 |
| 32 | 4×8 | 256 |

`--grid-rows`, `--grid-cols`, and `--summa-bk` are optional for ordinary runs. If only one grid dimension is provided, the other is inferred. The executable receives the selected values through `MGXBENCH_GRID_ROWS`, `MGXBENCH_GRID_COLS`, and `MGXBENCH_SUMMA_BK`. For an explicit grid, use:

```bash
[username ~]$  python3 benchmark.py \
               --decomposition 2d \
               --size 8192 \
               --ranks 8 \
               --grid-rows 2 \
               --grid-cols 4 \
               --summa-bk 256 \
               --libraries MMM YYY CCC XXX NNN WWW SSS \
               --warmup 2 \
               --repeats 3 \
               --output results/rect-2x4
```

<a id="plots"></a>
## 8. Reports and Visualizations

### 7.1 Regenerate Reports from Saved Data

```bash
[username ~]$ python3 configuration/plot_benchmark.py \
              --input results/sweep-4gpus \
              --output results/sweep-4gpus/report
```

The program searches subdirectories for results. There is no need to rerun the benchmark. To select a specific configuration:

```bash
[username ~]$ python3 configuration/plot_benchmark.py \
              --input results/1d results/2d \
              --size 4096 --ranks 4 \
              --output results/comparison-N4096-p4
```

Additional filters include `--environment-id` and `--label`. A report is regenerated when its output directory is reused; choose another directory to preserve earlier reports.

### 7.2 Available Plots

| Plot | Organization and requirements |
|---|---|
| Mean execution time | Bars by model, N, and p; error bars show standard deviation |
| Throughput | Effective GFLOP/s by model |
| Speedup relative to MMM | Bars by model for each N and p; requires MMM |
| Execution-time distribution | Box plots; requires individual samples |
| Execution time vs. N | Curves by model, separated by decomposition and p; requires multiple N values |
| Strong scaling | Speedup and efficiency vs. p; fixed N and a single-GPU reference |
| Weak scaling | Efficiency vs. p; properly labeled campaign and compatible per-GPU work |



<a id="profiling"></a>
## 9. Profiling

To capture an execution profile within a GPU allocation, run:

```bash
[username ~]$ export MGXBENCH_PROFILE_DIR="$PWD/results/profile-1d-N8192-p4-NNN-$(date -u +%Y%m%dT%H%M%S)"
[username ~]$ export MGXBENCH_INNER_LOOPS=1
[username ~]$ mpirun -np 4 bash tools/profile_rank.sh "$PWD/build/mm_1d" auto 8192 NNN
```

The wrapper generates a separate profiling capture for each MPI rank. For Nsight Systems versions that produce `.qdrep` files, inspect the profiling statistics and generate plots using:

```bash
[username ~]$ nsys stats "$MGXBENCH_PROFILE_DIR/rank-0.qdrep"
[username ~]$ python3 configuration/plot_nsys_stats.py "$MGXBENCH_PROFILE_DIR/rank-0.qdrep" \
              --output-dir "$MGXBENCH_PROFILE_DIR/stats/rank-0" \
              --top 15
```

If your Nsight Systems installation generates `.nsys-rep` files, use that extension instead. No Nsight graphical interface is required. The script exports profiling statistics using `nsys stats` and generates PNG/PDF plots for individual kernels.

### 8.1 Export CSV Manually, Then Generate Plots

```bash
[username ~]$ nsys stats --report gpukernsum --format csv --output "$MGXBENCH_PROFILE_DIR/rank-0" "$MGXBENCH_PROFILE_DIR/rank-0.qdrep"
[username ~]$ python3 configuration/plot_nsys_stats.py "$MGXBENCH_PROFILE_DIR/rank-0_gpukernsum.csv" --top 15
```

The first command generates the CSV; the second creates plots alongside it. If necessary, replace the report name with `cuda_gpu_kern_sum` in both the commands and file paths.

The script expects **a CSV kernel summary** with `Name` or `Kernel Name` and a column containing both `Time` and `%`. An aligned terminal table or multiple concatenated reports do not meet this format. Text preceding the CSV header is tolerated. Rows with invalid names or percentages are skipped with a warning; analysis fails if no valid data remains.

Additional options: `--report gpukernsum`, `--report cuda_gpu_kern_sum`, `--nsys /path/to/nsys`, `--timeout 600`. Nsight does not need to be installed or run when the input is already a CSV file.


<a id="methodology"></a>
## 10. Methodology and Reproducibility

### 10.1 Samples and numerical validation

Warm-up runs are excluded from statistics. Each measured sample corresponds to one process launch; in 1D, it may represent the average of several inner iterations if requested. The default runner uses one iteration for both decompositions. A native 1D run without `MGXBENCH_INNER_LOOPS` retains ten iterations; set this variable to $1$ for comparable profiling.

The numerical validation described for this version uses constant-valued matrices and an analytically expected result. It is not equivalent to a comprehensive test suite with random inputs. The timed regions retain the scope defined by each algorithm. For more comprehensive validation, future tests should include identity matrices, seeded random matrices, distinct block-wise patterns, and absolute/relative error checks against a trusted reference implementation. These additional tests are recommendations, not features claimed to be implemented in this version.

The confidence interval uses a percentile bootstrap of the mean, with $2,000$ resamples and a fixed random seed; it is not computed for a single sample. A small number of repetitions yields fragile estimates. Three repetitions are useful for testing the workflow; increase this to $10$ or $30$ depending on the objective and observed variability. Additional repetitions do not fix an unstable environment.

### 10.2 Grouping results

Reports group only data with compatible `environment_id`, `label`, and inner-iteration counts. Each comparison also accounts for decomposition, model, N, and p. Repeated runs of the same configuration combine their raw samples. Duplicate CSV summaries without samples are rejected because the underlying distribution cannot be reconstructed.

JSON takes precedence over an adjacent CSV; overlapping entries are deduplicated. Older files without environment metadata are isolated by directory to avoid invalid comparisons. If a curve or comparison is missing, check these fields before assuming a plotting error.

### 10.3 Record the environment

Use `examples/environment.json` as a template and fill in the GPU, CPU, nodes, interconnect, CUDA/MPI/NCCL/NVSHMEM versions, and build configuration:

```bash
[username ~]$ python3 benchmark.py \
              --decomposition 1d \
              --size 4096 \
              --ranks 4 \
              --libraries MMM NNN \
              --warmup 2 \
              --repeats 3 \
              --environment examples/environment.json \
              --output results/sdumont
```

The file must contain a JSON object. Example fields containing `null` must be replaced with actual values. Automatic collection records available information about the launch host, GPU topology, CUDA compiler, MPI, modules, selected communication variables, resource allocation, and executable hashes. Failed queries remain `null`; this is not an inventory of every node.

The environment identifier is strict: changes to the GPU, host, runtime, allocation, or binaries may split results into separate groups. Supplying the same JSON file does not force different environments to be treated as identical. For scaling studies, maintain a fixed allocation and controlled process placement whenever possible.

### 10.4 GPU mapping and multi-node execution

The default `--device-map local` setting makes the runner pass `auto` to the executable. An MPI shared-memory communicator determines the local rank. If multiple GPUs are visible, the local index is selected; if only one is visible, device index zero is used.

When each process can see only one GPU, the scheduler must ensure that these are distinct physical GPUs. The program does not verify exclusivity across separate CUDA visibility spaces. `--device-map explicit` restores selection by global rank and is appropriate only when the mapping has been planned accordingly.

The default launcher is `mpirun`; placement options can be passed through `--launcher-args`. For multi-node execution, CUDA-aware MPI/NVSHMEM transport, symmetric heaps, affinity, and networking must be configured on the cluster. Local-rank support does not replace multi-node validation of each backend.

<a id="validation"></a>
## 11. Tool Validation

```bash
[username ~]$ python3 -m unittest discover -s tests -v
[username ~]$ python3 -m py_compile benchmark.py configuration/*.py
[username ~]$ bash -n tools/profile_rank.sh
[username ~]$ bash -n tools/stats_rank.sh
```
---
<a id="autotuning"></a>
## 12. Predictive Autotuning

The autotuner predicts execution time from empirical calibration and ranks compatible candidates. It selects among configurations that were successfully calibrated; it does not search unmeasured libraries, grids, or BK values. The commands below describe the documented interface. Confirm that your installed version supports `--auto-validate` and `--auto-feedback` before running the extended workflows.

### 12.1 Prerequisites and data flow

Build MGXBench first, then run from the project root within a GPU allocation. Use the same GPU/runtime/build/allocation configuration for calibration and prediction. The signature includes GPU model and driver, ranks, decomposition, CUDA visibility, and recorded SLURM allocation fields. It is not a complete fingerprint of interconnect topology or every runtime setting.

```bash
[username ~]$ python3 -m autotuning.predictive calibrate --help
[username ~]$ python3 -m autotuning.predictive auto --help
```

### 12.2 `1D autotuning with four GPU` 

#### 12.2.1 Calibration

```bash
[username ~]$ python3 -m autotuning.predictive calibrate \
              --decomposition 1d \
              --ranks 4 \
              --sizes 2048 4096 6144 8192 12288 \
              --libraries MMM YYY CCC XXX NNN WWW SSS \
              --warmup  2 \
              --repeats 3 \
              --output results/auto_calibration_1d_4gpu
```

#### 12.2.2 Prediction: power-law extrapolation and Ridge regression

```bash
[username ~]$ python3 benchmark.py \
              --decomposition 1d \
              --size 16384 \
              --ranks 4 \
              --libraries AUTO \
              --prediction-model extrapolation \
              --output results/prediction_extrapolation_1d
```

```bash
[username ~]$ python3 benchmark.py \
              --decomposition 1d \
              --size 16384 \
              --ranks 4 \
              --libraries AUTO \
              --prediction-model ml \
              --output results/prediction_ridge_1d
```

#### 12.2.3 Validation of the selected configuration

```bash
[username ~]$ python3 benchmark.py \
              --decomposition 1d \
              --size 16384 \
              --ranks 4 \
              --libraries AUTO \
              --prediction-model extrapolation \
              --auto-validate \
              --warmup 2 \
              --repeats 3 \
              --output results/validation_extrapolation_1d

[username ~]$ python3 benchmark.py \
              --decomposition 1d \
              --size 16384 \
              --ranks 4 \
              --libraries AUTO \
              --prediction-model ml \
              --auto-validate \
              --warmup 2 \
              --repeats 3 \
              --output results/validation_ridge_1d
```

#### 12.2.4 Ground truth across all seven implementations

```bash
[username ~]$ python3 benchmark.py \
              --decomposition 1d \
              --size 16384 \
              --ranks 4 \
              --libraries MMM YYY CCC XXX NNN WWW SSS \
              --warmup 2 \
              --repeats 3 \
              --output results/ground_truth_1d_16384
```

#### 12.2.5 Adaptive feedback with power-law extrapolation

Run this only after collecting independent validation results, because feedback may add N=16384 to the training set.

```bash
[username ~]$ python3 benchmark.py \
              --decomposition 1d \
              --size 16384 \
              --ranks 4 \
              --libraries AUTO \
              --prediction-model extrapolation \
              --auto-validate \
              --auto-feedback \
              --warmup 2 \
              --repeats 3 \
              --output results/prediction_ridge_feedback_1d
```


#### 12.2.6 Adaptive feedback with Ridge regression

Run this only after collecting independent validation results, because feedback may add N=$16384$ to the training set.

```bash
[username ~]$ python3 benchmark.py \
              --decomposition 1d \
              --size 16384 \
              --ranks 4 \
              --libraries AUTO \
              --prediction-model ml \
              --auto-validate \
              --auto-feedback \
              --warmup 2 \
              --repeats 3 \
              --output results/prediction_ridge_feedback_1d
```

### 12.3 `2D SUMMA autotuning with four GPUs`

#### 12.3.1 Calibration

```bash
[username ~]$ python3 -m autotuning.predictive calibrate \
              --decomposition 2d \
              --ranks 4 \
              --sizes 2048 4096 6144 8192 12288 \
              --grids 2x2 \
              --bks 128 256 \
              --libraries MMM YYY CCC XXX NNN WWW SSS \
              --warmup 2 \
              --repeats 3 \
              --output results/auto_calibration_2d_4gpu
```

#### 12.3.2 Prediction: power-law extrapolation and Ridge regression

```bash
[username ~]$ python3 benchmark.py \
              --decomposition 2d \
              --size 16384 \
              --ranks 4 \
              --grid-rows 2 \
              --grid-cols 2 \
              --libraries AUTO \
              --prediction-model extrapolation \
              --output results/prediction_extrapolation_2d

[username ~]$ python3 benchmark.py \
              --decomposition 2d \
              --size 16384 \
              --ranks 4 \
              --grid-rows 2 \
              --grid-cols 2 \
              --libraries AUTO \
              --prediction-model ml \
              --output results/prediction_ridge_2d
```

#### 12.3.3 Validation of the selected configuration

```bash
[username ~]$ python3 benchmark.py \
              --decomposition 2d \
              --size 16384 \
              --ranks 4 \
              --grid-rows 2 \
              --grid-cols 2 \
              --libraries AUTO \
              --prediction-model extrapolation \
              --auto-validate \
              --warmup 2 \
              --repeats 3 \
              --output results/validation_extrapolation_2d

[username ~]$ python3 benchmark.py \
              --decomposition 2d \
              --size 16384 \
              --ranks 4 \
              --grid-rows 2 \
              --grid-cols 2 \
              --libraries AUTO \
              --prediction-model ml \
              --auto-validate \
              --warmup 2 \
              --repeats 3 \
              --output results/validation_ridge_2d
```

#### 12.3.4 Ground truth for a fixed SUMMA block size

This command tests **BK=128 only**. To establish ground truth across both calibrated block sizes, repeat the benchmark with `--summa-bk 256` and a distinct output directory.

```bash
[username ~]$ python3 benchmark.py \
              --decomposition 2d \
              --size 16384 \
              --ranks 4 \
              --grid-rows 2 \
              --grid-cols 2 \
              --summa-bk 128 \
              --libraries MMM YYY CCC XXX NNN WWW SSS \
              --warmup 2 \
              --repeats 3 \
              --output results/ground_truth_2d_16384_bk128
```


#### 12.3.5 Adaptive feedback

The autotuner should record the selected library, 2×2 process grid, and the **BK actually selected from calibration**. Do not assume that the ordinary `--summa-bk` flag restricts AUTO candidate selection.

```bash
[username ~]$ python3 benchmark.py \
              --decomposition 2d \
              --size 16384 \
              --ranks 4 \
              --grid-rows 2 \
              --grid-cols 2 \
              --libraries AUTO \
              --prediction-model extrapolation \
              --auto-validate \
              --auto-feedback \
              --warmup 2 \
              --repeats 3 \
              --output results/prediction_ridge_feedback_2d
```

```bash
[username ~]$ python3 benchmark.py \
                --decomposition 2d \
                --size 16384 \
                --ranks 4 \
                --grid-rows 2 \
                --grid-cols 2 \
                --libraries AUTO \
                --prediction-model ml \
                --auto-validate \
                --auto-feedback \
                --warmup 2 \
                --repeats 3 \
                --output results/prediction_ridge_feedback_2d
```

### 12.4 What the prediction methods compute

| Option | Data used per candidate | Model |
|---|---|---|
| `extrapolation` | Two neighboring points; the two endpoint sizes outside the measured interval | Local power law in log-log space |
| `ml` | All valid calibration measurements | Ridge regression on log(time) |
| `auto` | Checks the minimum distinct-size count across all stored candidates | Ridge if that minimum is at least four and imports succeed; otherwise power law |

For `extrapolation`, the implementation computes:

$$
p = \mathrm{clip}\left(
\frac{\log(T_2/T_1)}{\log(N_2/N_1)}, 0, 4
\right),
\qquad
\widehat{T}(N) = T_1\left(\frac{N}{N_1}\right)^p.
$$

For an already measured size, it returns the stored time (or the median of duplicate measurements). For the 1D and 2D calibration commands above, extrapolation at N=$16384$ uses the two largest valid calibration sizes, N=$8192$ and N=$12288$, provided both measurements were accepted. If points are missing, the available endpoints are used. 

For Ridge, with candidate `c` and the minimum eligible measured size `Nref`:

$$
\log \widehat T_c(N)=b_c+w_c\log(N/N_{\mathrm{ref}}),
\qquad \widehat T_c(N)=e^{b_c}(N/N_{\mathrm{ref}})^{w_c}.
$$

The feature representation uses candidate-specific indicator and log-size interaction terms. The code uses `Ridge(alpha=0.05, fit_intercept=False)`, minimizing squared log-time error plus an L2 penalty on all coefficients, including the explicit candidate intercepts. Candidate coefficients are separate; the fit does not learn shared hardware/API features. Unlike the local method, Ridge has no explicit exponent bound and does not necessarily reproduce a calibration time exactly. 

The `auto` choice does not perform cross-validation or compare prediction errors. Neither method produces uncertainty intervals, learns performance across GPU counts, or guarantees correctness beyond the calibration range.

### 12.5 Read predictions and measure the selected configuration

`--libraries AUTO` must be used alone. In prediction-only mode, it loads calibration data, checks the environment signature, ranks compatible candidates, and exits without running the target multiplication. The `--auto-validate` and `--auto-feedback` flags are intended to extend that behavior; their exact implementation must be verified.

A prediction-only run writes `<output>/autotuning_prediction.json`:

| Field | Meaning |
|---|---|
| `best` | Selected library, grid, BK, predicted seconds, and extrapolation flag |
| `ranking` | All compatible predictions sorted by time |
| `prediction_model` | Method actually used (`extrapolation` or `ml`) |
| `speedup_vs_MPI_predicted` | Predicted MMM time / best time, using the best candidate's grid and BK; null if that baseline is absent |
| `status`, `measured` | `PREDICTED`, `false` |
| `architecture_id` | Matched calibration/environment signature identifier |

The displayed improvement is `(1 - 1/speedup) × 100%`. These are predictions, not measured speedups. Both Ridge and the power-law method report `extrapolated=true` when the target lies outside a candidate's observed size range.

These are conditional examples, not claims about the winning implementation. Replace the library/grid/BK with each model's actual selection. If the models select different candidates, measure both. Compare predicted seconds with measured `median_s`, since calibration uses medians:

$$
E_{\mathrm{rel}} = 100 \cdot \frac{|T_{\mathrm{pred}} - T_{\mathrm{measured}}|}{T_{\mathrm{measured}}}
$$

The ordinary benchmark's `speedup_vs_MMM` uses means, so distinguish it from a median-based comparison. Measuring only the selected candidate and MMM tests those configurations; establishing the actual fastest candidate requires measurements across the target-size candidate set. Reserve sizes not included in calibration for independent extrapolation evaluation.

### 12.6 Manage calibration files and troubleshoot

The prediction entry point currently reads the fixed active path `autotuning/models/calibration.json`. Calibration supports `--model` to save elsewhere, but prediction has no model-path option in the supplied interface. To reuse the archived 1D model after the 2D example:

```bash
[username ~]$ cp autotuning/models/calibration-1d-4gpu.json autotuning/models/calibration.json
```

Restore only a model compatible with the current allocation/build. Archive names here identify decomposition and ranks; use additional descriptive names for different hardware or builds. Changing `--output` separates reports but does not select another calibration model.

| Symptom | Action |
|---|---|
| `AUTO model missing` | Calibrate, or restore a compatible model at the active path. |
| Signature mismatch | Restore the correct calibration or recalibrate under the current build/allocation. |
| No compatible candidates | Inspect calibration logs, ensure two valid sizes per candidate, and check target/grid divisibility. |
| Missing NumPy/scikit-learn with `ml` | Install compatible dependencies in the executing Python environment, or use `extrapolation`. Explicit `ml` does not silently fall back. |
| Calibration produces no valid candidates | Inspect `<output>/calibration_logs/` and benchmark failures; the partial file preserves progress without replacing a previous valid model. |

Calibration stores matching benchmark reports under `<output>/runs/`, launch commands/stdout/stderr under `<output>/calibration_logs/`, and accepted data in the model JSON. It writes a `.partial.json` checkpoint after each candidate and replaces the requested model only when at least one candidate is eligible. The JSON stores empirical samples; Ridge is fitted when prediction runs, rather than stored as a serialized estimator.

For cluster launch placement, calibration accepts `--launcher`, `--launcher-args`, and `--build-dir`. Use scheduler-compatible placement for all calibration and validation runs. GPU-memory or transport failures are failed measurements, not evidence that a candidate is inherently slow.

---

### 12.7 Experimental workflow and validation strategy

| Phase | Purpose | Training data modified? |
|---|---|---|
| Calibration | Measure candidates at known sizes and build the empirical model | Yes, creates/replaces calibration data |
| Prediction | Rank compatible candidates at an unseen target size | No |
| `--auto-validate` | Measure the selected candidate and compare predicted versus observed time, if implemented | Normally no |
| Ground truth | Measure every specified library at the target size | No automatic autotuner update |
| `--auto-feedback` | Measure a selected candidate and add a validated observation for later predictions, if implemented | Intended: yes |



### 12.8 What changes after feedback?

**Intended feedback lifecycle:** (1) predict the fastest compatible candidate; (2) execute and numerically validate that candidate; (3) obtain a valid observed execution time, preferably a median consistent with calibration; (4) append/update that candidate's empirical measurements in the matching calibration model; (5) save the updated calibration; (6) refit Ridge on a later prediction (or immediately if explicitly implemented).

The baseline guide documents **empirical measurements persisted in `autotuning/models/calibration.json`, not a serialized Ridge estimator**. Ridge is fitted when a prediction is requested. Therefore, if feedback writes a valid new measurement to the active calibration file, a subsequent prediction can refit Ridge using the expanded data. Do not claim that feedback updates a stored `.pkl` model unless code actually creates one.

For 1D, the feedback observation must identify at least `(decomposition=1d, ranks, library, N, measured time, environment signature)`. For 2D it must additionally identify `(grid_rows, grid_cols, summa_bk)`. A measurement of `XXX` at BK 128 is **not** a measurement of `XXX` at BK 256, nor of `SSS`. A four-GPU measurement must not be mixed into an eight-GPU calibration.

The documented Ridge model is **log-linear per candidate**, not cubic polynomial regression:

$$
\log \widehat{T}_c(N)=b_c+w_c\log(N/N_{\mathrm{ref}}).
$$

Adding a new measured `(N, T)` observation changes the training set. Upon refitting, the candidate coefficients may change, modifying future time estimates and potentially the selected library/grid/BK.  Illustrative example:

| Quantity | Example |
|---|---:|
| Predicted time at N=16384 | 0.75 s |
| Observed median | 0.90 s |
| Absolute percentage error | 16.67% |
| New training observation | N=16384, T=0.90 s, selected candidate |

The error is $100|\widehat T-T|/T$. After feedback, do **not** evaluate model generalization on the same observation that has just been added to training. Use a held-out size or candidate, and preserve the pre-feedback prediction for comparison.

### 12.9 Verification checklist

1. `python3 benchmark.py --help` lists both flags.
2. `--auto-validate` actually executes and measures the selected candidate (rather than only producing `PREDICTED`).
3. `--auto-feedback` persists a validated observation for the **correct candidate** in the correct calibration file.
4. A subsequent `--prediction-model ml` invocation reads that file and refits Ridge; compare predictions before/after on a held-out N.
5. If the file is unchanged, do not report that the model was updated. Inspect the implementation/logs instead.

---

<a id="cleaning"></a>
## 13. Cleaning the Build

Remove compiled objects and binaries while retaining the CMake configuration:

```bash
[username ~]$ cmake --build build --target clean
```

If the project has been moved to another directory, `build/CMakeCache.txt` may contain outdated absolute paths. In that case, a normal `clean` target may fail; remove the old build directory and regenerate the configuration using the installer. This removes compiled artifacts and cached build settings, unlike `cmake --build build --target clean`.

Remove the entire CMake build directory:

```bash
[username ~]$ rm -rf build
```

To configure and build again after deleting `build/`, run:

```bash
[username ~]$ bash install.sh
```

Neither cleaning method removes `results/`. Delete a specific result directory explicitly if you intend to discard its reports.

---