# MGXBench — User Guide | v1.0.0

A practical guide to running MGXBench benchmarks in multi-GPU environments, comparing communication models, generating plots, and analyzing execution profiles from the command line. Adapt the commands shown to the resources and modules available on your cluster.

## Table of Contents

1. [Choose a Workflow](#workflow)
2. [Environment Setup and Installation](#installation)
3. [Run Your First Benchmark and Compare 1D vs. 2D](#first-benchmark)
4. [Output Files and Performance Metrics](#outputs)
5. [Benchmark Campaigns Across Matrix Sizes and Communication Models](#campaigns)
6. [Strong and Weak Scaling](#scaling)
7. [Reports and Visualizations](#plots)
8. [Profiling](#profiling)
9. [Methodology and Reproducibility](#methodology)
10. [Tool Validation](#validation)
11. [Cleaning the Build](#cleaning)

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
| Capture execution activity | `tools/profile_rank.sh`, via MPI | Per-rank Nsight capture |
| Inspect capture statistics | `nsys stats` | Terminal tables or CSV |
| Plot a kernel summary | `configuration/plot_nsys_stats.py` | Per-kernel bar charts in PNG/PDF |


<a id="installation"></a>
## 2. Environment Setup and Installation

### 2.1 Navigate to the project directory

```bash
[username ~]$ cd MGXBench
```

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
Example module configuration for a specific SDumont environment (module names and versions may vary across clusters):

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

The documented installer expects the static library `$NVSHMEM_HOME/lib/libnvshmem.a`. Before building, make sure `NVSHMEM_HOME` is set and the file exists.

The installer's default CUDA architecture is `70`. For a different architecture, set `CUDA_ARCH` to match the target GPU. The `--clean` option removes the build directory before rebuilding; use it only when a clean rebuild is needed.

### 2.3 Install visualization dependencies

```bash
[username ~]$ python3 -m pip install -r configuration/requirements-viz.txt
```

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
| `--libraries` | Models to run, without duplicate codes |
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

### 6.1 Strong scaling: fixed N, more GPUs

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

The `strong` mode requires exactly one matrix size. For each model and decomposition, compare measured execution times at the same matrix size. The 2D/SUMMA decomposition requires a square process grid (for example, 1, 2, or 4 ranks); incompatible configurations do not produce valid data points.

$$
S(p)=\frac{T(1)}{T(p)},\qquad E(p)=\frac{S(p)}{p}\times100\%.
$$

### 6.2 Weak scaling: approximately constant computational work per GPU

This is **compute-based weak scaling**, rather than memory-based weak scaling. This example requires resources and appropriate mapping for up to nine GPUs. $N$ increases approximately as $N(p) = N(1) \times p^{1/3}$ and is rounded to satisfy decomposition constraints. GPU memory usage is not held constant.

The runner aligns the theoretical size to the nearest valid multiple using
`d = lcm(--size-alignment, process-grid divisor)`, where the divisor is `p` for
1D and `sqrt(p)` for 2D. The default is `--size-alignment 4`.
<<<<<<< HEAD
For example, with `--base-size 4096 --ranks 1 4 8`, the 1D sizes are
`4096`, `6500`, and `8192`. **Multiples of four are not necessarily powers of two.**
The 2D/SUMMA implementation skips `p=8`, because its process grid must be square.
=======
For example, with `--base-size 4096 --ranks 1 2 4`, the 1D sizes are
`4096`, `6500`, and `8192`. **Multiples of four are not necessarily powers of two.**
The 2D/SUMMA implementation skips `p=4`, because its process grid must be square.
>>>>>>> f0be9e6 (Update weak scaling, .gitignore, and plots  pending error)

```bash
[username ~]$ python3 configuration/run_campaign.py \
              --mode weak  \
              --base-size 4096  \
<<<<<<< HEAD
              --ranks 1 4 8 \
=======
              --ranks 1 2 4 \
>>>>>>> f0be9e6 (Update weak scaling, .gitignore, and plots  pending error)
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

Configurations incompatible with the 2D process grid or divisibility requirements are reported and skipped. Missing data points are never fabricated.

<a id="plots"></a>
## 7. Reports and Visualizations

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
## 8. Profiling

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

### 8.2 Process All Four MPI Ranks

```bash
[username ~]$ for report in "$MGXBENCH_PROFILE_DIR"/rank-*.qdrep; do
    [ -f "$report" ] || continue

    rank_name=$(basename "$report" .qdrep)

    python3 configuration/plot_nsys_stats.py "$report" \
        --output-dir "$MGXBENCH_PROFILE_DIR/stats/$rank_name" \
        --top 15 || break
done
```

For newer captures, replace `.qdrep` with `.nsys-rep` in both the filename pattern and `basename`.

### 8.3 Interpret the Plot

The Y-axis lists kernel names. The X-axis shows each kernel's share of the **sum of recorded kernel durations**, not the application's total execution time. A 65% bar does not mean 65% GPU utilization. Concurrent kernel durations may be summed, and NCCL kernels represent communication.

The original percentages are retained when selecting the top N entries, so the bars may add up to less than $100%$. Entries with duplicate names are preserved. Depending on the capture, initialization and validation kernels may appear in the summary.

### 8.4 Try It Without Nsight Systems

```bash
[username ~]$ python3 configuration/plot_nsys_stats.py examples/nsys-stats/SYNTHETIC_gpukernsum.csv
```

<a id="methodology"></a>
## 9. Methodology and Reproducibility

### 9.1 Samples and numerical validation

Warm-up runs are excluded from statistics. Each measured sample corresponds to one process launch; in 1D, it may represent the average of several inner iterations if requested. The default runner uses one iteration for both decompositions. A native 1D run without `MGXBENCH_INNER_LOOPS` retains ten iterations; set this variable to $1$ for comparable profiling.

The numerical validation described for this version uses constant-valued matrices and an analytically expected result. It is not equivalent to a comprehensive test suite with random inputs. The timed regions retain the scope defined by each algorithm. For more comprehensive validation, future tests should include identity matrices, seeded random matrices, distinct block-wise patterns, and absolute/relative error checks against a trusted reference implementation. These additional tests are recommendations, not features claimed to be implemented in this version.

The confidence interval uses a percentile bootstrap of the mean, with $2,000$ resamples and a fixed random seed; it is not computed for a single sample. A small number of repetitions yields fragile estimates. Three repetitions are useful for testing the workflow; increase this to $10$ or $30$ depending on the objective and observed variability. Additional repetitions do not fix an unstable environment.

### 9.2 Grouping results

Reports group only data with compatible `environment_id`, `label`, and inner-iteration counts. Each comparison also accounts for decomposition, model, N, and p. Repeated runs of the same configuration combine their raw samples. Duplicate CSV summaries without samples are rejected because the underlying distribution cannot be reconstructed.

JSON takes precedence over an adjacent CSV; overlapping entries are deduplicated. Older files without environment metadata are isolated by directory to avoid invalid comparisons. If a curve or comparison is missing, check these fields before assuming a plotting error.

### 9.3 Record the environment

Use `examples/environment.json` as a template and fill in the GPU, CPU, nodes, interconnect, CUDA/MPI/NCCL/NVSHMEM versions, and build configuration:

```bash
[username ~]$ python3 benchmark.py --decomposition 1d --size 4096 --ranks 4 \
              --libraries MMM NNN --warmup 2 --repeats 3 \
              --environment examples/environment.json \
              --output results/sdumont
```

The file must contain a JSON object. Example fields containing `null` must be replaced with actual values. Automatic collection records available information about the launch host, GPU topology, CUDA compiler, MPI, modules, selected communication variables, resource allocation, and executable hashes. Failed queries remain `null`; this is not an inventory of every node.

The environment identifier is strict: changes to the GPU, host, runtime, allocation, or binaries may split results into separate groups. Supplying the same JSON file does not force different environments to be treated as identical. For scaling studies, maintain a fixed allocation and controlled process placement whenever possible.

### 9.4 GPU mapping and multi-node execution

The default `--device-map local` setting makes the runner pass `auto` to the executable. An MPI shared-memory communicator determines the local rank. If multiple GPUs are visible, the local index is selected; if only one is visible, device index zero is used.

When each process can see only one GPU, the scheduler must ensure that these are distinct physical GPUs. The program does not verify exclusivity across separate CUDA visibility spaces. `--device-map explicit` restores selection by global rank and is appropriate only when the mapping has been planned accordingly.

The default launcher is `mpirun`; placement options can be passed through `--launcher-args`. For multi-node execution, CUDA-aware MPI/NVSHMEM transport, symmetric heaps, affinity, and networking must be configured on the cluster. Local-rank support does not replace multi-node validation of each backend.

<a id="validation"></a>
## 10. Tool Validation

```bash
[username ~]$ python3 -m unittest discover -s tests -v
[username ~]$ python3 -m py_compile benchmark.py configuration/*.py
[username ~]$ bash -n tools/profile_rank.sh
[username ~]$ bash -n tools/stats_rank.sh
```

<a id="cleaning"></a>
## 11. Cleaning the Build

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
