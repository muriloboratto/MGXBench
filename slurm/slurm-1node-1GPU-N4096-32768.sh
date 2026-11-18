#!/bin/bash
#SBATCH --job-name=MGXBench-8GPU-1GPU
#SBATCH --partition=sequana_gpu_dev
#SBATCH --account=treinamento
#SBATCH --qos=normal

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1

#SBATCH --output=mgx-1GPU-%j.out
#SBATCH --error=mgx-1GPU-%j.err

# Environment used in the source SLURM example.
module load nvshmem/3.1.7_cuda-11.2_sequana
module load anaconda3/2020.07_sequana
module load cmake/3.30.3_sequana

# 1D row-block, 1-GPU
for SIZE in 4096 8192 16384 32768; do
    echo "Running 1D: N=${SIZE}, GPU=1"
    python3 benchmark.py \
        --decomposition 1d \
        --size "${SIZE}" \
        --ranks 1 \
        --libraries MMM YYY CCC XXX NNN WWW SSS \
        --warmup 2 \
        --repeats 3 \
        --plot \
        --output results/1GPU/1d
done

# 2D SUMMA, explicit rectangular 1 x 1 process grid.
for SIZE in 4096 8192 16384 32768; do
    echo "Running 2D: N=${SIZE}, GPUs=1, grid=1x1"
    python3 benchmark.py \
        --decomposition 2d \
        --size "${SIZE}" \
        --ranks 1 \
        --grid-rows 1 \
        --grid-cols 1 \
        --summa-bk 256 \
        --libraries MMM YYY CCC XXX NNN WWW SSS \
        --warmup 2 \
        --repeats 3 \
        --plot \
        --output results/1GPU/2d
done

echo ""
echo "MGXBench execution completed - 1GPU."

