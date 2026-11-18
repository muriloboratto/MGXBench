#!/bin/bash
#SBATCH --job-name=MGXBench-8GPUs
#SBATCH --partition=sequana_gpu_dev
#SBATCH --account=treinamento
#SBATCH --qos=normal

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

#SBATCH --output=mgx-8GPU-%j.out
#SBATCH --error=mgx-8GPU-%j.err

# Environment used in the source SLURM example.
module load nvshmem/3.1.7_cuda-11.2_sequana
module load anaconda3/2020.07_sequana
module load cmake/3.30.3_sequana

# 1D row-block, 8-GPU
for SIZE in 32768; do
    echo "Running 1D: N=${SIZE}, GPU=8"
    python3 benchmark.py \
        --decomposition 1d \
        --size "${SIZE}" \
        --ranks 8 \
        --libraries MMM YYY CCC XXX NNN WWW SSS \
        --warmup 1 \
        --repeats 1 \
        --plot \
        --output results/8GPUs/1d_32768
done

# 2D SUMMA, explicit rectangular 2 x 4 process grid.
for SIZE in 32768; do
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
        --repeats 1 \
        --plot \
        --output results/8GPUs/2d_32768
done

echo ""
echo "MGXBench execution completed - 8GPU."

