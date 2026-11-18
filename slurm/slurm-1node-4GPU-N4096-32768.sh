#!/bin/bash
#SBATCH --job-name=MGXBench-8GPU-4GPU
#SBATCH --partition=sequana_gpu_dev
#SBATCH --account=treinamento
#SBATCH --qos=normal

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

#SBATCH --output=mgx-4GPU-%j.out
#SBATCH --error=mgx-4GPU-%j.err

# Environment used in the source SLURM example.
module load nvshmem/3.1.7_cuda-11.2_sequana
module load anaconda3/2020.07_sequana
module load cmake/3.30.3_sequana

# 1D row-block, 4-GPU
#for SIZE in 4096 8192 16384 32768; do
#    echo "Running 1D: N=${SIZE}, GPU=2"
#    python3 benchmark.py \
#        --decomposition 1d \
#        --size "${SIZE}" \
#        --ranks 4 \
#        --libraries MMM YYY CCC XXX NNN WWW SSS \
#        --warmup 2 \
#        --repeats 3 \
#        --plot \
#        --output results/4GPUs/1d
#done

# 2D SUMMA, explicit rectangular 2 x 2 process grid.
for SIZE in  32768; do
    echo "Running 2D: N=${SIZE}, GPUs=4, grid=2x2"
    python3 benchmark.py \
        --decomposition 2d \
        --size "${SIZE}" \
        --ranks 4 \
        --grid-rows 2 \
        --grid-cols 2 \
        --summa-bk 256 \
        --libraries MMM YYY CCC XXX NNN WWW SSS \
        --warmup 1 \
        --repeats 1 \
        --plot \
        --output results/4GPUs/2d
done

echo ""
echo "MGXBench execution completed - 4GPU."

