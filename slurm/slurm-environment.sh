#!/bin/bash
#SBATCH --job-name=MGXBench-Topology
#SBATCH --partition=sequana_gpu_dev
#SBATCH --account=treinamento
#SBATCH --qos=normal

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1

#SBATCH --output=mgx-topology-SD-%j.out
#SBATCH --error=mgx-topology-SD-%j.err

set -euo pipefail

# ============================================================
# MGXBench - SDumont Hardware Topology
#
# System: Santos Dumont (SDumont)
# Configuration:
#   Nodes       : 1
#   GPUs        : 4 x NVIDIA V100
#   MPI ranks   : 4 allocated (not launched)
#
# Hardware characterization:
#   CPU / NUMA / Memory
#   GPU / NVLink / PCIe
#   Mellanox / InfiniBand
#   SLURM / CUDA / MPI / Modules
# ============================================================

# ------------------------------------------------------------
# 1. Environment
# ------------------------------------------------------------

module load nvshmem/3.1.7_cuda-11.2_sequana
module load anaconda3/2020.07_sequana
module load cmake/3.30.3_sequana

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# ------------------------------------------------------------
# 2. Helper functions
# ------------------------------------------------------------

section() {
    echo ""
    echo "================================================"
    echo " $1"
    echo "================================================"
}

run_optional() {
    if command -v "$1" >/dev/null 2>&1; then
        "$@" || echo "WARNING: $1 failed (exit=$?)"
    else
        echo "WARNING: $1 is unavailable"
    fi
}

# ------------------------------------------------------------
# 3. SLURM information
# ------------------------------------------------------------

section "SLURM ALLOCATION"

echo "Date             : $(date)"
echo "Hostname         : $(hostname)"
echo "Job ID           : ${SLURM_JOB_ID:-N/A}"
echo "Partition        : ${SLURM_JOB_PARTITION:-N/A}"
echo "Nodes            : ${SLURM_JOB_NUM_NODES:-N/A}"
echo "Tasks            : ${SLURM_NTASKS:-N/A}"
echo "GPUs per node    : ${SLURM_GPUS_ON_NODE:-N/A}"
echo "CUDA visible     : ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "GPU allocation   : ${SLURM_JOB_GPUS:-N/A}"

run_optional scontrol show job "$SLURM_JOB_ID"
run_optional scontrol show hostnames "$SLURM_JOB_NODELIST"

# ------------------------------------------------------------
# 4. CPU architecture
# ------------------------------------------------------------

section "CPU ARCHITECTURE"

run_optional lscpu

section "CPU SOCKETS / CORES / THREADS"

run_optional lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE

section "CPU MODEL"

if [[ -r /proc/cpuinfo ]]; then
    grep -m 1 "model name" /proc/cpuinfo || true
fi

# ------------------------------------------------------------
# 5. NUMA topology
# ------------------------------------------------------------

section "NUMA TOPOLOGY"

run_optional numactl --hardware

section "NUMA MEMORY INFORMATION"

run_optional numastat -m

section "NUMA NODE DISTANCES"

if [[ -d /sys/devices/system/node ]]; then
    for file in /sys/devices/system/node/node*/distance; do
        [[ -f "$file" ]] || continue
        echo "$file:"
        cat "$file"
    done
fi

# ------------------------------------------------------------
# 6. System memory
# ------------------------------------------------------------

section "SYSTEM MEMORY"

run_optional free -h

section "MEMORY INFORMATION"

if [[ -r /proc/meminfo ]]; then
    cat /proc/meminfo
fi

# ------------------------------------------------------------
# 7. GPU inventory
# ------------------------------------------------------------

section "GPU INVENTORY"

run_optional nvidia-smi -L

section "GPU SUMMARY"

run_optional nvidia-smi \
    --query-gpu=index,name,uuid,memory.total,driver_version \
    --format=csv

# ------------------------------------------------------------
# 8. GPU detailed information
# ------------------------------------------------------------

section "GPU DETAILS"

run_optional nvidia-smi -q

section "GPU COMPUTE INFORMATION"

run_optional nvidia-smi \
    --query-gpu=index,name,compute_mode,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem \
    --format=csv

# ------------------------------------------------------------
# 9. GPU / CPU / NUMA topology
# ------------------------------------------------------------

section "GPU-CPU-NUMA TOPOLOGY"

run_optional nvidia-smi topo -m

section "GPU PCIE TOPOLOGY"

run_optional nvidia-smi topo -mp

section "GPU COMMUNICATION PATHS"

run_optional nvidia-smi topo -n

section "GPU NEAREST CPU"

run_optional nvidia-smi topo -c

# ------------------------------------------------------------
# 10. NVLink information
# ------------------------------------------------------------

section "NVLINK STATUS"

run_optional nvidia-smi nvlink --status

section "NVLINK CAPABILITIES"

run_optional nvidia-smi nvlink --capabilities

section "NVLINK ERROR COUNTERS"

run_optional nvidia-smi nvlink --errorcounters

# ------------------------------------------------------------
# 11. PCIe information
# ------------------------------------------------------------

section "GPU PCIE INFORMATION"

run_optional nvidia-smi \
    --query-gpu=index,name,pci.bus_id,pcie.link.gen.current,pcie.link.width.current,memory.total \
    --format=csv

section "PCIE DEVICES"

if command -v lspci >/dev/null 2>&1; then
    lspci -nn | grep -Ei \
        'NVIDIA|Mellanox|InfiniBand|Ethernet' || true
else
    echo "WARNING: lspci unavailable"
fi

section "PCIE TREE"

run_optional lspci -tv

# ------------------------------------------------------------
# 12. Mellanox / InfiniBand
# ------------------------------------------------------------

section "INFINIBAND STATUS"

run_optional ibstat

section "INFINIBAND DEVICES"

run_optional ibv_devices

section "RDMA DEVICE INFORMATION"

run_optional ibv_devinfo

section "RDMA LINKS"

run_optional rdma link show

section "INFINIBAND SYSFS INFORMATION"

if [[ -d /sys/class/infiniband ]]; then
    for device in /sys/class/infiniband/*; do
        [[ -d "$device" ]] || continue

        echo ""
        echo "Device: $(basename "$device")"

        for port in "$device"/ports/*; do
            [[ -d "$port" ]] || continue

            echo "Port: $(basename "$port")"

            for attribute in state phys_state rate; do
                if [[ -r "$port/$attribute" ]]; then
                    echo "$attribute: $(cat "$port/$attribute")"
                fi
            done
        done
    done
fi

# ------------------------------------------------------------
# 13. Network interfaces
# ------------------------------------------------------------

section "NETWORK INTERFACES"

run_optional ip -br link

section "NETWORK ADDRESSES"

run_optional ip -br addr

section "INFINIBAND NETWORK INTERFACES"

if command -v ip >/dev/null 2>&1; then
    ip -br link | grep -Ei \
        'ib|mlx|enp|eth' || true
fi

# ------------------------------------------------------------
# 14. Software environment
# ------------------------------------------------------------

section "LOADED MODULES"

module list 2>&1

section "CUDA COMPILER"

run_optional nvcc --version

section "NVIDIA DRIVER"

run_optional nvidia-smi

section "MPI VERSION"

if command -v mpirun >/dev/null 2>&1; then
    mpirun --version || true
else
    echo "WARNING: mpirun unavailable"
fi

section "MPI COMPILER"

if command -v mpicc >/dev/null 2>&1; then
    mpicc --version || true
else
    echo "WARNING: mpicc unavailable"
fi

section "NVSHMEM ENVIRONMENT"

env | grep -Ei \
    'NVSHMEM|CUDA|NCCL|UCX|OMPI|SLURM' \
    | sort || true

# ------------------------------------------------------------
# 15. CPU / GPU affinity
# ------------------------------------------------------------

section "PROCESS CPU AFFINITY"

run_optional taskset -pc $$

section "SLURM CPU BINDING"

echo "SLURM_CPU_BIND=${SLURM_CPU_BIND:-N/A}"

section "GPU VISIBILITY"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-N/A}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-N/A}"

# ------------------------------------------------------------
# 16. GPU-to-GPU communication topology
# ------------------------------------------------------------

section "GPU PEER-TO-PEER TOPOLOGY"

run_optional nvidia-smi topo -p2p r

section "GPU PEER-TO-PEER WRITE"

run_optional nvidia-smi topo -p2p w

section "GPU PEER-TO-PEER NVLINK"

run_optional nvidia-smi topo -p2p n

# ------------------------------------------------------------
# 17. Final summary
# ------------------------------------------------------------

section "MGXBENCH HARDWARE SUMMARY"

echo "System           : Santos Dumont"
echo "Hostname         : $(hostname)"
echo "Job ID           : ${SLURM_JOB_ID:-N/A}"
echo "Nodes            : ${SLURM_JOB_NUM_NODES:-N/A}"
echo "Tasks            : ${SLURM_NTASKS:-N/A}"
echo "Allocated GPUs   : ${SLURM_GPUS_ON_NODE:-N/A}"

echo ""
echo "GPU models:"
run_optional nvidia-smi \
    --query-gpu=index,name,memory.total \
    --format=csv,noheader

echo ""
echo "CPU model:"
if [[ -r /proc/cpuinfo ]]; then
    grep -m 1 "model name" /proc/cpuinfo || true
fi

echo ""
echo "NUMA nodes:"
if command -v lscpu >/dev/null 2>&1; then
    lscpu | grep -E \
        'NUMA node\(s\)|Socket\(s\)|CPU\(s\)' || true
fi

echo ""
echo "InfiniBand adapters:"
run_optional ibv_devices

# ------------------------------------------------------------
# 18. Completion
# ------------------------------------------------------------

section "MGXBENCH TOPOLOGY COMPLETED"

echo "Node       : $(hostname)"
echo "Job ID     : ${SLURM_JOB_ID:-N/A}"
echo "Date       : $(date)"
echo ""
echo "Hardware topology inspection completed."

