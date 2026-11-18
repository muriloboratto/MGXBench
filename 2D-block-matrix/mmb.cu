/* MGXBench: rectangular 2D SUMMA with two panel buffers.
 * The communication of panel k+1 may overlap the computation of panel k.
 * Memory-efficient local validation; checked CUDA calls.
 * FP64, SUMMA panel pipeline and timing boundaries are preserved.
 */
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cmath>
#include <limits.h>
#include <vector>
#include <algorithm>

extern void ABMultiplyAccumulateAsync(const double *, const double *, double *,
                                      int, int, int, int, int, int, int, int, int,
                                      cudaStream_t);

#define TILE_DIM 32

static int env_int(const char *name, int fallback) {
    const char *s = getenv(name);
    return (s && *s) ? atoi(s) : fallback;
}

static void fail(const char *message, int rank) {
    if (rank == 0) fprintf(stderr, "2D SUMMA: %s\n", message);
    MPI_Abort(MPI_COMM_WORLD, 2);
}

static void cuda_check(cudaError_t status, const char *operation, int rank) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s failed: %s\n",
                rank, operation, cudaGetErrorString(status));
        fflush(stderr);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

static void grid_shape(int p, int *pr, int *pc) {
    int r = env_int("MGXBENCH_GRID_ROWS", 0);
    int c = env_int("MGXBENCH_GRID_COLS", 0);

    if (r < 0 || c < 0) { *pr = 0; *pc = 0; return; }
    if (r && c) { *pr = r; *pc = c; return; }
    if (r) { *pr = r; *pc = (p % r == 0 ? p / r : 0); return; }
    if (c) { *pc = c; *pr = (p % c == 0 ? p / c : 0); return; }

    int dims[2] = {0, 0};
    MPI_Dims_create(p, 2, dims);
    *pr = std::min(dims[0], dims[1]);
    *pc = std::max(dims[0], dims[1]);
}

static void pack_a(const double *src, double *dst, int N, int m,
                   int start, int k) {
    for (int i = 0; i < m; ++i) {
        memcpy(dst + (size_t)i * k,
               src + (size_t)i * N + start,
               (size_t)k * sizeof(double));
    }
}

static void pack_b(const double *src, double *dst, int n,
                   int start, int k) {
    memcpy(dst, src + (size_t)start * n, (size_t)k * n * sizeof(double));
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    if (argc != 4) {
        fail("Usage: mm_2d <auto|device> <N> <MMM|YYY|CCC|XXX|NNN|WWW|SSS>", rank);
    }

    int N = atoi(argv[2]);
    const char *lib = argv[3];
    const char *codes[] = {"MMM", "YYY", "CCC", "XXX", "NNN", "WWW", "SSS"};
    bool supported = false;
    for (int i = 0; i < 7; ++i) {
        if (!strcmp(lib, codes[i])) supported = true;
    }
    if (!supported || N <= 0) fail("invalid matrix size or library", rank);

    int Pr, Pc;
    grid_shape(p, &Pr, &Pc);
    if (Pr <= 0 || Pc <= 0 || (long long)Pr * Pc != p) {
        fail("invalid process grid: rows*cols must equal MPI ranks", rank);
    }
    if (N % Pr || N % Pc) {
        fail("N must be divisible by both process-grid dimensions", rank);
    }

    int m = N / Pr;
    int n = N / Pc;
    int rr = rank / Pc;
    int cc = rank % Pc;
    int BK = env_int("MGXBENCH_SUMMA_BK", 256);
    if (BK <= 0) fail("MGXBENCH_SUMMA_BK must be positive", rank);
    BK = std::min(BK, N);

    if ((size_t)m * n > INT_MAX || (size_t)m * BK > INT_MAX ||
        (size_t)n * BK > INT_MAX) {
        fail("MPI count overflow", rank);
    }

    MPI_Comm row_comm, col_comm, local_comm;
    MPI_Comm_split(MPI_COMM_WORLD, rr, cc, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, cc, rr, &col_comm);
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &local_comm);

    int local_rank = 0, device_count = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    cuda_check(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", rank);
    int dev = !strcmp(argv[1], "auto")
                  ? (device_count == 1 ? 0 : local_rank)
                  : atoi(argv[1]);
    if (dev < 0 || dev >= device_count) fail("GPU device mapping invalid", rank);
    cuda_check(cudaSetDevice(dev), "cudaSetDevice", rank);
    MPI_Comm_free(&local_comm);

    bool nv = !strcmp(lib, "WWW") || !strcmp(lib, "SSS");
    bool nc = !strcmp(lib, "NNN");
    bool host = !strcmp(lib, "MMM") || !strcmp(lib, "YYY");
    bool async = !strcmp(lib, "YYY") || !strcmp(lib, "XXX") ||
                 !strcmp(lib, "NNN") || !strcmp(lib, "SSS");

    if (nv) {
        nvshmemx_init_attr_t attr;
        memset(&attr, 0, sizeof(attr));
        MPI_Comm comm = MPI_COMM_WORLD;
        attr.mpi_comm = &comm;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    }

    if (rank == 0) {
        printf("SUMMA rectangular grid=%dx%d N=%d BK=%d library=%s ranks=%d pipeline=2\n",
               Pr, Pc, N, BK, lib, p);
    }

    std::vector<double> Arow(cc == 0 ? (size_t)m * N : 0);
    std::vector<double> Bcol(rr == 0 ? (size_t)N * n : 0);
    if (cc == 0) std::fill(Arow.begin(), Arow.end(), 1.0);
    if (rr == 0) std::fill(Bcol.begin(), Bcol.end(), 2.0);

    size_t acap = (size_t)m * BK;
    size_t bcap = (size_t)BK * n;
    size_t csize = (size_t)m * n;
    size_t abytes = acap * sizeof(double);
    size_t bbytes = bcap * sizeof(double);
    size_t cbytes = csize * sizeof(double);

    if (rank == 0) {
        const double gib = 1073741824.0;
        const size_t panel_bytes = abytes + bbytes;
        printf("2D explicit device buffers per rank: %.6f GiB "
               "(C + panels; excludes runtime/heap reservation)\n",
               (cbytes + (nv ? 3.0 : 2.0) * panel_bytes) / gib);
        if (nv) {
            const char *heap = getenv("NVSHMEM_SYMMETRIC_SIZE");
            printf("NVSHMEM symmetric panel payload per rank: %.6f GiB; "
                   "NVSHMEM_SYMMETRIC_SIZE=%s\n",
                   3.0 * panel_bytes / gib, heap ? heap : "<runtime default>");
        }
    }

    double *dC = nullptr;
    double *dA[2] = {nullptr, nullptr};
    double *dB[2] = {nullptr, nullptr};
    double *sA = nullptr;
    double *sB = nullptr;
    double *hA[2] = {nullptr, nullptr};
    double *hB[2] = {nullptr, nullptr};

    cuda_check(cudaMalloc((void **)&dC, cbytes), "cudaMalloc", rank);
    cuda_check(cudaMemset(dC, 0, cbytes), "cudaMemset", rank);

    if (nv) {
        // Symmetric allocation order must match across all PEs.
        sA = (double *)nvshmem_malloc(abytes);
        sB = (double *)nvshmem_malloc(bbytes);
        for (int i = 0; i < 2; ++i) {
            dA[i] = (double *)nvshmem_malloc(abytes);
            dB[i] = (double *)nvshmem_malloc(bbytes);
        }
        if (!sA || !sB || !dA[0] || !dB[0] || !dA[1] || !dB[1]) {
            fail("nvshmem_malloc failed", rank);
        }
    } else {
        for (int i = 0; i < 2; ++i) {
            cuda_check(cudaMalloc((void **)&dA[i], abytes), "cudaMalloc", rank);
            cuda_check(cudaMalloc((void **)&dB[i], bbytes), "cudaMalloc", rank);
        }
    }

    // Page-locked host buffers support genuinely asynchronous H2D copies.
    for (int i = 0; i < 2; ++i) {
        cuda_check(cudaMallocHost((void **)&hA[i], abytes), "cudaMallocHost", rank);
        cuda_check(cudaMallocHost((void **)&hB[i], bbytes), "cudaMallocHost", rank);
    }

    cudaStream_t computeStream, transferStream;
    cuda_check(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags", rank);
    cuda_check(cudaStreamCreateWithFlags(&transferStream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags", rank);

    cudaEvent_t panelReady[2], computeDone[2];
    for (int i = 0; i < 2; ++i) {
        cuda_check(cudaEventCreateWithFlags(&panelReady[i], cudaEventDisableTiming), "cudaEventCreateWithFlags", rank);
        cuda_check(cudaEventCreateWithFlags(&computeDone[i], cudaEventDisableTiming), "cudaEventCreateWithFlags", rank);
    }

    ncclComm_t nccl_row, nccl_col;
    if (nc) {
        ncclUniqueId rid, cid;
        if (cc == 0) ncclGetUniqueId(&rid);
        if (rr == 0) ncclGetUniqueId(&cid);
        MPI_Bcast(&rid, sizeof(rid), MPI_BYTE, 0, row_comm);
        MPI_Bcast(&cid, sizeof(cid), MPI_BYTE, 0, col_comm);
        ncclGroupStart();
        ncclCommInitRank(&nccl_row, Pc, rid, cc);
        ncclCommInitRank(&nccl_col, Pr, cid, rr);
        ncclGroupEnd();
    }

    cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    const int numPanels = (N + BK - 1) / BK;

    for (int iteration = 0; iteration < numPanels; ++iteration) {
        int current = iteration % 2;
        int k0 = iteration * BK;
        int kc = std::min(BK, N - k0);
        size_t ab = (size_t)m * kc * sizeof(double);
        size_t bb = (size_t)kc * n * sizeof(double);

        // Only wait when the same buffer is about to be reused.
        // The previous panel can continue computing while the CPU
        // prepares/communicates this panel in the other buffer.
        if (iteration >= 2) {
            cuda_check(cudaEventSynchronize(computeDone[current]), "cudaEventSynchronize", rank);
        }

        // NVSHMEM uses a single shared source panel per PE. Before any PE
        // overwrites that source, every PE must have finished remote GETs
        // from the previous panel. This barrier is conservative and limits
        // NVSHMEM overlap; a fully pipelined variant needs two source panels.
        if (nv && iteration > 0) {
            cuda_check(cudaEventSynchronize(computeDone[1 - current]), "cudaEventSynchronize", rank);
            nvshmem_barrier_all();
        }

        if (cc == 0) pack_a(Arow.data(), hA[current], N, m, k0, kc);
        if (rr == 0) pack_b(Bcol.data(), hB[current], n, k0, kc);

        if (host) {
            MPI_Request ra = MPI_REQUEST_NULL;
            MPI_Request rb = MPI_REQUEST_NULL;

            if (async) {
                MPI_Ibcast(hA[current], m * kc, MPI_DOUBLE, 0, row_comm, &ra);
                MPI_Ibcast(hB[current], kc * n, MPI_DOUBLE, 0, col_comm, &rb);
                MPI_Wait(&ra, MPI_STATUS_IGNORE);
                MPI_Wait(&rb, MPI_STATUS_IGNORE);
            } else {
                MPI_Bcast(hA[current], m * kc, MPI_DOUBLE, 0, row_comm);
                MPI_Bcast(hB[current], kc * n, MPI_DOUBLE, 0, col_comm);
            }

            cuda_check(cudaMemcpyAsync(dA[current], hA[current], ab,
                            cudaMemcpyHostToDevice, transferStream), "cudaMemcpyAsync", rank);
            cuda_check(cudaMemcpyAsync(dB[current], hB[current], bb,
                            cudaMemcpyHostToDevice, transferStream), "cudaMemcpyAsync", rank);

        } else if (nv) {
            if (cc == 0) {
                cuda_check(cudaMemcpy(sA, hA[current], ab, cudaMemcpyHostToDevice), "cudaMemcpy", rank);
            }
            if (rr == 0) {
                cuda_check(cudaMemcpy(sB, hB[current], bb, cudaMemcpyHostToDevice), "cudaMemcpy", rank);
            }

            nvshmem_barrier_all();
            int a_pe = rr * Pc;
            int b_pe = cc;

            if (!strcmp(lib, "WWW")) {
                nvshmem_getmem(dA[current], sA, ab, a_pe);
                nvshmem_getmem(dB[current], sB, bb, b_pe);
            } else {
                nvshmemx_getmem_nbi_on_stream(dA[current], sA, ab,
                                              a_pe, transferStream);
                nvshmemx_getmem_nbi_on_stream(dB[current], sB, bb,
                                              b_pe, transferStream);
                nvshmemx_quiet_on_stream(transferStream);
            }

        } else if (nc) {
            if (cc == 0) {
                cuda_check(cudaMemcpyAsync(dA[current], hA[current], ab,
                                cudaMemcpyHostToDevice, transferStream), "cudaMemcpyAsync", rank);
            }
            if (rr == 0) {
                cuda_check(cudaMemcpyAsync(dB[current], hB[current], bb,
                                cudaMemcpyHostToDevice, transferStream), "cudaMemcpyAsync", rank);
            }

            ncclGroupStart();
            ncclBroadcast(dA[current], dA[current], (size_t)m * kc,
                          ncclDouble, 0, nccl_row, transferStream);
            ncclBroadcast(dB[current], dB[current], (size_t)kc * n,
                          ncclDouble, 0, nccl_col, transferStream);
            ncclGroupEnd();

        } else {
            // CUDA-aware MPI: root H2D must finish before MPI accesses dA/dB.
            if (cc == 0) {
                cuda_check(cudaMemcpyAsync(dA[current], hA[current], ab,
                                cudaMemcpyHostToDevice, transferStream), "cudaMemcpyAsync", rank);
            }
            if (rr == 0) {
                cuda_check(cudaMemcpyAsync(dB[current], hB[current], bb,
                                cudaMemcpyHostToDevice, transferStream), "cudaMemcpyAsync", rank);
            }
            if (cc == 0 || rr == 0) cuda_check(cudaStreamSynchronize(transferStream), "cudaStreamSynchronize", rank);

            MPI_Request ra = MPI_REQUEST_NULL;
            MPI_Request rb = MPI_REQUEST_NULL;
            if (async) {
                MPI_Ibcast(dA[current], m * kc, MPI_DOUBLE, 0, row_comm, &ra);
                MPI_Ibcast(dB[current], kc * n, MPI_DOUBLE, 0, col_comm, &rb);
                MPI_Wait(&ra, MPI_STATUS_IGNORE);
                MPI_Wait(&rb, MPI_STATUS_IGNORE);
            } else {
                MPI_Bcast(dA[current], m * kc, MPI_DOUBLE, 0, row_comm);
                MPI_Bcast(dB[current], kc * n, MPI_DOUBLE, 0, col_comm);
            }
        }

        // For MPI paths the operations above have completed on the CPU;
        // for NCCL/NVSHMEM the transfer stream tracks their completion.
        cuda_check(cudaEventRecord(panelReady[current], transferStream), "cudaEventRecord", rank);
        cuda_check(cudaStreamWaitEvent(computeStream, panelReady[current], 0), "cudaStreamWaitEvent", rank);

        ABMultiplyAccumulateAsync(dA[current], dB[current], dC,
                                  m, n, kc, 0, kc, kc, n, n,
                                  TILE_DIM, computeStream);
        cuda_check(cudaGetLastError(), "cudaGetLastError", rank);
        cuda_check(cudaEventRecord(computeDone[current], computeStream), "cudaEventRecord", rank);
    }

    // One completion wait after the last panel, not after every kernel.
    cuda_check(cudaStreamSynchronize(computeStream), "cudaStreamSynchronize", rank);
    if (nv) nvshmem_barrier_all();
    MPI_Barrier(MPI_COMM_WORLD);

    double local_elapsed = MPI_Wtime() - t0;
    double elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);

    // Release input storage before allocating the host validation block.
    std::vector<double>().swap(Arow);
    std::vector<double>().swap(Bcol);

    // Validate every local element without gathering a full global matrix.
    std::vector<double> localC(csize);
    cuda_check(cudaMemcpy(localC.data(), dC, cbytes, cudaMemcpyDeviceToHost), "cudaMemcpy", rank);
    int local_valid = 1;
    const double expected = 2.0 * N;
    const double tolerance = 1e-9 + 1e-12 * fabs(expected);
    for (size_t i = 0; i < localC.size(); ++i) {
        const double value = localC[i];
        if (!std::isfinite(value) || fabs(value - expected) > tolerance) {
            fprintf(stderr,
                    "Rank %d: 2D validation failed at local index=%zu: "
                    "got %.17g expected %.17g\n",
                    rank, i, value, expected);
            local_valid = 0;
            break;
        }
    }
    int valid = 0;
    MPI_Allreduce(&local_valid, &valid, 1, MPI_INT,
                  MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("BENCHMARK_RESULT decomposition=2D library=%s size=%d ranks=%d seconds=%.9f valid=%d loops=1 grid=%dx%d\n",
               lib, N, p, elapsed, valid, Pr, Pc);
        fflush(stdout);
    }

    if (nc) {
        ncclCommDestroy(nccl_row);
        ncclCommDestroy(nccl_col);
    }

    for (int i = 0; i < 2; ++i) {
        cuda_check(cudaEventDestroy(panelReady[i]), "cudaEventDestroy", rank);
        cuda_check(cudaEventDestroy(computeDone[i]), "cudaEventDestroy", rank);
    }
    cuda_check(cudaStreamDestroy(computeStream), "cudaStreamDestroy", rank);
    cuda_check(cudaStreamDestroy(transferStream), "cudaStreamDestroy", rank);

    if (nv) {
        nvshmem_barrier_all();
        for (int i = 0; i < 2; ++i) {
            nvshmem_free(dA[i]);
            nvshmem_free(dB[i]);
        }
        nvshmem_free(sA);
        nvshmem_free(sB);
    } else {
        for (int i = 0; i < 2; ++i) {
            cuda_check(cudaFree(dA[i]), "cudaFree", rank);
            cuda_check(cudaFree(dB[i]), "cudaFree", rank);
        }
    }
    for (int i = 0; i < 2; ++i) {
        cuda_check(cudaFreeHost(hA[i]), "cudaFreeHost", rank);
        cuda_check(cudaFreeHost(hB[i]), "cudaFreeHost", rank);
    }
    cuda_check(cudaFree(dC), "cudaFree", rank);
    if (nv) nvshmem_finalize();

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return valid ? 0 : 1;
}

