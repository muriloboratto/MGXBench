/* MGXBench: rectangular 2D SUMMA, row-major Pr x Pc process grid.
 * Distributed C blocks (N/Pr)x(N/Pc). A row slabs and B column slabs
 * are held by roots of their row/column communicators; K panels are
 * broadcast or fetched from those roots. All backends share the same math.
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
#include <limits.h>
#include <vector>
#include <algorithm>

extern void ABMultiplyAccumulateAsync(const double *, const double *, double *, int,int,int,int,int,int,int,int,int,cudaStream_t);
extern int validate_matrix_C(const double *, int, int, double, double, double, int);
#define TILE_DIM 32
#define CHECK_CUDA(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));MPI_Abort(MPI_COMM_WORLD,7);} }while(0)
#define CHECK_NCCL(x) do { ncclResult_t e=(x); if(e!=ncclSuccess){fprintf(stderr,"NCCL %s:%d: %s\n",__FILE__,__LINE__,ncclGetErrorString(e));MPI_Abort(MPI_COMM_WORLD,8);} }while(0)

static int env_int(const char *name,int fallback){const char *s=getenv(name);return s&&*s?atoi(s):fallback;}
static void fail(const char *msg,int rank){if(rank==0)fprintf(stderr,"2D SUMMA: %s\n",msg);MPI_Abort(MPI_COMM_WORLD,2);}
static void grid_shape(int p,int *pr,int *pc){
  int r=env_int("MGXBENCH_GRID_ROWS",0),c=env_int("MGXBENCH_GRID_COLS",0);
  if(r<0||c<0) { *pr=0;*pc=0;return; }
  if(r&&c){*pr=r;*pc=c;return;}
  if(r){*pr=r;*pc=(p%r==0?p/r:0);return;}
  if(c){*pc=c;*pr=(p%c==0?p/c:0);return;}
  int dims[2]={0,0};MPI_Dims_create(p,2,dims);
  *pr=std::min(dims[0],dims[1]);*pc=std::max(dims[0],dims[1]);
}
static void pack_a(const double *src,double *dst,int N,int m,int start,int k){
  for(int i=0;i<m;i++)memcpy(dst+(size_t)i*k,src+(size_t)i*N+start,(size_t)k*sizeof(double));
}
static void pack_b(const double *src,double *dst,int n,int start,int k){
  memcpy(dst,src+(size_t)start*n,(size_t)k*n*sizeof(double));
}
static void unpack_c(double *dst,const double *src,int N,int m,int n,int r,int c){
  for(int i=0;i<m;i++)memcpy(dst+(size_t)(r*m+i)*N+c*n,src+(size_t)i*n,(size_t)n*sizeof(double));
}
int main(int argc,char **argv){
  MPI_Init(&argc,&argv);
  int rank,p;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&p);
  if(argc!=4)fail("Usage: mm_2d <auto|device> <N> <MMM|YYY|CCC|XXX|NNN|WWW|SSS>",rank);
  int N=atoi(argv[2]);const char *lib=argv[3];
  const char *codes[]={"MMM","YYY","CCC","XXX","NNN","WWW","SSS"};
  bool supported=false;for(int i=0;i<7;i++)if(!strcmp(lib,codes[i]))supported=true;
  if(!supported||N<=0)fail("invalid matrix size or library",rank);
  int Pr,Pc;grid_shape(p,&Pr,&Pc);
  if(Pr<=0||Pc<=0||(long long)Pr*Pc!=p)fail("invalid process grid: rows*cols must equal MPI ranks",rank);
  if(N%Pr||N%Pc)fail("N must be divisible by both process-grid dimensions",rank);
  int m=N/Pr,n=N/Pc,rr=rank/Pc,cc=rank%Pc;
  int BK=env_int("MGXBENCH_SUMMA_BK",256);
  if(BK<=0)fail("MGXBENCH_SUMMA_BK must be positive",rank);
  BK=std::min(BK,N);
  if((size_t)m*n>INT_MAX||(size_t)m*BK>INT_MAX||(size_t)n*BK>INT_MAX)fail("MPI count overflow",rank);
  MPI_Comm row_comm,col_comm,local_comm;
  MPI_Comm_split(MPI_COMM_WORLD,rr,cc,&row_comm);
  MPI_Comm_split(MPI_COMM_WORLD,cc,rr,&col_comm);
  MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&local_comm);
  int local_rank=0,device_count=0;MPI_Comm_rank(local_comm,&local_rank);
  CHECK_CUDA(cudaGetDeviceCount(&device_count));
  int dev=!strcmp(argv[1],"auto")?(device_count==1?0:local_rank):atoi(argv[1]);
  if(dev<0||dev>=device_count)fail("GPU device mapping invalid",rank);
  CHECK_CUDA(cudaSetDevice(dev));MPI_Comm_free(&local_comm);
  bool nv=!strcmp(lib,"WWW")||!strcmp(lib,"SSS");
  bool nc=!strcmp(lib,"NNN");
  bool host=!strcmp(lib,"MMM")||!strcmp(lib,"YYY");
  bool async=!strcmp(lib,"YYY")||!strcmp(lib,"XXX")||!strcmp(lib,"NNN")||!strcmp(lib,"SSS");
  if(nv){nvshmemx_init_attr_t attr;memset(&attr,0,sizeof(attr));MPI_Comm comm=MPI_COMM_WORLD;attr.mpi_comm=&comm;nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM,&attr);}
  if(rank==0)printf("SUMMA rectangular grid=%dx%d N=%d BK=%d library=%s ranks=%d\n",Pr,Pc,N,BK,lib,p);
  // Root of each row owns A's full row slab; root of each column owns B's full column slab.
  std::vector<double> Arow(cc==0?(size_t)m*N:0);
  std::vector<double> Bcol(rr==0?(size_t)N*n:0);
  if(cc==0)std::fill(Arow.begin(),Arow.end(),1.0);
  if(rr==0)std::fill(Bcol.begin(),Bcol.end(),2.0);
  size_t acap=(size_t)m*BK, bcap=(size_t)BK*n, csize=(size_t)m*n;
  size_t abytes=acap*sizeof(double),bbytes=bcap*sizeof(double),cbytes=csize*sizeof(double);
  double *dC=nullptr,*dA=nullptr,*dB=nullptr,*sA=nullptr,*sB=nullptr;
  CHECK_CUDA(cudaMalloc((void**)&dC,cbytes));CHECK_CUDA(cudaMemset(dC,0,cbytes));
  if(nv){
    // Every PE performs identical symmetric allocation order and sizes.
    sA=(double*)nvshmem_malloc(abytes);sB=(double*)nvshmem_malloc(bbytes);
    dA=(double*)nvshmem_malloc(abytes);dB=(double*)nvshmem_malloc(bbytes);
    if(!sA||!sB||!dA||!dB)fail("nvshmem_malloc failed",rank);
  }else{
    CHECK_CUDA(cudaMalloc((void**)&dA,abytes));CHECK_CUDA(cudaMalloc((void**)&dB,bbytes));
  }
  std::vector<double> hA(acap),hB(bcap);
  cudaStream_t stream;CHECK_CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
  ncclComm_t nccl_row,nccl_col;
  if(nc){
    ncclUniqueId rid,cid;
    if(cc==0)CHECK_NCCL(ncclGetUniqueId(&rid));
    if(rr==0)CHECK_NCCL(ncclGetUniqueId(&cid));
    MPI_Bcast(&rid,sizeof(rid),MPI_BYTE,0,row_comm);
    MPI_Bcast(&cid,sizeof(cid),MPI_BYTE,0,col_comm);
    CHECK_NCCL(ncclGroupStart());
    CHECK_NCCL(ncclCommInitRank(&nccl_row,Pc,rid,cc));
    CHECK_NCCL(ncclCommInitRank(&nccl_col,Pr,cid,rr));
    CHECK_NCCL(ncclGroupEnd());
  }
  CHECK_CUDA(cudaDeviceSynchronize());MPI_Barrier(MPI_COMM_WORLD);
  double t0=MPI_Wtime();
  for(int k0=0;k0<N;k0+=BK){
    int kc=std::min(BK,N-k0);size_t ab=(size_t)m*kc*sizeof(double),bb=(size_t)kc*n*sizeof(double);
    if(cc==0)pack_a(Arow.data(),hA.data(),N,m,k0,kc);
    if(rr==0)pack_b(Bcol.data(),hB.data(),n,k0,kc);
    if(host){
      MPI_Request ra=MPI_REQUEST_NULL,rb=MPI_REQUEST_NULL;
      if(async){
        MPI_Ibcast(hA.data(),m*kc,MPI_DOUBLE,0,row_comm,&ra);
        MPI_Ibcast(hB.data(),kc*n,MPI_DOUBLE,0,col_comm,&rb);
        MPI_Wait(&ra,MPI_STATUS_IGNORE);MPI_Wait(&rb,MPI_STATUS_IGNORE);
      }else{
        MPI_Bcast(hA.data(),m*kc,MPI_DOUBLE,0,row_comm);
        MPI_Bcast(hB.data(),kc*n,MPI_DOUBLE,0,col_comm);
      }
      CHECK_CUDA(cudaMemcpyAsync(dA,hA.data(),ab,cudaMemcpyHostToDevice,stream));
      CHECK_CUDA(cudaMemcpyAsync(dB,hB.data(),bb,cudaMemcpyHostToDevice,stream));
    }else if(nv){
      if(cc==0)CHECK_CUDA(cudaMemcpy(sA,hA.data(),ab,cudaMemcpyHostToDevice));
      if(rr==0)CHECK_CUDA(cudaMemcpy(sB,hB.data(),bb,cudaMemcpyHostToDevice));
      // Source buffers must be ready on all PEs before remote GET.
      nvshmem_barrier_all();
      int a_pe=rr*Pc,b_pe=cc;
      if(!strcmp(lib,"WWW")){
        nvshmem_getmem(dA,sA,ab,a_pe);
        nvshmem_getmem(dB,sB,bb,b_pe);
      }else{
        nvshmemx_getmem_nbi_on_stream(dA,sA,ab,a_pe,stream);
        nvshmemx_getmem_nbi_on_stream(dB,sB,bb,b_pe,stream);
        nvshmemx_quiet_on_stream(stream);
      }
    }else{
      if(cc==0)CHECK_CUDA(cudaMemcpy(dA,hA.data(),ab,cudaMemcpyHostToDevice));
      if(rr==0)CHECK_CUDA(cudaMemcpy(dB,hB.data(),bb,cudaMemcpyHostToDevice));
      if(nc){
        CHECK_NCCL(ncclGroupStart());
        CHECK_NCCL(ncclBroadcast(dA,dA,(size_t)m*kc,ncclDouble,0,nccl_row,stream));
        CHECK_NCCL(ncclBroadcast(dB,dB,(size_t)kc*n,ncclDouble,0,nccl_col,stream));
        CHECK_NCCL(ncclGroupEnd());
      }else{
        MPI_Request ra=MPI_REQUEST_NULL,rb=MPI_REQUEST_NULL;
        if(async){
          MPI_Ibcast(dA,m*kc,MPI_DOUBLE,0,row_comm,&ra);
          MPI_Ibcast(dB,kc*n,MPI_DOUBLE,0,col_comm,&rb);
          MPI_Wait(&ra,MPI_STATUS_IGNORE);MPI_Wait(&rb,MPI_STATUS_IGNORE);
        }else{
          MPI_Bcast(dA,m*kc,MPI_DOUBLE,0,row_comm);
          MPI_Bcast(dB,kc*n,MPI_DOUBLE,0,col_comm);
        }
      }
    }
    // Both panels are packed row-major with lda=kc and ldb=n.
    ABMultiplyAccumulateAsync(dA,dB,dC,m,n,kc,0,kc,kc,n,n,TILE_DIM,stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
    if(nv)nvshmem_barrier_all();
  }
  CHECK_CUDA(cudaDeviceSynchronize());MPI_Barrier(MPI_COMM_WORLD);
  double local_elapsed=MPI_Wtime()-t0,elapsed=0;
  MPI_Reduce(&local_elapsed,&elapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  std::vector<double> localC(csize);
  CHECK_CUDA(cudaMemcpy(localC.data(),dC,cbytes,cudaMemcpyDeviceToHost));
  std::vector<double> globalC(rank==0?(size_t)N*N:0);
  if(rank==0){
    unpack_c(globalC.data(),localC.data(),N,m,n,rr,cc);
    std::vector<double> tmp(csize);
    for(int r=1;r<p;r++){
      MPI_Recv(tmp.data(),(int)csize,MPI_DOUBLE,r,901,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      unpack_c(globalC.data(),tmp.data(),N,m,n,r/Pc,r%Pc);
    }
  }else MPI_Send(localC.data(),(int)csize,MPI_DOUBLE,0,901,MPI_COMM_WORLD);
  int valid=1;
  if(rank==0){
    for(size_t i=0;i<globalC.size();i++){
      double expected=2.0*N;
      if(fabs(globalC[i]-expected)>1e-9+1e-12*fabs(expected)){
        fprintf(stderr,"2D validation failed at index=%zu: got %.17g expected %.17g\n",i,globalC[i],expected);
        valid=0;break;
      }
    }
    printf("BENCHMARK_RESULT decomposition=2D library=%s size=%d ranks=%d seconds=%.9f valid=%d loops=1 grid=%dx%d\n",lib,N,p,elapsed,valid,Pr,Pc);
    fflush(stdout);
  }
  if(nc){CHECK_NCCL(ncclCommDestroy(nccl_row));CHECK_NCCL(ncclCommDestroy(nccl_col));}
  CHECK_CUDA(cudaStreamDestroy(stream));
  if(nv){nvshmem_barrier_all();nvshmem_free(dA);nvshmem_free(dB);nvshmem_free(sA);nvshmem_free(sB);}
  else {CHECK_CUDA(cudaFree(dA));CHECK_CUDA(cudaFree(dB));}
  CHECK_CUDA(cudaFree(dC));
  if(nv)nvshmem_finalize();
  MPI_Comm_free(&row_comm);MPI_Comm_free(&col_comm);
  MPI_Finalize();return valid?0:1;
}
