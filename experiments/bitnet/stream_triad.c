/* ARMILLA-2X-0 step 0: STREAM-style bandwidth (clean-room, McCalpin's triad idea). usage: stream_triad THREADS ; arrays 3x512MB fp32. also READ-only sum (what decode does). */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(int argc,char**argv){
  int T=atoi(argv[1]); omp_set_num_threads(T); size_t N=(size_t)128<<20; /* floats per array = 512 MB */
  float*a=aligned_alloc(64,N*4),*b=aligned_alloc(64,N*4),*c=aligned_alloc(64,N*4);
  #pragma omp parallel for schedule(static)
  for(size_t i=0;i<N;i++){a[i]=0;b[i]=1;c[i]=2;}
  double bt=1e9,br=1e9; float sink=0;
  for(int r=0;r<5;r++){
    double t=now();
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<N;i++)a[i]=b[i]+3.0f*c[i];
    t=now()-t; if(t<bt)bt=t;
    t=now(); float tot=0;
    #pragma omp parallel
    { __m256 acc=_mm256_setzero_ps(); int nt=omp_get_num_threads(),id=omp_get_thread_num(); size_t lo=N/nt*id,hi=N/nt*(id+1);
      for(size_t i=lo;i<hi;i+=8)acc=_mm256_add_ps(acc,_mm256_load_ps(b+i));
      float v[8];_mm256_storeu_ps(v,acc); float s=0;for(int k=0;k<8;k++)s+=v[k];
      #pragma omp atomic
      tot+=s; }
    t=now()-t; sink+=tot; if(t<br)br=t;
  }
  printf("threads=%d triad %.1f GB/s  read-only %.1f GB/s (sink %g) load=%s\n",T,3.0*N*4/bt/1e9,1.0*N*4/br/1e9,sink,"-");
  return 0;}
