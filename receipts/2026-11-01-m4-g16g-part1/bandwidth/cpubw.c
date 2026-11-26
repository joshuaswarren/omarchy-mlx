// Multi-threaded CPU read bandwidth: each thread xors its slice of a 1 GiB
// buffer with four independent 16-byte NEON loads per 64-byte block; a
// slice is a whole number of 64-byte blocks except the last, whose tail
// bytes are read one at a time. Allocation and thread errors stop the run.
// Prints decimal GB/s (bytes read / best of 5 wall-clock passes).
#include <arm_neon.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static uint8_t *buf; static size_t total; static int nt;
static volatile uint64_t sink;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void *work(void *arg){
  size_t id=(size_t)arg, blocks=total/64, per=blocks/nt;
  size_t b0=id*per, b1=(id+1==(size_t)nt)?blocks:b0+per;
  const uint64x2_t *p=(const uint64x2_t*)(buf+b0*64);
  uint64x2_t a=vdupq_n_u64(0),b=a,c=a,d=a;
  for(size_t i=0;i<(b1-b0)*4;i+=4){a=veorq_u64(a,p[i]);b=veorq_u64(b,p[i+1]);c=veorq_u64(c,p[i+2]);d=veorq_u64(d,p[i+3]);}
  uint64x2_t r=veorq_u64(veorq_u64(a,b),veorq_u64(c,d));
  uint64_t x=vgetq_lane_u64(r,0)^vgetq_lane_u64(r,1);
  if(id+1==(size_t)nt) for(size_t i=blocks*64;i<total;i++) x^=buf[i];
  return (void*)(uintptr_t)x;
}
int main(void){
  total=(size_t)1<<30; buf=aligned_alloc(4096,total);
  if(!buf){fprintf(stderr,"FAIL alloc\n");return 1;}
  memset(buf,1,total);
  int counts[]={1,2,4,6,8,10};
  for(int k=0;k<6;k++){nt=counts[k]; double best=1e9;
    for(int r=0;r<5;r++){pthread_t th[16]; double t=now();
      for(int i=0;i<nt;i++) if(pthread_create(&th[i],0,work,(void*)(size_t)i)){fprintf(stderr,"FAIL thread\n");return 1;}
      for(int i=0;i<nt;i++){void *v; if(pthread_join(th[i],&v)){fprintf(stderr,"FAIL join\n");return 1;} sink^=(uint64_t)(uintptr_t)v;}
      t=now()-t; if(t<best)best=t;}
    printf("threads=%2d read %6.1f GB/s\n",nt,total/best/1e9);}
  return 0;
}
