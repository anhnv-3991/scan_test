#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "debug.h"
#include "scan_common.h"
#include "tuple.h"

int num;
int *array;
int *result;

void
printDiff(struct timeval begin, struct timeval end)
{
  long diff;  
  diff = (end.tv_sec - begin.tv_sec) * 1000 * 1000 + (end.tv_usec - begin.tv_usec);
  printf("Diff: %ld us (%ld ms)\n", diff, diff/1000);
}

void shuffle(int *ary,int size) {    
  srand((unsigned)time(NULL));
  for(int i=0;i<size;i++){
    int j = rand()%size;
    int t = ary[i];
    ary[i] = ary[j];
    ary[j] = t;
  }
}

void createArray()
{


  array = (int *)malloc(num * sizeof(int));
  result = (int *)malloc(num * sizeof(int));


  /*
  srand((unsigned)time(NULL));
  uint *used;//usedなnumberをstoreする
  used = (uint *)calloc(SELECTIVITY,sizeof(uint));
  for(uint i=0; i<SELECTIVITY ;i++){
    used[i] = i;
  }
  uint selec = SELECTIVITY;

  //uniqueなnumberをvalにassignする
  for (uint i = 0; i < num; i++) {
    if(&(array[i])==NULL){
      printf("allocate error.\n");
      exit(1);
    }
    uint temp = rand()%selec;
    uint temp2 = used[temp];
    selec = selec-1;
    used[temp] = used[selec];

    array[i] = temp2; 
  }

  free(used);

  shuffle(array,num);

  */

  for(int i = 0; i<num ; i++){
    array[i] = 1;
    result[i] = 0;
  }

}


void join(){

  //uint *count;
  CUresult res;
  CUdevice dev;
  CUcontext ctx;
  CUdeviceptr c_dev;
  struct timeval time_scan_s,time_scan_f;
  double time_cal;


  createArray();

  /******************** GPU init here ************************************************/
  //GPU仕様のために

  res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    printf("cuInit failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuDeviceGet(&dev, 0);
  if (res != CUDA_SUCCESS) {
    printf("cuDeviceGet failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }
  res = cuCtxCreate(&ctx, 0, dev);
  if (res != CUDA_SUCCESS) {
    printf("cuCtxCreate failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  /*********************************************************************************/

  /*
   *指定したファイルからモジュールをロードする。これが平行実行されると思っていいもかな？
   *今回はjoin_gpu.cubinとcountJoinTuple.cubinの二つの関数を実行する
   */


  /*count */
  res = cuMemAlloc(&c_dev, (num+1) * sizeof(uint));
  if (res != CUDA_SUCCESS) {
    printf("cuMemAlloc (count) failed\n");
    exit(1);
  }
  
  /********************** upload lt , rt , bucket ,buck_array ,idxcount***********************/

  res = cuMemcpyHtoD(c_dev, array, num * sizeof(int));
  if (res != CUDA_SUCCESS) {
    printf("cuMemcpyHtoD (bucket) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }



  /**************************** prefix sum *************************************/
  gettimeofday(&time_scan_s, NULL);

  if(!(presum(&c_dev,(uint)num+1))){
    printf("count scan error\n");
    exit(1);
  }

  gettimeofday(&time_scan_f, NULL);


  /********************************************************************/

  res = cuMemcpyDtoH(result,c_dev,num * sizeof(uint));
  if(res != CUDA_SUCCESS){
    printf("cuMemcpyDtoH (p_sum) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  
  res = cuMemFree(c_dev);
  if (res != CUDA_SUCCESS) {
    printf("cuMemFree (count) failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  printf("********scan time**************\n");

  printDiff(time_scan_s,time_scan_f);
  printf("\n");

  printf("scan[0] = %d,scan[%d] = %d\n",result[0],num-1,result[num-1]);

  res = cuCtxDestroy(ctx);
  if (res != CUDA_SUCCESS) {
    printf("cuCtxDestroy failed: res = %lu\n", (unsigned long)res);
    exit(1);
  }

  free(array);

  /****************************************************************************/

}


int 
main(int argc,char *argv[])
{


  if(argc>2){
    printf("引数が多い\n");
    return 0;
  }else if(argc<1){
    printf("引数が足りない\n");
    return 0;
  }else{
    num=atoi(argv[1]);

    printf("num=%d\n",num);
  }

  join();

  return 0;
}
