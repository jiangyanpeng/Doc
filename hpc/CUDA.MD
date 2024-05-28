# CUDA c++
- GPU和CPU异构计算：计算密集型->GPU, 逻辑流控制->CPU
(https://github.com/ifromeast/cuda_learning)
## CUDA Workflow
### CUDA的典型工作流
  1. 先把数据从CPU内存拷贝到GPU显存上
  2. 执行GPU程序计算，以多个线程(gridDim * blockDim)一起计算，SIMT的方式执行
  3. 结果从GPU显层拷贝到CPU内存里
### CUDA的简单示例
host和device的内存物理上是隔离的，设备指针(device pointer/ gpu pointer)指向GPU的显层地址，无法再CPU端解引用，主机指针(host pointer/cpu pointer)指向CPU内存地址，无法再GPU端解引用，需要cudaMalloc()、cudaFree()...等CUDA API实现对设备显层的操作。

``` 
//nvcc simplekernel.cu -o simplekernel
#include "cuda_runtime.h"
__global__ void simplekkernel(void) { // __device__, __host__
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
}

int main() {
    const int m = 1;
    const int n = 1;
    simplekernel<<<m, n>>>(); // m = gridDim.x, n = gridDim.y
}
```
```
   threadIdx.x         threadIx.x         threadId.x         threadId.x
 - - - - - - - -     - - - - - - - -    - - - - - - - -     - - - - - - - -
|0|1|2|3|4|5|6|7|   |0|1|2|3|4|5|6|7|  |0|1|2|3|4|5|6|7|   |0|1|2|3|4|5|6|7|
  blockId.x = 0       blockId.x = 1      blockId.x = 2       blockId.x = 3

data:
|0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|
4 bolcks in total, each of them contains 8 threads
data[20] = 2(blockId.x) * 8(blockDim.x) + 4(threadIdx.x)
```