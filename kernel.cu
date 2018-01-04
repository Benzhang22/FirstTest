//点乘运算
//（a,b,c）*(d,e,f)=a*d+b*e+c*f; 
//warp为32，因此将blocksPerGrid一般设置为32
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#define imin(a,b) (a<b?a:b)
const int N = 2 * 4;
const int threadsPerBlock = 256;
const int blockPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float *a, float *b, float *c)
{
	__shared__ float cache[threadsPerBlock];
	//对于GPU上启动的每个线程块，CUDA C编译器都将创建该共享变量的一个副本。线程块中的每个线程都共享这块内存

	int tid = threadIdx.x + blockDim.x*blockIdx.x;//总索引
	int cacheIndex = threadIdx.x;
	float temp = 0;

	while (tid < N)
	{
		temp += a[tid] + b[tid];
		tid += blockDim.x*gridDim.x;
	}

	cache[cacheIndex] = temp;

	__syncthreads();//保证线程块中的线程都执行完__synthreads()之前的语句

	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (cacheIndex < i) cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];//将每个block内的线程之和保存到c中
}

int main()
{
	float *a, *b, sum = 0, *partial_c;
	float *deva, *devb, *devpartial_c;
	a = new float[N];
	b = new float[N];
	partial_c = new float[blockPerGrid];
	//在GPU上分配内存
	cudaMalloc((void **)&deva, N * sizeof(float));
	cudaMalloc((void **)&devb, N * sizeof(float));
	cudaMalloc((void **)&devpartial_c, blockPerGrid * sizeof(float));

	//在CPU上为数组赋值
	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = 2 * i;
	}
	//将数组a和b传到GPU
	cudaMemcpy(deva, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devb, b, N * sizeof(float), cudaMemcpyHostToDevice);

	dot << <blockPerGrid, threadsPerBlock >> >(deva, devb, devpartial_c);

	//将数组c从GPU传到CPU
	cudaMemcpy(partial_c, devpartial_c, blockPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

	//在CPU上完成最终求和运算
	for (int i = 0; i < blockPerGrid; i++)
		sum += partial_c[i];

	printf("value %g\n", sum);

	cudaFree(deva);
	cudaFree(devb);
	cudaFree(devpartial_c);

	return 0;
}