#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
// 可以自行添加需要的头文件
typedef long long LL;

using namespace std;

const int MAXN = 300000;
int G = 3;
__int128 m=(__int128)1<<64;
LL r;
LL barrett(int a, int b, int p)
{
    LL ab = (LL)a * b;
    LL q = ((__int128)ab * r) >> 64;  // r is 2^64 / p
    LL result = (LL)(ab - q * p);
    if (result >= p) result -= p;
    if (result < 0) result += p;
    return result;
}
LL quick_mi(int a, int b, int p)
{
    LL res = 1 % p;
    while (b)
    {
        if (b & 1) res = barrett(res , a , p);
        a = barrett(a ,a , p);
        b >>= 1;
    }
    return res;
}
__device__ LL d_barrett(int a, int b, int p,LL r)
{
    LL ab = (LL)a * b;
    LL q = ((__int128)ab * r) >> 64;  // r is 2^64 / p
    LL result = (LL)(ab - q * p);
    if (result >= p) result -= p;
    if (result < 0) result += p;
    return result;
}
__device__ LL d_quick_mi(int a, int b, int p,LL r)
{
    LL res = 1 % p;
    while (b)
    {
        if (b & 1) res = d_barrett(res , a , p,r);
        a = d_barrett(a ,a , p,r);
        b >>= 1;
    }
    return res;
}
// 位逆序置换核函数（基础版）
__global__ void bit_reversal_kernel(int *data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;
    
    // 确定n的有效位数m（即n-1的最高位位置+1）
    int m = 0;
    int nn = n - 1;
    while (nn > 0) {
        nn >>= 1;
        m++;
    }
    
    // 计算逆序后的索引
    int reversed = 0;
    int temp = idx;
    for (int i = 0; i < m; i++) {
        reversed = (reversed << 1) | (temp & 1);
        temp >>= 1;
    }
    
    // 交换元素（只处理idx < reversed的情况，避免重复交换）
    if (idx < reversed && reversed < n) {
        int temp = data[idx];
        data[idx] = data[reversed];
        data[reversed] = temp;
    }
}
__global__ void roots_kernel(int G, int p, int m, int current_len, int* h_wn, int* h_wn_inv,LL r) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (j < current_len / 2) {
        int root = d_quick_mi(G, (p - 1) / current_len, p,r);
        int root_inv = d_quick_mi(root, p - 2, p,r);

        int w1 = d_quick_mi(root, j, p,r);
        int w2 = d_quick_mi(root_inv, j, p,r);

        h_wn[j * (m / current_len)] = w1;
        h_wn_inv[j * (m / current_len)] = w2;
    }
}
// 元素乘以常数的核函数
__global__ void multiply_const_kernel(int *a, int constant, int n, int p,LL r) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        a[idx] = d_barrett(a[idx] , constant,p,r);
    }
}
// 蝶形运算核函数 - 全局内存版本
__global__ void ntt_butterfly_kernel(int *a, int *wn, int n, int p, int len,LL r) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n/2) return;
    
    int half_len = len / 2;
    int i = (idx / half_len) * len;
    int j = idx % half_len;
    
    // 确保索引不会越界
    if (i + j + half_len >= n) return;
    
    int twiddle_idx = j * (n / len);
    int u = a[i + j];
    int v = d_barrett(wn[twiddle_idx] , a[i + j + half_len] , p,r);
    
    a[i + j] = d_barrett((u + v) ,1, p,r);
    a[i + j + half_len] = d_barrett((u - v + p) ,1, p,r);
}
// 点乘运算核函数
__global__ void multiply_kernel(int *a, int *b, int *c, int n, int p,LL r) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = d_barrett(a[idx] , b[idx] , p,r); // 模运算确保结果在域内
    }
}
void ntt(int *d_a, int n, int p, int inv, int G,int* d_wn,LL r)
{

    // 位逆序置换
    dim3 block_size(512);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    bit_reversal_kernel<<<grid_size, block_size>>>(d_a, n);
   
    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float milliseconds = 0;

    // 开始计时
    cudaEventRecord(start);

    // 蝶形运算
    for (int len = 2; len <= n; len <<= 1) {
        ntt_butterfly_kernel<<<grid_size, block_size>>>(d_a, d_wn, n, p, len, r);
        // 注意：不要在这里同步，否则会破坏并行性
    }

    // 停止计时前需要确保所有内核都已完成
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算时间差
    cudaEventElapsedTime(&milliseconds, start, stop);
    if(n>10)
    printf("蝶形运算总时间: %.3f ms\n", milliseconds);

    // 清理事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    if (inv) {
        int inv_n = quick_mi(n, p - 2, p);
        multiply_const_kernel<<<grid_size, block_size>>>(d_a, inv_n, n, p,r);
        cudaDeviceSynchronize();
    }
}
void poly_multiply(int* a, int* b, int* ab, int n, int p) {

    r=m/p;
    int m = 1;
    while (m < 2 * n) m <<= 1;
    dim3 block_size(512);
    dim3 grid_size((m + block_size.x - 1) / block_size.x);
    int *d_a,*d_b, *d_wn,*d_wn_inv,*d_result;
    int *h_wn = new int[m];
    int *h_wn_inv=new int[m];
    
    // 分配GPU内存
    cudaMalloc((void**)&d_a, m * sizeof(int));
    cudaMalloc((void**)&d_b, m * sizeof(int));
    cudaMalloc((void**)&d_wn, m * sizeof(int));
    cudaMalloc((void**)&d_wn_inv, m * sizeof(int));
    cudaMalloc((void**)&d_result, m * sizeof(int));
     for (int len = 2; len <= m; len <<= 1) {
        int num_j_values = len / 2;
        int blockSize = 256;
        int gridSize = (num_j_values + blockSize - 1) / blockSize;
        roots_kernel<<<gridSize, blockSize>>>(G, p, m, len, d_wn, d_wn_inv,r);
    }
    // 复制数据到GPU
    cudaMemcpy(d_a, a, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, m * sizeof(int), cudaMemcpyHostToDevice);

    // 正向变换
    ntt(d_a, m, p, false, 3,d_wn,r);
    ntt(d_b, m, p, false, 3,d_wn,r);

    // 点乘运算
    multiply_kernel<<<grid_size, block_size>>>(d_a, d_b, d_result, m, p,r);
    cudaDeviceSynchronize();
    
    // 逆向变换
    ntt(d_result, m, p, true, 3,d_wn_inv,r);
    // 复制结果回CPU
    cudaMemcpy(ab, d_result, m * sizeof(int), cudaMemcpyDeviceToHost);
    
    // 释放GPU内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_wn);
    cudaFree(d_result);
}
int a[300000], b[300000], ab[300000];
int main(int argc, char *argv[])
{
    
    // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1 的形式
    // 输入模数分别为 7340033 104857601 469762049 263882790666241
    // 第四个模数超过了整型表示范围, 如果实现此模数意义下的多项式乘法需要修改框架
    // 对第四个模数的输入数据不做必要要求, 如果要自行探索大模数 NTT, 请在完成前三个模数的基础代码及优化后实现大模数 NTT
    // 输入文件共五个, 第一个输入文件 n = 4, 其余四个文件分别对应四个模数, n = 131072
    // 在实现快速数论变化前, 后四个测试样例运行时间较久, 推荐调试正确性时只使用输入文件 1
    int test_begin = 0;
    int test_end = 8;
    int N[8]={4,8,256,1024,4096,16384,65536,131072};
    for(int i = test_begin; i < test_end; ++i){
        long double ans = 0;
        int n_=N[i], p_=104857601;
        for (int j = 0; j < n_; ++j) {
            a[j] = std::rand() % 2001 - 1000;
            b[j] = std::rand() % 2001 - 1000;
        }
        if(i==0) for(int j=0;j<100;j++)poly_multiply(a, b, ab, n_, p_);
        else{
        auto Start = std::chrono::high_resolution_clock::now();
        // TODO : 将 poly_multiply 函数替换成你写的 ntt
        poly_multiply(a, b, ab, n_, p_);
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        ans += elapsed.count();
        
        std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (ms) "<<std::endl;
        }
        // 可以使用 fWrite 函数将 ab 的输出结果打印到 files 文件夹下
        // 禁止使用 cout 一次性输出大量文件内容

    }
    return 0;
}
