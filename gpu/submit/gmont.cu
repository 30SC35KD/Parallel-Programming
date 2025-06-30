#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
#include <device_launch_parameters.h>
// 可以自行添加需要的头文件
typedef long long LL;
void fRead(int *a, int *b, int *n, int *p, int input_id){
    // 数据输入函数
    std::string str1 = "./nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    std::copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    fin>>*n>>*p;
    for (int i = 0; i < *n; i++){
        fin>>a[i];
    }
    for (int i = 0; i < *n; i++){   
        fin>>b[i];
    }
}

void fCheck(int *ab, int n, int input_id){
    // 判断多项式乘法结果是否正确
    std::string str1 = "./nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    for (int i = 0; i < n * 2 - 1; i++){
        int x;
        fin>>x;
        if(x != ab[i]){
            std::cout<<"多项式乘法结果错误"<<std::endl;
            return;
        }
    }
    std::cout<<"多项式乘法结果正确"<<std::endl;
    return;
}

void fWrite(int *ab, int n, int input_id){
    // 数据输出函数, 可以用来输出最终结果, 也可用于调试时输出中间数组
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    for (int i = 0; i < n * 2 - 1; i++){
        fout<<ab[i]<<'\n';
    }
}

using namespace std;
const int MAXN=300000;
int G=3;
// 类型定义
using u32 = unsigned int;
using i32 = int;
using u64 = unsigned long long;
using i64 = long long;
// 全局变量，模数
u32 m;
// 模的逆元
u32 inv;
// R^2 mod m
u32 R2;

// 计算模的逆元
u32 getinv() {
    u32 inv = m;
    for (int i = 0; i < 4; ++i) {
        inv *= 2 - inv * m;
    }
    return inv;
}

// 蒙哥马利规约核心函数

u32 reduce(u64 x) {
    u32 y = u32(x >> 32) - u32((u64(u32(x) * inv) * m) >> 32);
    return i32(y) < 0? y + m : y;
}
// 将普通整数转换到蒙哥马利域
u32 intToMont(i32 x) {
    return reduce(u64(x) * R2);
}
// 蒙哥马利域内的模加法
u32 Add(u32 x, u32 y) {
    x += y - m;
    return i32(x) < 0? x + m : x;
}

// 蒙哥马利域内的模减法
u32 Dec(u32 x, u32 y) {
    x -= y;
    return i32(x) < 0? x + m : x;
}

// 蒙哥马利域内的模乘法
u32 Mul(u32 x, u32 y) {
    return reduce(u64(x) * y);
}

// 从蒙哥马利域转换回普通整数
i32 get(u32 x) {
    return reduce(x);
}
// 蒙哥马利域内的模幂算法
u32 Pow(u32 base, u32 exponent) {
    u32 result = intToMont(1);
    u32 mont_base = base;

    while (exponent > 0) {
        if (exponent & 1) {
            result = Mul(result, mont_base);
        }
        mont_base = Mul(mont_base, mont_base);
        exponent >>= 1;
    }

    return result;
}
__device__ u32 d_reduce(u64 x,u32 m,u32 inv) {
    u32 y = u32(x >> 32) - u32((u64(u32(x) * inv) * m) >> 32);
    return i32(y) < 0? y + m : y;
}
__device__ u32 d_intToMont(i32 x,u32 R2,u32 m,u32 inv) {
    return d_reduce(u64(x) * R2,m,inv);
}
// 蒙哥马利域内的模加法
__device__ u32 d_Add(u32 x, u32 y,u32 m) {
    x += y - m;
    return i32(x) < 0? x + m : x;
}

// 蒙哥马利域内的模减法
__device__ u32 d_Dec(u32 x, u32 y,u32 m) {
    x -= y;
    return i32(x) < 0? x + m : x;
}

// 蒙哥马利域内的模乘法
__device__ u32 d_Mul(u32 x, u32 y,u32 m,u32 inv) {
    return d_reduce(u64(x) * y,m,inv);
}

// 从蒙哥马利域转换回普通整数
__device__ i32 d_get(u32 x,u32 m,u32 inv) {
    return d_reduce(x,m,inv);
}
__device__ u32 d_Pow(u32 base, u32 exponent,u32 m,u32 inv,u32 R2) {
    u32 result = d_intToMont(1,R2,m,inv);
    u32 mont_base = base;

    while (exponent > 0) {
        if (exponent & 1) {
            result = d_Mul(result, mont_base,m,inv);
        }
        mont_base = d_Mul(mont_base, mont_base,m,inv);
        exponent >>= 1;
    }

    return result;
}

// 位逆序置换核函数（基础版）
__global__ void bit_reversal_kernel(u32 *data, int n) {
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
        u32 temp = data[idx];
        data[idx] = data[reversed];
        data[reversed] = temp;
    }
}
// 点乘运算核函数
__global__ void multiply_kernel(u32 *a, u32 *b, u32 *c, int n, int p,u32 m,u32 inv) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = d_Mul(a[idx] , b[idx],m,inv); // 模运算确保结果在域内
    }
}
// 点乘运算核函数
__global__ void get_kernel(u32 *a,int n,u32 m,u32 inv) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        a[idx] = d_get(a[idx],m,inv); // 模运算确保结果在域内
    }
}
__global__ void intToMont_kernel(u32 *fa, int *a, u32 *fb, int *b, int n, u32 R2,u32 m,u32 inv) {
        
int idx = threadIdx.x + blockIdx.x * blockDim.x;
if (idx < n) 
{
// 并行转换a和b数组到Montgomery域
fa[idx] = d_intToMont(a[idx], R2,m,inv);
fb[idx] = d_intToMont(b[idx], R2,m,inv);
}
}
// 元素乘以常数的核函数
__global__ void multiply_const_kernel(u32 *a, int constant, int n, int p,u32 m,u32 inv) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        a[idx] = d_Mul(a[idx] , constant,m,inv);
    }
}
__global__ void roots_kernel(int g, int p, int k, int current_len, u32* h_wn, u32* h_wn_inv,u32 m,u32 inv,u32 R2) {
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (j < current_len / 2) {
        int root = d_Pow(g, (p - 1) / current_len,m,inv,R2);
        int root_inv = d_Pow(root, p - 2, m,inv,R2);

        int w1 = d_Pow(root, j, m,inv,R2);
        int w2 = d_Pow(root_inv, j, m,inv,R2);

        h_wn[j * (k / current_len)] = w1;
        h_wn_inv[j * (k / current_len)] = w2;
    }
}
// 蝶形运算核函数 - 全局内存版本
__global__ void ntt_butterfly_kernel(u32 *a, u32 *wn, int n, int p, int len,u32 m,u32 inv) {
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n/2) return;
    
    int half_len = len / 2;
    int i = (idx / half_len) * len;
    int j = idx % half_len;
    
    // 确保索引不会越界
    if (i + j + half_len >= n) return;
    
    int twiddle_idx = j * (n / len);
    int u = a[i + j];
    int v =d_Mul(wn[twiddle_idx] , a[i + j + half_len],m,inv);
    
    a[i + j] = d_Add(u , v,m);
    a[i + j + half_len] = d_Dec(u , v ,m) ;
}

void ntt(u32 *d_a, int n, int p, int inv_flag,u32 *d_wn,u32 m,u32 inv) {
    
    u32 g=intToMont(G);
    // 位逆序置换
    dim3 block_size(512);
    dim3 grid_size((n + block_size.x - 1) / block_size.x);
    bit_reversal_kernel<<<grid_size, block_size>>>(d_a, n);
    //cudaDeviceSynchronize();

    // 蝶形运算
    for (int len = 2; len <= n; len <<= 1) {
        ntt_butterfly_kernel<<<grid_size, block_size>>>(d_a, d_wn, n, p, len,m,inv);
        //cudaDeviceSynchronize();
    }
 
     
    if (inv_flag) {
        // 计算 n 关于 p 的逆元 inv_n
        u32 inv_n = Pow(intToMont(n), p- 2);
        u32 mont_inv_n = inv_n;
        multiply_const_kernel<<<grid_size, block_size>>>(d_a, mont_inv_n, n, p,m,inv);
        cudaDeviceSynchronize();
    }
}

// 多项式乘法函数，利用 NTT 实现
void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    //m=intToMont(p);
    m=p;
    inv = getinv();
    // 计算 R^2 mod m
    R2 = -u64(m) % m;
   
    int k = 1;
    while (k < 2 * n) {
        k <<= 1;
    }

    dim3 block_size(512);
    dim3 grid_size((k + block_size.x - 1) / block_size.x);
    u32 *d_a,*d_b, *d_wn,*d_wn_inv,*d_result;
    u32 *h_wn = new u32[k];
    u32 *h_wn_inv=new u32[k];
    int* fa,*fb;
    // 分配GPU内存
    cudaMalloc((void**)&d_a, k * sizeof(u32));
    cudaMalloc((void**)&d_b, k * sizeof(u32));
    cudaMalloc((void**)&d_wn, k * sizeof(u32));
    cudaMalloc((void**)&d_wn_inv, k * sizeof(u32));
    cudaMalloc((void**)&d_result, k * sizeof(u32));
    cudaMalloc((void**)&fa, k * sizeof(int));
    cudaMalloc((void**)&fb, k * sizeof(int));
     // 使用主机端快速幂计算root
     u32 g=intToMont(G);
     for (int len = 2; len <= k; len <<= 1) {
        int num_j_values = len / 2;
        int blockSize = 256;
        int gridSize = (num_j_values + blockSize - 1) / blockSize;
        roots_kernel<<<gridSize, blockSize>>>(g, p, k, len, d_wn, d_wn_inv,m,inv,R2);

    }
   
    // 复制数据到GPU
    cudaMemcpy(fa, a, k * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(fb, b, k * sizeof(int), cudaMemcpyHostToDevice);
    intToMont_kernel<<<grid_size, block_size>>>(d_a, fa, d_b, fb, k, R2,m,inv);
    ntt(d_a, k, p, false,d_wn,m,inv);
    ntt(d_b, k, p, false,d_wn,m,inv);
    // 点乘运算
    multiply_kernel<<<grid_size, block_size>>>(d_a, d_b, d_result, k, p,m,inv);
    cudaDeviceSynchronize();
    
    // 逆向变换
    ntt(d_result, k, p, true,d_wn_inv,m,inv);

    
    get_kernel<<<grid_size, block_size>>>(d_result,k,m,inv);
    // 复制结果回CPU
    cudaMemcpy(ab, d_result, k * sizeof(u32), cudaMemcpyDeviceToHost);
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
    int test_end = 3;
    for(int i = test_begin; i <= test_end; ++i){
        long double ans = 0;
        int n_, p_;
        memset(a,0,sizeof(a));
        memset(b,0,sizeof(b));
        memset(ab,0,sizeof(ab));
        fRead(a, b, &n_, &p_, i);
        auto Start = std::chrono::high_resolution_clock::now();
        // TODO : 将 poly_multiply 函数替换成你写的 ntt
        poly_multiply(a, b, ab, n_, p_);
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        ans += elapsed.count();
        fCheck(ab, n_, i);
        std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (us) "<<std::endl;
        // 可以使用 fWrite 函数将 ab 的输出结果打印到 files 文件夹下
        // 禁止使用 cout 一次性输出大量文件内容
        fWrite(ab, n_, i);
    }
    return 0;
}
