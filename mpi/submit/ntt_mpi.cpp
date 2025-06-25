#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include<algorithm>
#include <iomanip>
#include <omp.h>
#include <chrono>
#include <mpi.h>
#include <vector>
using namespace std;

typedef long long LL;
void fRead(int *a, int *b, int *n, int *p, int input_id){
    // 数据输入函数
    std::string str1 = "/nttdata/";
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
    std::string str1 = "/nttdata/";
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
        a = barrett( a , a , p);
        b >>= 1;
    }
    return res;
}
void ntt(int* a, int n, int p, int inv,int rank,int size)
{
    

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }

    int wn[MAXN];
    for (int len = 2; len <= n; len <<= 1) {
        int root = quick_mi(G, (p - 1) / len, p);
        if (inv) root = quick_mi(root, p - 2, p);
        int w = 1;
        for (int j = 0; j < len / 2; ++j) {
            wn[j * (n / len)] = w;
            w = barrett( w , root , p);
        }
    }
    int* recvcounts = new int[size];
    int* displs = new int[size];
    int* sendbuf = new int[MAXN];
    for (int len = 2; len <= n; len <<= 1) {
        int total_blocks = n / len;
        int blocks_per_thread = (total_blocks + size - 1) / size;
        int start_block = rank * blocks_per_thread;
        int end_block = std::min(start_block + blocks_per_thread, total_blocks);
    
        // 各进程并行处理自己的区块
        for (int blk = start_block; blk < end_block; ++blk) {
            int i = blk * len;
            for (int j = 0; j < len / 2; ++j) {
                int u = a[i + j];
                int v = barrett( wn[j * (n / len)] , a[i + j + len / 2] , p);
                a[i + j] = barrett((u+v),1,p);
                a[i + j + len / 2] = barrett((u - v + p) ,1,p);
            }
        }
    if(total_blocks<=size){
       // 计算每个进程发送数据大小
        int local_blocks = std::max(0, end_block - start_block);
        int local_count = local_blocks * len;

        // 进程间需要收集所有 local_count，构造 recvcounts 和 displs
        // 这里所有进程需要知道所有进程的 local_count，使用 MPI_Allgather
       
        MPI_Allgather(&local_count, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

     
        displs[0] = 0;
        for (int i = 1; i < size; ++i)
            displs[i] = displs[i - 1] + recvcounts[i - 1];

        // 准备发送缓冲区：将负责区块拷贝到 sendbuf
      
        for (int blk = start_block; blk < end_block; ++blk) {
            memcpy(sendbuf + (blk - start_block) * len, a + blk * len, sizeof(int) * len);
        }

        // 使用 MPI_Allgatherv 收集各进程数据到全局数组 a
        MPI_Allgatherv(sendbuf, local_count, MPI_INT, a, recvcounts, displs, MPI_INT, MPI_COMM_WORLD);
    }
    }
    


    if (inv) {
        int inv_n = quick_mi(n, p - 2, p);
        for (int i = 0; i < n; ++i) a[i] = barrett(a[i] , inv_n , p);
    }
}
void poly_multiply(int* a, int* b, int* ab, int n, int p,int rank, int size) {

    r=m/p;
    int fa[MAXN] = { 0 }, fb[MAXN] = { 0 };
    for (int i = 0; i < n; ++i) {
        fa[i] = a[i];
        fb[i] = b[i];
    }
    int m = 1;
    while (m < 2 * n) m <<= 1;
    ntt(fa, m, p, false,rank, size);
    ntt(fb, m, p, false,rank, size);

    for (int i = 0; i < m; ++i) {
        fa[i] = barrett(fa[i] , fb[i] , p);
    }
    ntt(fa, m, p, true,rank,size);

    for (int i = 0; i < 2 * n - 1; ++i) {
        ab[i] = fa[i];
    }
}
int a[MAXN], b[MAXN], ab[MAXN];
int main(int argc, char *argv[])
{
    // 保证输入的所有模数的原根均为 3
    int test_begin = 0;
    int test_end = 3;
    int rank, size;
    MPI_Init(&argc, &argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for(int i = test_begin; i <= test_end; ++i){
        double ans = 0.0;
        int n_, p_;
        if(rank == 0){
            fRead(a, b, &n_, &p_, i);
        }
        
        // // 广播 n_ 和 p_ 到所有进程
        MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // 广播 a, b
        MPI_Bcast(a, MAXN, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(b, MAXN, MPI_INT, 0, MPI_COMM_WORLD);
        
        // 确保所有进程都准备好
        MPI_Barrier(MPI_COMM_WORLD);

        double Start = MPI_Wtime();
        poly_multiply(a, b, ab, n_, p_, rank, size);
        double End = MPI_Wtime();

        ans = (End - Start) * 1000; // 转成 ms
        
        // rank 0 输出
        if(rank == 0){
            fCheck(ab, n_, i);
            std::cout << "average latency for n = " << n_ 
                      << " p = " << p_ 
                      << " : " << ans << " (ms) " << std::endl;
            fWrite(ab, n_, i);
        }
    }

    MPI_Finalize();
    return 0;
}
