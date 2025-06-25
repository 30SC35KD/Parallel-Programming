#include <iostream>
#include <cstring>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
#include <pthread.h>
#include <bits/stdc++.h>
#include <cstdlib>
#include <ctime>
#include <mpi.h>  // 引入MPI头文件
typedef long long LL;  // 全部使用__int128
using namespace std;

void fRead(LL *a, LL *b, int *n, LL *p, int input_id){
    // 数据输入函数
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    std::copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    
    // 读取输入
    long long temp_n, temp_p;
    fin >> temp_n >> temp_p;
    *n = temp_n;
    *p = temp_p;
    
    for (int i = 0; i < *n; i++){
        long long temp;
        fin >> temp;
        a[i] = temp;
    }
    for (int i = 0; i < *n; i++){   
        long long temp;
        fin >> temp;
        b[i] = temp;
    }
}

void fCheck(LL *ab, int n, int input_id){
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
        long long x;
        fin >> x;
        if(x != ab[i]){
           //std::cout << "第 " << i << " 个元素错误, 预期值为 " << x << ", 实际值为 " << (long long)ab[i] << std::endl;
            std::cout << "多项式乘法结果错误" << std::endl;
            return;
        }
    }
    std::cout << "多项式乘法结果正确" << std::endl;
    return;
}

void fWrite(LL *ab, int n, int input_id){
    // 数据输出函数
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    
    for (int i = 0; i < n * 2 - 1; i++){
        fout << (long long)ab[i] << '\n';  // 转换为long long输出
    }
}

const int MAXN = 270000;
const int MOD = 4;

LL G = 3; // 原根
//const LL M[MOD] = {7340033, 5767169,469762049,998244353,1004535809}; // 5个模数
const LL M[MOD] = {167772161 ,469762049,998244353,1004535809}; 
LL C[MOD][MAXN];

LL mul_mod(LL a, LL b, LL mod) {
    return ((__int128)a * b) % mod;
}
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
void ntt(int* a, int n, int p, int inv)
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
            w = barrett(w , root , p);
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < len / 2; ++j) {
                int u = a[i + j], v = barrett(wn[j * (n / len)] ,a[i + j + len / 2] , p);
                a[i + j] = barrett((u + v),1, p);
                a[i + j + len / 2] = barrett((u - v + p),1, p);
            }
        }
    }


    if (inv) {
        int inv_n = quick_mi(n, p - 2, p);
        for (int i = 0; i < n; ++i) a[i] = barrett(a[i] , inv_n , p);
    }
}

void poly_multiply_part(LL* a, LL* b, LL* ab, int n, LL p) {
    r=m/p;
    int fa[MAXN] = { 0 }, fb[MAXN] = { 0 };
    for (int i = 0; i < n; ++i) {
        fa[i] = a[i] % p;  // 对输入数据取模
        fb[i] = b[i] % p;  // 对输入数据取模
    }
    
    int m = 1;
    while (m < 2 * n) m <<= 1;
    
    ntt(fa, m, p, false);
    ntt(fb, m, p, false);

    for (int i = 0; i < m; ++i) {
        fa[i] = barrett(fa[i], fb[i], p);
    }
    
    ntt(fa, m, p, true);

    for (int i = 0; i < 2 * n - 1; ++i) {
        ab[i] = fa[i]%p;
    }
}

// 使用__int128的扩展欧几里得
LL exgcd(LL a, LL b, LL& x, LL& y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    LL d = exgcd(b, a % b, y, x);
    y -= a / b * x;
    return d;
}


LL inv(LL a, LL mod) {
    LL x, y;
    exgcd(a, mod, x, y); // 调用已有的exgcd函数
    x = (x % mod + mod) % mod; // 确保结果在[0, mod)范围内
    return x;
}
LL crt_merge(int i, LL P) {
    const LL* mods = M; // 五模数数组 M[5]
    
    // 第一阶段：合并前两个模数 
    LL m1 = mods[0], m2 = mods[1];
    LL a1 = (C[0][i] % m1 + m1) % m1;
    LL a2 = (C[1][i] % m2 + m2) % m2;
    
    // 计算m1在m2下的逆元
    LL inv_m1 = inv(m1, m2);
    __int128 delta = (a2 - a1 + m2) % m2;
    __int128 x = a1 + (delta * inv_m1 % m2) * m1;
    __int128 m = (__int128)m1 * m2; // 当前合并模数乘积

    // 循环合并剩余模数（从第三个开始）
    for (int k = 2; k < MOD; k++) {
        LL mk = mods[k];
        LL ak = (C[k][i] % mk + mk) % mk;
        
        // 计算当前x在mk下的余数
        __int128 x_mod_mk = (x % mk + mk) % mk;
        __int128 delta_k = (ak - x_mod_mk + mk) % mk;
        
        // 计算当前m在mk下的逆元
        LL m_mod_mk = (m % mk + mk) % mk;
        LL inv_m = inv(m_mod_mk, mk);
        __int128 term = (delta_k * inv_m) % mk;
        
        // 更新合并结果
        x = x + term * m;
        m = m * mk; // 更新合并模数乘积
    }
    
    return (x % P + P) % P; 
}
// 封装函数：支持任意模数
void poly_multiply(LL* a, LL* b, LL* ab, int n, LL p_target) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    LL local_C[MAXN];

    // 计算每个进程负责的 k
    if (rank < MOD) {
        poly_multiply_part(a, b, local_C, n, M[rank]);
        // 发送给 rank 0
        if (rank != 0) {
            MPI_Send(local_C, 2*n-1, MPI_LONG_LONG, 0, rank, MPI_COMM_WORLD);
        } else {
            // 主进程直接放到 C[0]
            memcpy(C[rank], local_C, sizeof(LL)*(2*n-1));
        }
    }

    if (rank == 0) {
        for (int k = 1; k < MOD; ++k) {
            MPI_Recv(C[k], 2*n-1, MPI_LONG_LONG, k, k, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    for (int k = 0; k < MOD; ++k) {
        MPI_Bcast(C[k], 2*n-1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    }

    int total = 2*n-1;
    int chunk_size[size];
    int displs[size];

    int base = total / size;
    int rem = total % size;
    int offset = 0;
    for (int i = 0; i < size; ++i) {
        chunk_size[i] = base + (i < rem ? 1 : 0);
        displs[i] = offset;
        offset += chunk_size[i];
    }

    LL local_ab[MAXN];
    int my_chunk = chunk_size[rank];
    int start = displs[rank];
    for (int i = 0; i < my_chunk; ++i) {
        local_ab[i] = crt_merge(start + i, p_target);
    }

    MPI_Gatherv(local_ab, my_chunk, MPI_LONG_LONG,
                ab, chunk_size, displs, MPI_LONG_LONG,
                0, MPI_COMM_WORLD);
}
LL a[MAXN], b[MAXN], ab[MAXN];

int main(int argc, char *argv[])
{
    //MPI_Init(&argc, &argv);
    int provided, required = MPI_THREAD_FUNNELED;
    
    // 初始化MPI并请求线程支持
    MPI_Init_thread(&argc, &argv, required, &provided);
    if (provided < required) {
        printf("MPI_THREAD_FUNNELED not available!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int test_begin = 0;
    int test_end = 4;
    
    for(int i = test_begin; i <= test_end; ++i){
        long double ans = 0;
        int n_;
        LL p_;
        
        if(rank == 0){
            memset(a, 0, sizeof(a));
            memset(b, 0, sizeof(b));
            memset(ab, 0, sizeof(ab));
            
            fRead(a, b, &n_, &p_, i);
        }
        
        // 广播 n_ 和 p_
        MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p_, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        // 广播多项式 a 和 b
        MPI_Bcast(a, MAXN, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(b, MAXN, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

        //auto Start = std::chrono::high_resolution_clock::now();
        auto Start=MPI_Wtime();  // 使用MPI_Wtime获取时间
        poly_multiply(a, b, ab, n_, p_);
        auto End=MPI_Wtime();  // 使用MPI_Wtime获取时间
        //auto End = std::chrono::high_resolution_clock::now();
        
        if(rank == 0){
            //std::chrono::duration<double, std::ratio<1,1000>> elapsed = End - Start;
            //ans += elapsed.count();
            ans= (End - Start) * 1000;  // 转换为微秒
            fCheck(ab, n_, i);
            std::cout << "average latency for n = " << n_ << " p = " << (long long)p_ << " : " << ans << " (ms) " << std::endl;
            
            fWrite(ab, n_, i);
        }
    }

    MPI_Finalize();
    return 0;
}
