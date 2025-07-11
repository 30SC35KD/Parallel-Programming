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
#include<climits>
#include <ctime>
typedef long long LL;  // 全部使用__int128


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
    fin>>*n>>*p;
    for (int i = 0; i < *n; i++){
        fin>>a[i];
    }
    for (int i = 0; i < *n; i++){   
        fin>>b[i];
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
        LL x;
        fin>>x;
        if(x != ab[i]){
            std::cout<<"多项式乘法结果错误"<<std::endl;
            return;
        }
    }
    std::cout<<"多项式乘法结果正确"<<std::endl;
    return;
}

void fWrite(LL *ab, int n, int input_id){
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

typedef long long LL;
const int MAXN = 300000;
int G = 3;
__int128 m=(__int128)1<<64;
__int128 r;
LL barrett(LL a, LL b, LL p)
{
    LL ab = (LL)a * b;
    LL q = (ab * r) >> 64;  // r is 2^64 / p
    LL result = (LL)(ab - q * p);
    if (result >= p) result -= p;
    if (result < 0) result += p;
    return result;
}
LL quick_mi(LL a, LL b, LL p)
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
void ntt(LL* a, int n, LL p, int inv)
{

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }

    LL wn[MAXN];
    for (int len = 2; len <= n; len <<= 1) {
        LL root = quick_mi(G, (p - 1) / len, p);
        if (inv) root = quick_mi(root, p - 2, p);
        LL w = 1;
        for (int j = 0; j < len / 2; ++j) {
            wn[j * (n / len)] = w;
            w = barrett(w , root , p);
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        //#pragma omp parallel for
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < len / 2; ++j) {
                LL u = a[i + j], v = barrett(wn[j * (n / len)] ,a[i + j + len / 2] , p);
                a[i + j] = barrett((u + v),1, p);
                a[i + j + len / 2] = barrett((u - v + p),1, p);
            }
        }
    }


    if (inv) {
        LL inv_n = quick_mi(n, p - 2, p);
        for (int i = 0; i < n; ++i) a[i] = barrett(a[i] , inv_n , p);
    }
}
void poly_multiply(LL* a, LL* b, LL* ab, int n, LL p) {

    r=m/p;
    LL fa[MAXN] = { 0 }, fb[MAXN] = { 0 };
    for (int i = 0; i < n; ++i) {
        fa[i] = a[i];
        fb[i] = b[i];
    }
    int m = 1;
    while (m < 2 * n) m <<= 1;
    ntt(fa, m, p, false);
    ntt(fb, m, p, false);

    for (int i = 0; i < m; ++i) {
        fa[i] = barrett(fa[i],fb[i] , p);
    }
    ntt(fa, m, p, true);

    for (int i = 0; i < 2 * n - 1; ++i) {
        ab[i] = fa[i];
    }
}
LL a[MAXN], b[MAXN], ab[MAXN];
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
        int n_; LL p_;
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
