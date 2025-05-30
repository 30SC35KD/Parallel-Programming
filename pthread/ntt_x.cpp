#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include<algorithm>
#include <iomanip>
#include <omp.h>

using namespace std;

typedef long long LL;
const int MAXN = 30000;
int G = 3;
LL quick_mi(int a, int b, int p)
{
    LL res = 1 % p;
    while (b)
    {
        if (b & 1) res = 1LL * res * a % p;
        a = 1LL * a * a % p;
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
            w = 1LL * w * root % p;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < len / 2; ++j) {
                int u = a[i + j], v = 1LL * wn[j * (n / len)] * a[i + j + len / 2] % p;
                a[i + j] = (u + v) % p;
                a[i + j + len / 2] = (u - v + p) % p;
            }
        }
    }


    if (inv) {
        int inv_n = quick_mi(n, p - 2, p);
        for (int i = 0; i < n; ++i) a[i] = 1LL * a[i] * inv_n % p;
    }
}
void poly_multiply(int* a, int* b, int* ab, int n, int p) {

    int fa[MAXN] = { 0 }, fb[MAXN] = { 0 };
    for (int i = 0; i < n; ++i) {
        fa[i] = a[i];
        fb[i] = b[i];
    }
    int m = 1;
    while (m < 2 * n) m <<= 1;
    ntt(fa, m, p, false);
    ntt(fb, m, p, false);

    for (int i = 0; i < m; ++i) {
        fa[i] = 1LL * fa[i] * fb[i] % p;
    }
    ntt(fa, m, p, true);

    for (int i = 0; i < 2 * n - 1; ++i) {
        ab[i] = fa[i];
    }
}
int a[MAXN], b[MAXN], ab[MAXN];
int main(int argc, char *argv[])
{
    
    // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1 的形式
    // 输入模数分别为 7340033 104857601 469762049 263882790666241
    // 第四个模数超过了整型表示范围, 如果实现此模数意义下的多项式乘法需要修改框架
    // 对第四个模数的输入数据不做必要要求, 如果要自行探索大模数 NTT, 请在完成前三个模数的基础代码及优化后实现大模数 NTT
    // 输入文件共五个, 第一个输入文件 n = 4, 其余四个文件分别对应四个模数, n = 131072
    // 在实现快速数论变化前, 后四个测试样例运行时间较久, 推荐调试正确性时只使用输入文件 1
    int test_begin=0;
    int test_end=7;
    int N[7]={8,256,1024,4096,16384,65536,131072};
    for(int i = test_begin; i < test_end; ++i){
        long double ans = 0;
        int n_=N[i];int p_=2127483642;
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        memset(a, 0, sizeof(a));
        memset(b, 0, sizeof(b));
        memset(ab, 0, sizeof(ab));
        // 为数组元素随机赋值
        for (int i = 0; i < n_; ++i) {
            a[i] = rand() % 10001; // 生成0~100的整数（%取模可能导致分布不均）
            b[i] = rand() % 10001;
        }
    
        auto Start = std::chrono::high_resolution_clock::now();
        // TODO : 将 poly_multiply 函数替换成你写的 ntt
        for(int t=0;t<20;t++) poly_multiply(a, b, ab, n_, p_);
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        ans += elapsed.count();
        std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans/20<<" (us) "<<std::endl;
        // 可以使用 fWrite 函数将 ab 的输出结果打印到 files 文件夹下
        // 禁止使用 cout 一次性输出大量文件内容
    }
    return 0;
}