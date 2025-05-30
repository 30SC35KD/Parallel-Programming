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

using namespace std;

typedef long long LL;
const int MAXN=300000;
const int MOD = 5;

LL G = 3; // 原根
const LL M[MOD] = {7340033, 104857601,469762049,998244353,1004535809}; // 3个模数
LL C[MOD][MAXN];

LL mul_mod(LL a, LL b, LL mod) {
    return ((__int128)a * b) % mod;
}

LL quick_mi(LL a, LL b, LL p) {
    LL res = 1 % p;
    while (b) {
        if (b & 1) res = mul_mod(res, a, p);
        a = mul_mod(a, a, p);
        b >>= 1;
    }
    return res;
}


void ntt(LL *a, int n, LL p, bool invert) {
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    int wn[MAXN];
    for (int len = 2; len <= n; len <<= 1) {
        int root = quick_mi(G, (p - 1) / len, p);
        if (invert) root = quick_mi(root, p - 2, p);
        int w = 1;
        for (int j = 0; j < len / 2; ++j) {
            wn[j * (n / len)] = w;
            w = mul_mod(w , root , p);
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < len / 2; ++j) {
                int u = a[i + j], v = mul_mod(wn[j * (n / len)] , a[i + j + len / 2] , p);
                a[i + j] = (u + v) % p;
                a[i + j + len / 2] = (u - v + p) % p;
            }
        }
    }

    if (invert) {
        LL inv_n = quick_mi(n, p - 2, p);
        for (int i = 0; i < n; ++i)
            a[i] = mul_mod(a[i], inv_n, p);
    }
}


void poly_multiply_part(LL* a, LL* b, LL* ab, int n, LL p) {
    LL fa[MAXN] = { 0 }, fb[MAXN] = { 0 };
    for (int i = 0; i < n; ++i) {
        fa[i] = a[i] % p;  // 对输入数据取模
        fb[i] = b[i] % p;  // 对输入数据取模
    }
    
    int m = 1;
    while (m < 2 * n) m <<= 1;
    
    ntt(fa, m, p, false);
    ntt(fb, m, p, false);

    for (int i = 0; i < m; ++i) {
        fa[i] = mul_mod(fa[i], fb[i], p);
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
    const LL* mods = M; // 四模数数组 M[4]
    
    // ==== 第一阶段：合并前两个模数 ====
    LL m1 = mods[0], m2 = mods[1];
    LL a1 = (C[0][i] % m1 + m1) % m1;
    LL a2 = (C[1][i] % m2 + m2) % m2;
    
    // 计算m1在m2下的逆元
    LL inv_m1 = inv(m1, m2);
    __int128 delta = (a2 - a1 + m2) % m2;
    __int128 x12 = a1 + (delta * inv_m1 % m2) * m1;
    __int128 m12 = (__int128)m1 * m2; // 当前合并模数乘积

    // ==== 第二阶段：合并第三个模数 ====
    LL m3 = mods[2];
    LL a3 = (C[2][i] % m3 + m3) % m3;
    
    // 计算x12在m3下的余数
    __int128 x12_mod_m3 = (x12 % m3 + m3) % m3;
    __int128 delta2 = (a3 - x12_mod_m3 + m3) % m3;
    
    // 计算m12在m3下的逆元
    LL m12_mod_m3 = (m12 % m3 + m3) % m3;
    LL inv_m12 = inv(m12_mod_m3, m3);
    __int128 term2 = (delta2 * inv_m12) % m3;
    
    // 合并结果
    __int128 x123 = x12 + term2 * m12;
    __int128 m123 = m12 * m3; // 更新合并模数乘积

    // ==== 第三阶段：合并第四个模数 ====
    LL m4 = mods[3];
    LL a4 = (C[3][i] % m4 + m4) % m4;
    
    // 计算x123在m4下的余数
    __int128 x123_mod_m4 = (x123 % m4 + m4) % m4;
    __int128 delta3 = (a4 - x123_mod_m4 + m4) % m4;
    
    // 计算m123在m4下的逆元
    LL m123_mod_m4 = (m123 % m4 + m4) % m4;
    LL inv_m123 = inv(m123_mod_m4, m4);
    __int128 term3 = (delta3 * inv_m123) % m4;


    __int128 x1234 = x123 + term3 * m123;
    __int128 m1234 = m123 * m4; // 更新合并模数乘积

    LL m5=mods[4];
    LL a5 = (C[4][i] % m5 + m5) % m5;

    // 计算x1234在m5下的余数
    __int128 x1234_mod_m5 = (x1234 % m5 + m5) % m5;
    __int128 delta4 = (a5 - x1234_mod_m5 + m5) % m5;
    // 计算m1234在m5下的逆元
    LL m1234_mod_m5 = (m1234 % m5 + m5) % m5;
    LL inv_m1234 = inv(m1234_mod_m5, m5);
    __int128 term4 = (delta4 * inv_m1234) % m5;

    // 最终合并（分步取模防溢出）
    __int128 result = x1234 + term4 * m1234;
    
    return (result % P + P) % P; // 确保结果非负
}
// 封装函数：支持任意模数
void poly_multiply(LL* a, LL* b, LL* ab, int n, LL p_target) {
    // 对每个模数进行NTT计算
    #pragma omp parallel num_threads(5)
    #pragma omp for
    for (int k = 0; k < MOD; ++k) {
        poly_multiply_part(a, b, C[k], n, M[k]);
    }

    // 合并结果
    #pragma omp parallel num_threads(5)
    #pragma omp for
    for (int i = 0; i < 2 * n - 1; ++i) {
        ab[i] = crt_merge(i, p_target);
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
    int test_begin=0;
    int test_end=1;
    int N[1]={131072};
    for(int i = test_begin; i < test_end; ++i){
        long double ans = 0;
        int n_=N[i];LL  p_=1337006139375617;
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        memset(a, 0, sizeof(a));
        memset(b, 0, sizeof(b));
        memset(ab, 0, sizeof(ab));
        // 为数组元素随机赋值
        for (int i = 0; i < n_; ++i) {
            a[i] = std::rand() % 2001 - 1000;
            b[i] = std::rand() % 2001 - 1000;
        }
    
        auto Start = std::chrono::high_resolution_clock::now();
        // TODO : 将 poly_multiply 函数替换成你写的 ntt
        poly_multiply(a, b, ab, n_, p_);
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        ans += elapsed.count();
        std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans/20<<" (us) "<<std::endl;
        // 可以使用 fWrite 函数将 ab 的输出结果打印到 files 文件夹下
        // 禁止使用 cout 一次性输出大量文件内容
    }
    return 0;
}