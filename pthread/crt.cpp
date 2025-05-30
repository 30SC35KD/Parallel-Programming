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
        if (b & 1) res = 1LL*res* a% p;
        a = 1LL*a* a% p;
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
            w = 1LL*w * root % p;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < len / 2; ++j) {
                int u = a[i + j], v = 1LL*wn[j * (n / len)] * a[i + j + len / 2] % p;
                a[i + j] = (u + v) % p;
                a[i + j + len / 2] = (u - v + p) % p;
            }
        }
    }

    if (invert) {
        LL inv_n = quick_mi(n, p - 2, p);
        for (int i = 0; i < n; ++i)
            a[i] = 1LL*a[i]* inv_n% p;
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
        fa[i] = 1LL*fa[i]* fb[i]% p;
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
    
    LL m1 = mods[0], m2 = mods[1];
    LL a1 = (C[0][i] % m1 + m1) % m1;
    LL a2 = (C[1][i] % m2 + m2) % m2;
    
    // 计算m1在m2下的逆元
    LL inv_m1 = inv(m1, m2);
    __int128 delta = (a2 - a1 + m2) % m2;
    __int128 x12 = a1 + (delta * inv_m1 % m2) * m1;
    __int128 m12 = (__int128)m1 * m2; // 当前合并模数乘积


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

    __int128 result = x1234 + term4 * m1234;
    
    return (result % P + P) % P; // 确保结果非负
}
// 封装函数：支持任意模数
void poly_multiply(LL* a, LL* b, LL* ab, int n, LL p_target) {
    // 对每个模数进行NTT计算
    for (int k = 0; k < MOD; ++k) {
        poly_multiply_part(a, b, C[k], n, M[k]);
    }

    // 合并结果
    for (int i = 0; i < 2 * n - 1; ++i) {
        ab[i] = crt_merge(i, p_target);
    }
}

LL a[MAXN], b[MAXN], ab[MAXN];

int main(int argc, char *argv[])
{
    int test_begin = 0;
    int test_end = 4;
    
    for(int i = test_begin; i <= test_end; ++i){
        long double ans = 0;
        int n_;
        LL p_;
        
        memset(a, 0, sizeof(a));
        memset(b, 0, sizeof(b));
        memset(ab, 0, sizeof(ab));
        
        fRead(a, b, &n_, &p_, i);
        
        auto Start = std::chrono::high_resolution_clock::now();
        poly_multiply(a, b, ab, n_, p_);
        auto End = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::ratio<1,1000>> elapsed = End - Start;
        ans += elapsed.count();
        
        fCheck(ab, n_, i);
        std::cout << "average latency for n = " << n_ << " p = " << (long long)p_ << " : " << ans << " (us) " << std::endl;
        
        fWrite(ab, n_, i);
    }
    
    return 0;
}