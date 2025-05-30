#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
// 可以自行添加需要的头文件
#include<vector>
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


using namespace std;

typedef long long LL;
const int MAXN = 300000;
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
#include <vector>
#include <algorithm>

// 基4位反转
void bit_reverse(int* a, int n) {
    vector<int> rev(n);
    int logn = 0;
    while ((1 << logn) < n) logn++;
    logn = (logn + 1) / 2; // 基4的位数
    
    for (int i = 0; i < n; i++) {
        int x = i, r = 0;
        for (int j = 0; j < logn; j++) {
            r = (r << 2) | (x & 3);
            x >>= 2;
        }
        rev[i] = r;
    }
    
    for (int i = 0; i < n; i++) {
        if (i < rev[i]) {
            swap(a[i], a[rev[i]]);
        }
    }
}
// 优化的NTT主函数
void ntt(int* a, int n, int p, int inv) {
    bit_reverse(a, n);
    
    // 预计算所有旋转因子
    std::vector<int> roots(n);
    int root = quick_mi(G, (p-1)/n, p);
    if (inv) root = quick_mi(root, p-2, p);
    
    roots[0] = 1;
    for (int i = 1; i < n; i++) {
        roots[i] = 1LL * roots[i-1] * root % p;
    }
    
    // 处理所有可能的基4层
    for (int len = 4; len <= n; len <<= 2) {
        int m = n / len;
        
        #pragma omp parallel for
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < len/4; j++) {
                int idx = j * m;
                int w = roots[idx];
                int w2 = 1LL * w * w % p;
                int w3 = 1LL * w2 * w % p;
                
                int k0 = i + j;
                int k1 = k0 + len/4;
                int k2 = k1 + len/4;
                int k3 = k2 + len/4;
                
                int a0 = a[k0];
                int a1 = 1LL * a[k1] * w % p;
                int a2 = 1LL * a[k2] * w2 % p;
                int a3 = 1LL * a[k3] * w3 % p;
                
                // 基4蝶形运算
                int t0 = (a0 + a2) % p;
                int t2 = (a0 - a2 + p) % p;
                int t1 = (a1 + a3) % p;
                int t3 = (1LL * (a1 - a3 + p) * roots[n/4]) % p;  // 使用预计算的旋转因子
                
                a[k0] = (t0 + t1) % p;
                a[k1] = (t2 + t3) % p;
                a[k2] = (t0 - t1 + p) % p;
                a[k3] = (t2 - t3 + p) % p;
            }
        }
    }
    
    // 归一化
    if (inv) {
        int inv_n = quick_mi(n, p-2, p);
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            a[i] = 1LL * a[i] * inv_n % p;
        }
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
    int test_end = 4;
    for(int i = test_begin; i <= test_end; ++i){
        long double ans = 0;
        int n_, p_;
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
