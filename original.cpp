#include <iostream>
#include <cstring>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>

#include <bits/stdc++.h>
#include <cstdlib>
#include <ctime>
typedef long long LL;
const int MAXN=300000;
int G=3;
void poly_multiply(int *a, int *b, int *ab, int n, int p){
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            ab[i+j]=(1LL * a[i] * b[j] % p + ab[i+j]) % p;
        }
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
    int test_begin=0;
    int test_end=7;
    int N[7]={8,256,1024,4096,16384,65536,131072};
    for(int i = test_begin; i < test_end; ++i){
        long double ans = 0;
        int n_=N[i], p_=104857601;
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
        std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (us) "<<std::endl;
        // 可以使用 fWrite 函数将 ab 的输出结果打印到 files 文件夹下
        // 禁止使用 cout 一次性输出大量文件内容
    }
    return 0;
}