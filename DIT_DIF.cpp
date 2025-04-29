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



void dit(u32 *f,int n,int p,u32 *w)
{
    u32 g,h;
    for (int len = 2; len <= n; len <<= 1)
		for (int i = 0,t=0; i < n; i += len,++t)
			for (int j = 0; j < len/2; ++j)
				g = f[i+j], h = f[i+j + len/2],
				f[i+j] = Add(g,h),
				f[i+j + len/2] = Mul(Dec(g ,h), w[t]) ;
	u32 invl = Pow (intToMont(n), p-2);
	for (int i = 0; i < n; ++i)
		f[i] =Mul(invl,f[i]);
	std::reverse (f+1 , f + n);
}
void dif(u32 *f,int n,int p,u32*w)
{
    u32 g, h;
	for (int len = n; len > 1; len >>= 1)
		for (int i = 0,t=0; i < n; i += len,t++)
			for (int j = 0; j <  len/2; ++j)
				g = f[i+j], h = Mul(f[i+j + len/2] ,w[t]),
				f[i+j] = Add(g , h),
				f[i +j+ len/2] = Dec( g ,h);
}
void ntt(u32 *a, int n, int p, int inv_flag) {
    u32 g=intToMont(G);
   
     // 预计算单位根
    u32 wn[MAXN];
    wn[0] = intToMont(1); u32 f;
	u32 r = Pow (g, (p-1)/n);
	for (int t = 0; (1<<(t+1)) < n; ++t) {
		wn[1<<t] = Pow (r, n>>(t+2));
        f=wn[1<<t];
		for (int x = 1<<t; x < 1<<(t+1); ++x)
			wn[x] = Mul(f , wn[x - (1<<t)]);
	}
     if(inv_flag==false) dif(a,n,p,wn);
     else dit(a,n,p,wn);
}

void poly_multiply(int *a, int *b, int *ab, int n, int p) {
 
    m=p;
    inv = getinv();
    R2 = -u64(m) % m;
    u32 fa[MAXN] = {0}, fb[MAXN] = {0};
    for (int i = 0; i < n; ++i) {
        fa[i] = intToMont(a[i]);
        fb[i] = intToMont(b[i]);
    }
    int k = 1;
    while (k < 2 * n) {
        k <<= 1;
    }
    
    ntt(fa, k, p, false);
    ntt(fb, k, p, false);
    for (int i = 0; i < k; ++i) {
        u32 mont_fa_i = fa[i];
        u32 mont_fb_i = fb[i];
        fa[i] = Mul(mont_fa_i, mont_fb_i);
    }
    // 对结果进行逆 NTT 变换
    ntt(fa, k, p, true);

    // 将结果复制到 ab 数组
    for (int i = 0; i < 2 * n - 1; ++i) {
        ab[i] = get(fa[i]);
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
        for(int i=0;i<10;i++)
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