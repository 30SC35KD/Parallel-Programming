#include <iostream>
#include <cstring>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
#include <immintrin.h> 
#include <cstdlib>
#include <ctime>
#include <bits/stdc++.h>
typedef long long LL;
const int MAXN = 300000;
int G = 3;
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

__m128i reduce4(__m256i x) {
    
        const __m256i m_vec = _mm256_set1_epi64x(m);
        const __m256i inv_vec = _mm256_set1_epi64x(inv);
        const __m256i zero = _mm256_setzero_si256();
    
        __m256i x_lo = _mm256_and_si256(x, _mm256_set1_epi64x(0xFFFFFFFF));
        __m256i t1 = _mm256_mul_epu32(x_lo, inv_vec);
        __m256i t2 = _mm256_mul_epu32(t1, m_vec);
        __m256i t3 = _mm256_srli_epi64(t2, 32);
    
        __m256i x_hi = _mm256_srli_epi64(x, 32);
        x_hi = _mm256_and_si256(x_hi, _mm256_set1_epi64x(0xFFFFFFFF));
    
        __m256i y = _mm256_sub_epi64(x_hi, t3);
        __m256i mask = _mm256_cmpgt_epi64(zero, y);
        y = _mm256_add_epi64(y, _mm256_and_si256(mask, m_vec));
    
        // 提取 y 中每个 64-bit 的低 32 位，打包成 __m128i
        int32_t y0 = (int32_t)_mm256_extract_epi64(y, 0);
        int32_t y1 = (int32_t)_mm256_extract_epi64(y, 1);
        int32_t y2 = (int32_t)_mm256_extract_epi64(y, 2);
        int32_t y3 = (int32_t)_mm256_extract_epi64(y, 3);
    
        return _mm_set_epi32(y3, y2, y1, y0);
    
}

inline __m128i intToMont4(__m128i x) {
    __m128i sign_mask = _mm_cmpgt_epi32(_mm_setzero_si128(), x);
    __m128i abs_x = _mm_abs_epi32(x);

    // 扩展为 u64
    __m256i x_64 = _mm256_cvtepu32_epi64(abs_x);

    // x * R2
    __m256i r2_vec = _mm256_set1_epi64x(R2);
    __m256i x_r2 = _mm256_mul_epu32(x_64, r2_vec);

    // reduce
    __m128i y = reduce4(x_r2);

    __m128i m_vec = _mm_set1_epi32(m);
    __m128i neg_result = _mm_sub_epi32(m_vec, y);
    y = _mm_blendv_epi8(y, neg_result, sign_mask);

    return y;
}    
// 四路并行 Add
__m128i Add4(__m128i x, __m128i y) {
    __m128i m_vec = _mm_set1_epi32(m);
    __m128i res = _mm_add_epi32(x, _mm_sub_epi32(y, m_vec));
    __m128i mask = _mm_cmpgt_epi32(_mm_setzero_si128(), res);
    return _mm_add_epi32(res, _mm_and_si128(mask, m_vec));
}

// 四路并行 Dec
__m128i Dec4(__m128i x, __m128i y) {
    __m128i res = _mm_sub_epi32(x, y);
    __m128i mask = _mm_cmpgt_epi32(_mm_setzero_si128(), res);
    __m128i m_vec = _mm_set1_epi32(m);
    return _mm_add_epi32(res, _mm_and_si128(mask, m_vec));
}

// 四路并行 get
__m128i get4(__m128i x) {
    // 转为 64 位后乘 1，调用 reduce4
    __m256i x_64 = _mm256_cvtepu32_epi64(x);
    __m128i y = reduce4(x_64);
    return y;
}

// 四路并行 Mul
__m128i Mul4(__m128i x, __m128i y) {
    // 先将每个 u32 扩展为 u64（4 个）
    __m256i x64 = _mm256_cvtepu32_epi64(x);
    __m256i y64 = _mm256_cvtepu32_epi64(y);
    __m256i prod = _mm256_mul_epu32(x64, y64);

    // reduce4
    __m128i res = reduce4(prod);

    return res;
}

__m128i Pow4(__m128i base, __m128i exponent) {
    __m128i result = _mm_set1_epi32(intToMont(1));
    __m128i current_base = base;

    for (int bit = 31; bit >= 0; --bit) {
        __m128i bit_mask = _mm_set1_epi32(1 << bit);
        __m128i bit_test = _mm_and_si128(exponent, bit_mask);
        __m128i condition = _mm_cmpgt_epi32(bit_test, _mm_setzero_si128());

        __m128i mul_result = Mul4(result, current_base);
        result = _mm_blendv_epi8(result, mul_result, condition);

        current_base = Mul4(current_base, current_base);
    }

    return result;
}      

void dit(u32 *f, int n, int p, u32 *w) {
    u32 g, h;
    for (int len = 2; len <= n; len <<= 1) {
        for (int i = 0, t = 0; i < n; i += len, ++t) {int j=0;
            for (; j + 4 <= len / 2; j += 4) {
                __m128i g_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(f + i + j));
                __m128i h_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(f + i + j + len / 2));
                __m128i w_vec = _mm_set1_epi32(w[t]);
                __m128i add_result = Add4(g_vec, h_vec);

                __m128i dec_result = Dec4(g_vec, h_vec);

                __m128i mul_result = Mul4(dec_result, w_vec);

                // 存储结果
                _mm_storeu_si128(reinterpret_cast<__m128i*>(f + i + j), add_result);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(f + i + j + len / 2), mul_result);
            }
            for (; j < len / 2; ++j) {
                uint32_t g = f[i + j];
                uint32_t h = f[i + j + len / 2];
                f[i + j] = Add(g, h);
                f[i + j + len / 2] = Mul(Dec(g, h), w[t]);
            }
        }
    }

    u32 invl = Pow(intToMont(n), p - 2);
    __m128i invl_vec = _mm_set1_epi32(invl);
    int i = 0;
    for (; i < n; i += 4) {
        __m128i f_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(f + i));
        __m128i result_vec = Mul4(f_vec, invl_vec);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(f + i), result_vec);
    }
    for (; i < n; ++i) {
        f[i] = Mul(invl, f[i]);
    }
    std::reverse(f + 1, f + n);
}

void dif(u32 *f, int n, int p, u32 *w) {
    u32 g, h;
    for (int len = n; len > 1; len >>= 1) {
        for (int i = 0, t = 0; i < n; i += len, t++) {int j = 0;
            for (; j + 4 <= len / 2; j += 4) {
                // 加载数据
                __m128i g_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(f + i + j));
                __m128i h_vec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(f + i + j + len / 2));
                __m128i w_vec = _mm_set1_epi32(w[t]);
                __m128i mul_result = Mul4(h_vec, w_vec);
                __m128i add_result = Add4(g_vec, mul_result);

                __m128i dec_result = Dec4(g_vec, mul_result);

                _mm_storeu_si128(reinterpret_cast<__m128i*>(f + i + j), add_result);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(f + i + j + len / 2), dec_result);
            }
            for (; j < len / 2; ++j) {
                uint32_t g = f[i + j];
                uint32_t h = Mul(f[i + j + len / 2], w[t]);
                f[i + j] = Add(g, h);
                f[i + j + len / 2] = Dec(g, h);
            }
        }
    }
}

void ntt(u32 *a, int n, int p, int inv_flag) {
    u32 g = intToMont(G);

    // 
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

// 多项式乘法函数，利用 NTT 实现
void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    //m=intToMont(p);
    m=p;
    inv = getinv();
    // 计算 R^2 mod m
    R2 = -u64(m) % m;
    u32 fa[MAXN] = {0}, fb[MAXN] = {0};
    int i;
    for (i = 0; i < n; i += 4) {
        __m128i a_vec = _mm_loadu_si128((__m128i*)(a + i));
        __m128i b_vec = _mm_loadu_si128((__m128i*)(b + i));

        __m128i fa_vec = intToMont4(a_vec);
        __m128i fb_vec = intToMont4(b_vec);

        _mm_storeu_si128((__m128i*)(fa + i), fa_vec);
        _mm_storeu_si128((__m128i*)(fb + i), fb_vec);
    }
    
    int k = 1;
    while (k < 2 * n) {
        k <<= 1;
    }
    
    ntt(fa, k, p, false);
    ntt(fb, k, p, false);

    i = 0;
    for (; i  < k; i += 4) {
        __m128i x = _mm_loadu_si128(reinterpret_cast<__m128i*>(fa + i));
        __m128i y = _mm_loadu_si128(reinterpret_cast<__m128i*>(fb + i));
        __m128i result = Mul4(x, y);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(fa + i), result);
    }

    ntt(fa, k, p, true);
    i=0;
    for (; i+4 < 2 * n - 1; i+=4) {
        __m128i va = _mm_loadu_si128((__m128i*)(fa + i));
        __m128i res= get4(va);
        _mm_storeu_si128((__m128i*)(ab + i), res);
    }
    for(;i<2*n-1;i++) ab[i]=get(fa[i]);
}



int a[MAXN], b[MAXN], ab[MAXN];  // 数组类型改为 u32

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