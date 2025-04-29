#include <iostream>
#include <cstring>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
#include <immintrin.h> 
#include <bits/stdc++.h>
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
inline __m128i reduce4(__m256i x) {
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

    int32_t y0 = (int32_t)_mm256_extract_epi64(y, 0);
    int32_t y1 = (int32_t)_mm256_extract_epi64(y, 1);
    int32_t y2 = (int32_t)_mm256_extract_epi64(y, 2);
    int32_t y3 = (int32_t)_mm256_extract_epi64(y, 3);

    return _mm_set_epi32(y3, y2, y1, y0);
}


inline __m256i intToMont8(__m256i x) {
    __m256i sign_mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), x);
    __m256i abs_x = _mm256_abs_epi32(x);

    __m128i x_lo = _mm256_castsi256_si128(abs_x);             // 低 4 个 u32
    __m128i x_hi = _mm256_extracti128_si256(abs_x, 1);         // 高 4 个 u32

    __m256i x_lo_64 = _mm256_cvtepu32_epi64(x_lo);
    __m256i x_hi_64 = _mm256_cvtepu32_epi64(x_hi);
    __m256i r2_vec = _mm256_set1_epi64x(R2);
    __m256i x_lo_r2 = _mm256_mul_epu32(x_lo_64, r2_vec);
    __m256i x_hi_r2 = _mm256_mul_epu32(x_hi_64, r2_vec);

    __m128i y_lo = reduce4(x_lo_r2);
    __m128i y_hi = reduce4(x_hi_r2);

    __m256i result = _mm256_set_m128i(y_hi, y_lo);

    __m256i m_vec = _mm256_set1_epi32(m);
    __m256i neg_result = _mm256_sub_epi32(m_vec, result);
    result = _mm256_blendv_epi8(result, neg_result, sign_mask);

    return result;
}
inline __m256i Add8(__m256i x, __m256i y) {
    __m256i m_vec = _mm256_set1_epi32(m);
    __m256i res = _mm256_add_epi32(x, _mm256_sub_epi32(y, m_vec));
    __m256i mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), res);
    return _mm256_add_epi32(res, _mm256_and_si256(mask, m_vec));
}
inline __m256i Dec8(__m256i x, __m256i y) {
    __m256i res = _mm256_sub_epi32(x, y);
    __m256i mask = _mm256_cmpgt_epi32(_mm256_setzero_si256(), res);
    __m256i m_vec = _mm256_set1_epi32(m);
    return _mm256_add_epi32(res, _mm256_and_si256(mask, m_vec));
}
inline __m256i get8(__m256i x) {
    __m128i x_lo = _mm256_castsi256_si128(x);
    __m128i x_hi = _mm256_extracti128_si256(x, 1);

    __m256i x_lo_64 = _mm256_cvtepu32_epi64(x_lo);
    __m256i x_hi_64 = _mm256_cvtepu32_epi64(x_hi);

    __m128i y_lo = reduce4(x_lo_64);
    __m128i y_hi = reduce4(x_hi_64);

    return _mm256_set_m128i(y_hi, y_lo);
}

inline __m256i Mul8(__m256i x, __m256i y) {
    __m256i x64 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(x));
    __m256i y64 = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(y));
    __m256i prod_lo = _mm256_mul_epu32(x64, y64);  // 低 4 个

    x64 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(x, 1));
    y64 = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(y, 1));
    __m256i prod_hi = _mm256_mul_epu32(x64, y64);  // 高 4 个

    __m128i res_lo = reduce4(prod_lo);
    __m128i res_hi = reduce4(prod_hi);

    return _mm256_set_m128i(res_hi, res_lo);
}
inline __m256i Pow8(__m256i base, __m256i exponent) {
    __m256i one = _mm256_set1_epi32(intToMont(1));
    __m256i result = one;
    __m256i mont_base = base;

    for (int bit = 31; bit >= 0; --bit) {
        __m256i bit_mask = _mm256_set1_epi32(1 << bit);
        __m256i bit_test = _mm256_and_si256(exponent, bit_mask);
        __m256i condition = _mm256_cmpgt_epi32(bit_test, _mm256_setzero_si256());

        __m256i mul_result = Mul8(result, mont_base);
        result = _mm256_blendv_epi8(result, mul_result, condition);

        mont_base = Mul8(mont_base, mont_base);
    }

    return result;
}


void dit(u32 *f,int n,int p,u32 *w)
{
    u32 g,h;
    for (int len = 2; len <= n; len <<= 1) {
        for (int i = 0, t = 0; i < n; i += len, ++t) {int j=0;
            for (; j+8 <= len / 2; j += 8) {
                __m256i g_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(f + i + j));
                __m256i h_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(f + i + j + len / 2));
                __m256i w_vec = _mm256_set1_epi32(w[t]);
                __m256i add_result = Add8(g_vec, h_vec);
                __m256i dec_result = Dec8(g_vec, h_vec);

                __m256i mul_result = Mul8(dec_result, w_vec);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(f + i + j), add_result);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(f + i + j + len / 2), mul_result);
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
    __m256i invl_vec = _mm256_set1_epi32(invl);
    int i=0;
    for (; i < n; i += 8) {
        __m256i f_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(f + i));
        __m256i result_vec = Mul8(f_vec, invl_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(f + i), result_vec);
    }
    for (; i < n; ++i) {
        f[i] = Mul(invl, f[i]);
    }
	std::reverse (f+1 , f + n);
}
void dif(u32 *f,int n,int p,u32*w)
{
    u32 g, h;
	for (int len = n; len > 1; len >>= 1) {
        for (int i = 0, t = 0; i < n; i += len, t++) {int j=0;
            for (; j+8 <= len / 2; j += 8) {
                __m256i g_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(f + i + j));
                __m256i h_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(f + i + j + len / 2));
                __m256i w_vec = _mm256_set1_epi32(w[t]); 
                __m256i mul_result = Mul8(h_vec, w_vec);
                __m256i add_result = Add8(g_vec, mul_result);
                __m256i dec_result = Dec8(g_vec, mul_result);

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(f + i + j), add_result);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(f + i + j + len / 2), dec_result);
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
    u32 g=intToMont(G);
   
    u32 wn[MAXN];
    wn[0] = intToMont(1);

    u32 r = Pow(g, (p - 1) / n);
    __m256i r_vec = _mm256_set1_epi32(r);

    int t = 0;
    for (; (1 << (t + 8)) < n; t += 8) {
        int32_t n_arr[8] = {
            n >> (t + 2),
            n >> (t + 3),
            n >> (t + 4),
            n >> (t + 5),
            n >> (t + 6),
            n >> (t + 7),
            n >> (t + 8),
            n >> (t + 9)
        };
        __m256i n_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(n_arr));
        __m256i pow_vec = Pow8(r_vec, n_vec);

        for(int k=0;k<8;k++) wn[1<<(t+k)]=pow_vec[k];
    }
    for (; (1 << (t + 1)) < n; ++t) {
        wn[1 << t] = Pow(r, n >> (t + 2));
    }

    for (int t = 0; (1 << (t + 1)) < n; ++t) {
        u32 f = wn[1 << t];
        __m256i f_vec = _mm256_set1_epi32(f);

        int x = 1 << t;

        for (; x + 8 <= (1 << (t + 1)); x += 8) {
            __m256i wn_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(wn + x - (1 << t)));
            __m256i mul_vec = Mul8(f_vec, wn_vec);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(wn + x), mul_vec);
        }

        // 处理剩余不足 8 个元素的部分
        for (; x < (1 << (t + 1)); ++x) {
            wn[x] = Mul(f, wn[x - (1 << t)]);
        }
    }
     if(inv_flag==false) dif(a,n,p,wn);
     else dit(a,n,p,wn);
}


void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    m=p;
    inv = getinv();

    R2 = -u64(m) % m;
    u32 fa[MAXN] = {0}, fb[MAXN] = {0};
   
    for (int i = 0; i  <= n; i += 8) {
        __m256i a_vec = _mm256_loadu_si256((__m256i*)(a + i));
        __m256i b_vec = _mm256_loadu_si256((__m256i*)(b + i));

    
        __m256i fa_vec = intToMont8(a_vec);
        __m256i fb_vec = intToMont8(b_vec);
        _mm256_storeu_si256((__m256i*)(fa + i), fa_vec);
        _mm256_storeu_si256((__m256i*)(fb + i), fb_vec);
    }


       
    int k = 1;

    while (k < 2 * n) {
        k <<= 1;
    }
    
    ntt(fa, k, p, false);
    ntt(fb, k, p, false);

 
   
    for (int i=0; i  < k; i += 8) {
        __m256i va = _mm256_loadu_si256((__m256i*)(fa + i));
        __m256i vb = _mm256_loadu_si256((__m256i*)(fb + i));
        __m256i vc = Mul8(va, vb);
        _mm256_storeu_si256((__m256i*)(fa + i), vc);
    }

    ntt(fa, k, p, true);

    int i = 0;
    for (; i+8 <= 2 * n - 1; i+=8) {
        __m256i va = _mm256_loadu_si256((__m256i*)(fa + i));
        __m256i res= get8(va);
        _mm256_storeu_si256((__m256i*)(ab + i), res);
    }
    for(;i<2*n-1;i++) ab[i]=get(fa[i]);
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