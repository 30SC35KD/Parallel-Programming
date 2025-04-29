#include <iostream>
#include <cstring>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
#include<arm.neon.h>
#include <bits/stdc++.h>
#include <cstdlib>
#include <ctime>
typedef long long LL;
const int MAXN=300000;
int G=3;
// 类型定义
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

// 计算高 32 位的 (a * b) >> 32
inline uint32x2_t mulHi_u32(uint32x2_t a, uint32x2_t b) {
    uint64x2_t wide = vmull_u32(a, b);         // 低位乘法至64位结果
    return vshrn_n_u64(wide, 32);              // >> 32, 截取高32位
}

inline uint32x4_t reduce4(uint64x2_t x64_1, uint64x2_t x64_2) {
    // 常量向量
    uint32x2_t vinv = vdup_n_u32(inv);
    uint32x2_t vm = vdup_n_u32(m);

    // 第一组数据规约
    // x_lo 取低32位
    uint32x2_t x_lo_1 = vmovn_u64(x64_1);  // 截断64位 低位32
    // x_hi = x >> 32
    uint32x2_t x_hi_1 = vshrn_n_u64(x64_1, 32);
    // t = ((u32(x) * inv) * m) >> 32
    uint32x2_t t1_1 = vmul_u32(x_lo_1, vinv);             // x_lo * inv
    uint32x2_t t2_1 = mulHi_u32(t1_1, vm);                // 高32位部分
    // y = x_hi - t2
    int32x2_t y_1 = vreinterpret_s32_u32(vsub_u32(x_hi_1, t2_1));
    // 如果 y < 0，加 m
    uint32x2_t y_unsigned_1 = vreinterpret_u32_s32(y_1);
    uint32x2_t mask_1 = vclt_s32(y_1, vdup_n_s32(0));   
    uint32x2_t y_final_1 = vbsl_u32(mask_1, vadd_u32(y_unsigned_1, vm), y_unsigned_1);

    // 第二组数据规约
    // x_lo 取低32位
    uint32x2_t x_lo_2 = vmovn_u64(x64_2);  // 截断64位 低位32
    // x_hi = x >> 32
    uint32x2_t x_hi_2 = vshrn_n_u64(x64_2, 32);
    // t = ((u32(x) * inv) * m) >> 32
    uint32x2_t t1_2 = vmul_u32(x_lo_2, vinv);             // x_lo * inv
    uint32x2_t t2_2 = mulHi_u32(t1_2, vm);                // 高32位部分
    // y = x_hi - t2
    int32x2_t y_2 = vreinterpret_s32_u32(vsub_u32(x_hi_2, t2_2));
    // 如果 y < 0，加 m
    uint32x2_t y_unsigned_2 = vreinterpret_u32_s32(y_2);
    uint32x2_t mask_2 = vclt_s32(y_2, vdup_n_s32(0));   
    uint32x2_t y_final_2 = vbsl_u32(mask_2, vadd_u32(y_unsigned_2, vm), y_unsigned_2);

    // 合并两组结果
    return vcombine_u32(y_final_1, y_final_2);
}

inline uint32x4_t intToMont4(int32x4_t x) {
    // 将 x 拆成两段：低 2 个、 高 2 个
    int32x2_t x_low = vget_low_s32(x);
    int32x2_t x_high = vget_high_s32(x);

    // 转为 u64 并乘以 R2
    uint64x2_t x0 = vmull_u32(vreinterpret_u32_s32(x_low), vdup_n_u32(R2));
    uint64x2_t x1 = vmull_u32(vreinterpret_u32_s32(x_high), vdup_n_u32(R2));

    uint32x4_t r = reduce4(x0, x1);
    return r;
}

inline uint32x4_t Add4(uint32x4_t a, uint32x4_t b) {
    uint32x4_t sum = vaddq_u32(a, b);
    
    uint32x4_t mask = vcgeq_u32(sum, vdupq_n_u32(m));
    
    uint32x4_t to_subtract = vandq_u32(mask, vdupq_n_u32(m));
    
    return vsubq_u32(sum, to_subtract);
}

inline uint32x4_t Dec4(uint32x4_t a, uint32x4_t b) {
    uint32x4_t diff = vsubq_u32(a, b);
    
    uint32x4_t mask = vcgtq_u32(b, a);
    uint32x4_t to_add = vandq_u32(mask, vdupq_n_u32(m));

    return vaddq_u32(diff, to_add);
}

inline uint32x4_t Mul4(uint32x4_t a, uint32x4_t b) {
    // 将 32 位向量 a 和 b 分别拆成两个 64 位部分进行逐项乘法
    uint64x2_t x0 = vmull_u32(vget_low_u32(a), vget_low_u32(b));  // 低 32 位乘法
    uint64x2_t x1 = vmull_u32(vget_high_u32(a), vget_high_u32(b));  // 高 32 位乘法

    uint32x4_t reduced = reduce4(x0,x1);
   
    return reduced; 
}

inline uint32x4_t Pow4(uint32x4_t a, uint32x4_t b) {
    uint32x4_t result = intToMont4(vdupq_n_s32(1));
    uint32x4_t mont_base = a;

    for (int i = 31; i >= 0; i--) {
        uint32x4_t bit_mask = vshrq_n_u32(b, i);
        bit_mask = vandq_u32(bit_mask, vdupq_n_u32(1));

        result = Mul4(result, result);

        uint32x4_t temp = Mul4(result, mont_base);
        result = vbslq_u32(bit_mask, temp, result);
    }

    return result;
} 

int32x4_t get4(uint32x4_t x) {
    // 将 x 拆成两段：低 2 个、 高 2 个
    uint32x2_t x_low = vget_low_u32(x);
    uint32x2_t x_high = vget_high_u32(x);
    // 将 uint32x2_t 转换为 uint64x2_t
    uint64x2_t x64_low = vmovl_u32(x_low);
    uint64x2_t x64_high = vmovl_u32(x_high);
    uint32x4_t res=reduce4(x64_low, x64_high);
    int32x4_t result = vreinterpretq_s32_u32(res);
    // 合并为 4 路结果
    return result;
}
// 定义长整型别名

// 优化后的快速数论变换（NTT）函数
void ntt(u32 *a, int n, int p, int inv_flag) {
    u32 g=intToMont(G);
    // 初始化时进行一次位翻转
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

u32 wn[MAXN];
u32 rootCache[MAXN];

// 预计算所有的root
u32 len=2;
u32 idx=0;
for (; (len<<4) <= n; len <<=4,idx+=4 ) {
    uint32_t mi[4] = {
        (p-1)/len,
        (p-1)/(len<<1),
        (p-1)/(len<<2),
        (p-1)/(len<<3),
    };
    uint32x4_t mi_vec = vld1q_u32(mi);
    uint32x4_t g_vec = vdupq_n_u32(g);
    uint32x4_t root = Pow4(g_vec, mi_vec);
    if (inv_flag) {
        uint32x4_t mi = vdupq_n_u32(p-2);
        root = Pow4(root, mi);
    }
     // 临时数组用于存储 root 向量的值
    
     vst1q_u32(rootCache+idx, root);

}
for(;len<=n;len<<=1,idx++){
    uint32_t mi = (p-1)/len;
    uint32_t root = Pow(g, mi);
    if (inv_flag) {
        uint32_t mi = p-2;
        root = Pow(root, mi);
    }
    rootCache[idx] = root;
}
//预计算单位根
idx=0;
for (int len = 2; len <= n; len <<= 1,idx+=1) {
    u32 root = rootCache[idx];  // 使用预计算的root
    u32 w = intToMont(1);
    for (int j = 0; j < len / 2; ++j) {
        wn[j * (n / len)] = w;
        w = Mul(w, root);
    }
}
     // DIT（时域抽取）NTT 过程
     for (int len = 2; len <= n; len <<= 1) {
         int half = len / 2;
         int wn_stride = n / len;
         for (int i = 0; i < n; i += len) {
             int j=0;
             for (; j+4 <= half; j+=4) {
                uint32x4_t au = vld1q_u32(a + i + j);
                uint32x4_t av = vld1q_u32(a + i + j + half);

                uint32_t w_arr[4] = {
                    wn[(j + 0) * wn_stride],
                    wn[(j + 1) * wn_stride],
                    wn[(j + 2) * wn_stride],
                    wn[(j + 3) * wn_stride],
                };
                uint32x4_t w = vld1q_u32(w_arr);
            
                uint32x4_t avw = Mul4(w, av);
                uint32x4_t out0 = Add4(au, avw);
                uint32x4_t out1 = Dec4(au, avw);

                vst1q_u32(a + i + j, out0);
                vst1q_u32(a + i + j + half, out1);
             }
             for (; j < half; ++j) {
                u32 mont_u = a[i + j];
                u32 mont_v = Mul(wn[j * wn_stride], a[i + j + half]);
                a[i + j] = Add(mont_u, mont_v);
                a[i + j + half] = Dec(mont_u, mont_v);
            }
         }
     }

    if (inv_flag) {
        u32 inv_n = Pow(intToMont(n), p- 2);
        uint32x4_t mont_inv_n = vdupq_n_u32(inv_n);
        for (int i = 0; i+4 <= n; i+=4) {

            uint32x4_t a_i = vld1q_u32(a + i);
            a_i = Mul4(a_i, mont_inv_n);
            vst1q_u32(a + i, a_i);
            
        }
    }
}


void poly_multiply(int *a, int *b, int *ab, int n, int p) {
    m=p;
    inv = getinv();
    
    R2 = -u64(m) % m;
    u32 fa[MAXN] = {0}, fb[MAXN] = {0};
   
    for (int i = 0; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(a + i); 
        int32x4_t vb = vld1q_s32(b + i);  
    
        uint32x4_t fa_vec = intToMont4(va);
        uint32x4_t fb_vec = intToMont4(vb);
    
        vst1q_u32(fa + i, fa_vec);    
        vst1q_u32(fb + i, fb_vec);        
    }
    
    int k = 1;
    while (k < 2 * n) {
        k <<= 1;
    }
    ntt(fa, k, p, false);
    ntt(fb, k, p, false);

    for (int i = 0; i+4 <= k; i+=4) {
        uint32x4_t mont_fa_i = vld1q_u32(fa + i);
        uint32x4_t mont_fb_i = vld1q_u32(fb + i);
        uint32x4_t fa_vec = Mul4(mont_fa_i, mont_fb_i);
        vst1q_u32(fa + i, fa_vec);
    }
    ntt(fa, k, p, true);

    int i = 0;
    for (; i+4 <= 2 * n - 1; i+=4) {
        uint32x4_t fa_i = vld1q_u32(fa + i);
        int32x4_t ab_i = get4(fa_i);
        vst1q_s32(ab + i, ab_i);
    }
    for(;i<2*n-1;i++){
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