#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
#include <bits/stdc++.h>
#include<arm_neon.h>
void fRead(int *a, int *b, int *n, int *p, int input_id) {
    // 数据输入函数
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    std::copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    fin >> *n >> *p;
    for (int i = 0; i < *n; i++) {
        fin >> a[i];
    }
    for (int i = 0; i < *n; i++) {
        fin >> b[i];
    }
}
void fCheck(int *ab, int n, int input_id) {
    // 判断多项式乘法结果是否正确
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    for (int i = 0; i < n * 2 - 1; i++) {
        int x;
        fin >> x;
        if (x!= ab[i]) {
            std::cout << "多项式乘法结果错误" << std::endl;
            return;
        }
    }
    std::cout << "多项式乘法结果正确" << std::endl;
    return;
}
void fWrite(int *ab, int n, int input_id) {
    // 数据输出函数, 可以用来输出最终结果, 也可用于调试时输出中间数组
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    for (int i = 0; i < n * 2 - 1; i++) {
        fout << ab[i] << '\n';
    }
}
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
    uint64x2_t wide = vmull_u32(a, b);         // 低位乘法 → 64位结果
    return vshrn_n_u64(wide, 32);              // >> 32, 截取高32位
}

inline uint32x4_t reduce4(uint64x2_t x64_1, uint64x2_t x64_2) {
    // 常量向量
    uint32x2_t vinv = vdup_n_u32(inv);
    uint32x2_t vm = vdup_n_u32(m);

    // 第一组数据规约
    // x_lo = x & 0xFFFFFFFF, 取低32位
    uint32x2_t x_lo_1 = vmovn_u64(x64_1);  // 截断64位 → 低位32
    // x_hi = x >> 32
    uint32x2_t x_hi_1 = vshrn_n_u64(x64_1, 32);
    // t = ((u32(x) * inv) * m) >> 32
    uint32x2_t t1_1 = vmul_u32(x_lo_1, vinv);             // x_lo * inv
    uint32x2_t t2_1 = mulHi_u32(t1_1, vm);                // 高32位部分
    // y = x_hi - t2
    int32x2_t y_1 = vreinterpret_s32_u32(vsub_u32(x_hi_1, t2_1));
    // 如果 y < 0，加 m
    uint32x2_t y_unsigned_1 = vreinterpret_u32_s32(y_1);
    uint32x2_t mask_1 = vclt_s32(y_1, vdup_n_s32(0));     // mask: 0xFFFFFFFF if y < 0
    uint32x2_t y_final_1 = vbsl_u32(mask_1, vadd_u32(y_unsigned_1, vm), y_unsigned_1);

    // 第二组数据规约
    // x_lo = x & 0xFFFFFFFF, 取低32位
    uint32x2_t x_lo_2 = vmovn_u64(x64_2);  // 截断64位 → 低位32
    // x_hi = x >> 32
    uint32x2_t x_hi_2 = vshrn_n_u64(x64_2, 32);
    // t = ((u32(x) * inv) * m) >> 32
    uint32x2_t t1_2 = vmul_u32(x_lo_2, vinv);             // x_lo * inv
    uint32x2_t t2_2 = mulHi_u32(t1_2, vm);                // 高32位部分
    // y = x_hi - t2
    int32x2_t y_2 = vreinterpret_s32_u32(vsub_u32(x_hi_2, t2_2));
    // 如果 y < 0，加 m
    uint32x2_t y_unsigned_2 = vreinterpret_u32_s32(y_2);
    uint32x2_t mask_2 = vclt_s32(y_2, vdup_n_s32(0));     // mask: 0xFFFFFFFF if y < 0
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

    // 调用我们前面写的 reduce_vec
    uint32x4_t r = reduce4(x0, x1);
    // 合并为 4 路结果
    return r;
}

inline uint32x4_t Add4(uint32x4_t a, uint32x4_t b) {
    // 计算 a + b
    uint32x4_t sum = vaddq_u32(a, b);
    
    // 生成掩码: 如果 sum[i] >= m, 对应 mask[i] = 1
    uint32x4_t mask = vcgeq_u32(sum, vdupq_n_u32(m));
    
    // 需要减去 m 的位置: mask[i] 为 1 时，减去 m
    uint32x4_t to_subtract = vandq_u32(mask, vdupq_n_u32(m));
    
    // 执行减法：sum[i] - m（如果 sum[i] >= m），否则不变
    return vsubq_u32(sum, to_subtract);
}

inline uint32x4_t Dec4(uint32x4_t a, uint32x4_t b) {
    // 计算 a - b
    uint32x4_t diff = vsubq_u32(a, b);
    
    // 生成掩码: 如果 a[i] < b[i], 对应 mask[i] = 1
    uint32x4_t mask = vcgtq_u32(b, a);
    
    // 需要加上 m 的位置: mask[i] 为 1 时，差值加上 m
    uint32x4_t to_add = vandq_u32(mask, vdupq_n_u32(m));
    
    // 执行加法：diff[i] + m（如果 a[i] < b[i]），否则不变
    return vaddq_u32(diff, to_add);
}

inline uint32x4_t Mul4(uint32x4_t a, uint32x4_t b) {
    // 将 32 位向量 a 和 b 分别拆成两个 64 位部分进行逐项乘法
    uint64x2_t x0 = vmull_u32(vget_low_u32(a), vget_low_u32(b));  // 低 32 位乘法
    uint64x2_t x1 = vmull_u32(vget_high_u32(a), vget_high_u32(b));  // 高 32 位乘法

    // 使用 reduce2 进行蒙哥马利规约
    uint32x4_t reduced = reduce4(x0,x1);
   
    // 将 reduced 转回一个 32 位向量，做进一步操作（如果需要）
    return reduced;  // 返回一个包含相同元素的向量（你可以根据需要更改这里）
}
uint32x4_t Pow4(uint32x4_t base, uint32x4_t exponent) {
    uint32x4_t result = vdupq_n_u32(intToMont(1));
    uint32x4_t mont_base = base;

    while (true) {
        // 生成掩码，判断每个 exponent 的最低位是否为 1
        uint32x4_t mask = vandq_u32(exponent, vdupq_n_u32(1));

        // 对于掩码中非零的部分，执行 result = Mul(result, mont_base)
        uint32x4_t mul_result = Mul4(result, mont_base);  // 你需要定义 Mul4
        result = vbslq_u32(vceqq_u32(mask, vdupq_n_u32(1)), mul_result, result);

        // mont_base = mont_base * mont_base
        mont_base = Mul4(mont_base, mont_base);

        // exponent >>= 1
        exponent = vshrq_n_u32(exponent, 1);

        // 判断是否全部 exponent 都变成 0
        uint64x2_t done_mask = vreinterpretq_u64_u32(vceqq_u32(exponent, vdupq_n_u32(0)));
        if (vgetq_lane_u64(done_mask, 0) == ~0ULL && vgetq_lane_u64(done_mask, 1) == ~0ULL) {
            break;
        }
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
typedef long long LL;
const int MAXN = 300000;
// 定义原根 G
int G = 3;

void dit(u32 *f,int n,int p,u32 *w)
{
    u32 g,h;
    for (int len = 2; len <= n; len <<= 1){
        int half=len/2;
		for (int i = 0,t=0; i < n; i += len,++t){
            int j=0;
            uint32x4_t w_vec = vdupq_n_u32(w[t]);
			for (; j+4 <= half; j+=4){
                uint32x4_t g_vec = vld1q_u32(f+i+j);
                uint32x4_t h_vec = vld1q_u32(f+i+j+half);
                
                uint32x4_t add_vec = Add4(g_vec, h_vec);
                uint32x4_t dec_vec = Dec4(g_vec, h_vec);
                uint32x4_t mul_vec = Mul4(dec_vec, w_vec);
                vst1q_u32(f + i + j, add_vec);
                vst1q_u32(f + i + j + half, mul_vec);
            }
            for (; j < half; ++j){
                g = f[i+j], h = f[i+j + len/2];
                f[i+j] = Add(g,h);
                f[i+j + len/2] = Mul(Dec(g ,h), w[t]);
            }
        }
    }
	u32 invl = Pow (intToMont(n), p-2);
    uint32x4_t invl_vec = vdupq_n_u32(invl);
	for (int i = 0; i < n; i+=4)
        {uint32x4_t f_vec = vld1q_u32(f+i);
		uint32x4_t i_vec=Mul4(invl_vec,f_vec);
        vst1q_u32(f+i,i_vec);
        }
    // 反转数组
	std::reverse (f+1 , f + n);
}
void dif(u32 *f,int n,int p,u32*w)
{
    u32 g, h;
	for (int len = n; len > 1; len >>= 1){
        int half=len/2;
		for (int i = 0,t=0; i < n; i += len,t++){
            int j=0;
            uint32x4_t w_vec = vdupq_n_u32(w[t]);
			for (; j +4<=  half; j+=4)
				{
                uint32x4_t g_vec = vld1q_u32(f+i+j);
                uint32x4_t G_vec = vld1q_u32(f+i+j+half);
                
                uint32x4_t h_vec = Mul4(G_vec, w_vec);
                uint32x4_t sum_vec = Add4(g_vec, h_vec);
                uint32x4_t diff_vec = Dec4(g_vec, h_vec);
                vst1q_u32(f + i + j, sum_vec);
                vst1q_u32(f + i + j + half, diff_vec);
                }
                for(;j<half;j++)
                {g = f[i+j];h = Mul(f[i+j + half] ,w[t]);
                    f[i+j] = Add(g , h);
                    f[i +j+ half] = Dec( g ,h);
                }
            }
        }
}
        
// 优化后的快速数论变换（NTT）函数
void ntt(u32 *a, int n, int p, int inv_flag) {
    u32 g=intToMont(G);
   
     // 预计算单位根
    u32 wn[MAXN];
    wn[0] = intToMont(1); u32 f;
	u32 r = Pow (g, (p-1)/n);

    int t = 0;
    uint32x4_t r_vec = vdupq_n_u32(r);
	for (; (1<<(t+4)) < n; t+=4) {
        int32_t n_arr[4] = {
           n>>(t+2), 
           n>>(t+3), 
           n>>(t+4), 
           n>>(t+5),
        };
        int32x4_t n_vec = vld1q_s32(n_arr);
        uint32x4_t t_vec=vreinterpretq_u32_s32(n_vec);
       
		uint32x4_t pow_vec = Pow4 (r_vec, t_vec);
        wn[1<<t]=vgetq_lane_u32(pow_vec, 0);
        wn[1<<(t+1)]=vgetq_lane_u32(pow_vec, 1);
        wn[1<<(t+2)]=vgetq_lane_u32(pow_vec, 2);
        wn[1<<(t+3)]=vgetq_lane_u32(pow_vec, 3);
    }
    for(;(1<<(t+1)) < n; ++t) {
        wn[1<<t] = Pow(r,n>>(t+2));
    }

    for (int t = 0; (1<<(t+1)) < n; ++t) {
        f=wn[1<<t];
        uint32x4_t f_vec = vdupq_n_u32(f);
        int x = 1<<t;
		for (; x+4 <= 1<<(t+1); x+=4)
         {   uint32x4_t wn_vec = vld1q_u32(wn+x-(1<<t));
            uint32x4_t mul_vec = Mul4(f_vec, wn_vec);
            vst1q_u32(wn+x, mul_vec);
         }
         for (; x < 1<<(t+1); ++x)
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
    // 复制多项式 a 和 b 的系数到 fa 和 fb
    for (int i = 0; i + 4 <= n; i += 4) {
        int32x4_t va = vld1q_s32(a + i);  // 加载 a[i] ~ a[i+3]
        int32x4_t vb = vld1q_s32(b + i);  // 加载 b[i] ~ b[i+3]
    
        uint32x4_t fa_vec = intToMont4(va);
        uint32x4_t fb_vec = intToMont4(vb);
    
        vst1q_u32(fa + i, fa_vec);        // 存回 fa[i] ~ fa[i+3]
        vst1q_u32(fb + i, fb_vec);        // 存回 fb[i] ~ fb[i+3]
    }
    int k = 1;
    // 找到合适的长度 m ，满足 m 是 2 的幂且不小于 2 * n
    while (k < 2 * n) {
        k <<= 1;
    }
    
    // 对 fa 和 fb 分别进行 NTT 变换
    ntt(fa, k, p, false);
    ntt(fb, k, p, false);

   // 在频域上进行多项式乘法
   for (int i = 0; i+4 <= k; i+=4) {
    uint32x4_t mont_fa_i = vld1q_u32(fa + i);
    uint32x4_t mont_fb_i = vld1q_u32(fb + i);
    uint32x4_t fa_vec = Mul4(mont_fa_i, mont_fb_i);
    vst1q_u32(fa + i, fa_vec);
}
    // 对结果进行逆 NTT 变换
    ntt(fa, k, p, true);
    int i=0;
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
        total=0;
        reminder=0;
        auto Start = std::chrono::high_resolution_clock::now();
        // TODO : 将 poly_multiply 函数替换成你写的 ntt
        poly_multiply(a, b, ab, n_, p_);
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        ans += elapsed.count();
        std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (us) "<<" "<<reminder/total<<std::endl;
        // 可以使用 fWrite 函数将 ab 的输出结果打印到 files 文件夹下
        // 禁止使用 cout 一次性输出大量文件内容
    }
    return 0;
}