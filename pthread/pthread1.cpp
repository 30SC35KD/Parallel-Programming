#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include<algorithm>
#include <iomanip>
#include <omp.h>
#include<pthread.h>
// 可以自行添加需要的头文件
using namespace std;

typedef long long LL;
const int NUM_THREADS = 4; // 线程数可调
const int MAXN = 300000;
int G = 3;
int* shared_a;
int n_global, p_global, inv_global;
int wn[MAXN];
pthread_barrier_t barrier;

typedef struct {
    int t_id;
} threadParam_t;

int quick_mi(int a, int b, int p) {
    long long res = 1;
    while (b) {
        if (b & 1) res = res * a % p;
        a = (long long)a * a % p;
        b >>= 1;
    }
    return res;
}

// 多线程执行函数
void* threadNTT(void* param) {
    threadParam_t* p = (threadParam_t*)param;
    int t_id = p->t_id;
    int n = n_global, p_mod = p_global, inv = inv_global;
    int* a = shared_a;

    for (int len = 2; len <= n; len <<= 1) {
        int total_blocks = n / len;
        int blocks_per_thread = (total_blocks + NUM_THREADS - 1) / NUM_THREADS;
        int start_block = t_id * blocks_per_thread;
        int end_block = std::min(start_block + blocks_per_thread, total_blocks);

        for (int blk = start_block; blk < end_block; ++blk) {
            int i = blk * len;
            for (int j = 0; j < len / 2; ++j) {
                int u = a[i + j];
                int v = 1LL * wn[j * (n / len)] * a[i + j + len / 2] % p_mod;
                a[i + j] = (u + v) % p_mod;
                a[i + j + len / 2] = (u - v + p_mod) % p_mod;
            }
        }

        // 同步所有线程
        pthread_barrier_wait(&barrier);
    }

    pthread_exit(NULL);
    return nullptr;

}

// 主 NTT 函数
void ntt(int* a, int n, int p, int inv) {
    shared_a = a;
    n_global = n;
    p_global = p;
    inv_global = inv;

    // Bit-reversal 置换（单线程）
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    // 预处理 wn（单线程）
    for (int len = 2; len <= n; len <<= 1) {
        int root = quick_mi(G, (p - 1) / len, p);
        if (inv) root = quick_mi(root, p - 2, p);
        int w = 1;
        for (int j = 0; j < len / 2; ++j) {
            wn[j * (n / len)] = w;
            w = 1LL * w * root % p;
        }
    }

    // 初始化 barrier
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);

    // 创建线程
    pthread_t threads[NUM_THREADS];
    threadParam_t params[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; ++i) {
        params[i].t_id = i;
        pthread_create(&threads[i], NULL, threadNTT, (void*)&params[i]);
    }

    // 等待所有线程结束
    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], NULL);
    }

    pthread_barrier_destroy(&barrier);

    // 归一化（单线程）
    if (inv) {
        int inv_n = quick_mi(n, p - 2, p);
        for (int i = 0; i < n; ++i) {
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
int a[MAXN], b[MAXN], ab[MAXN];
int main()
{
    int test_begin = 0;
    int test_end = 0;
    for (int i = test_begin; i <= test_end; ++i) {

        int n_ = 16, p_ = 7340033;
        for (int j = 0; j < n_; ++j) {
            a[j] = 1;
            b[j] = 1;
        }

        poly_multiply(a, b, ab, n_, p_);
        for (int j = 0; j < 2 * n_ - 1; ++j) {
            cout << ab[j] << " ";
        }

    }
    return 0;
}
