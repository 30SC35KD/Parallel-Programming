#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include<algorithm>
#include <iomanip>
#include <omp.h>

using namespace std;

typedef long long LL;
const int MAXN = 30000;
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
void ntt(int* a, int n, int p, int inv)
{

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)j ^= bit;
        j ^= bit;
        if (i < j) swap(a[i], a[j]);
    }

    int wn[MAXN];
    for (int len = 2; len <= n; len <<= 1) {
        int root = quick_mi(G, (p - 1) / len, p);
        if (inv) root = quick_mi(root, p - 2, p);
        int w = 1;
        for (int j = 0; j < len / 2; ++j) {
            wn[j * (n / len)] = w;
            w = 1LL * w * root % p;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        //#pragma omp parallel for
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < len / 2; ++j) {
                int u = a[i + j], v = 1LL * wn[j * (n / len)] * a[i + j + len / 2] % p;
                a[i + j] = (u + v) % p;
                a[i + j + len / 2] = (u - v + p) % p;
            }
        }
    }


    if (inv) {
        int inv_n = quick_mi(n, p - 2, p);
        for (int i = 0; i < n; ++i) a[i] = 1LL * a[i] * inv_n % p;
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

        int n_ = 256, p_ = 7340033;
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
