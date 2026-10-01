#include <stdio.h>
#include <string.h>
#include <windows.h>

unsigned long long fnv1a(const char *s) {
    unsigned long long h = 1469598103934665603ULL;
    while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211ULL; }
    return h;
}

int partition(int *a, int lo, int hi) {
    int piv = a[hi], i = lo;
    for (int j = lo; j < hi; j++) {
        if (a[j] < piv) { int t = a[i]; a[i] = a[j]; a[j] = t; i++; }
    }
    int t = a[i]; a[i] = a[hi]; a[hi] = t;
    return i;
}
void quicksort(int *a, int lo, int hi) {
    if (lo < hi) {
        int p = partition(a, lo, hi);
        quicksort(a, lo, p - 1);
        quicksort(a, p + 1, hi);
    }
}

#define N 8
void matmul(int A[N][N], int B[N][N], int C[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            int s = 0;
            for (int k = 0; k < N; k++) s += A[i][k] * B[k][j];
            C[i][j] = s;
        }
}

int add_one(int x) { return x + 1; }

int main() {
    ULONGLONG t0 = GetTickCount64();

    int a[16] = {9,3,14,1,7,0,12,5,11,2,15,4,8,6,13,10};
    quicksort(a, 0, 15);
    int sorted_ok = 1;
    for (int i = 0; i < 16; i++) if (a[i] != i) sorted_ok = 0;

    const char *words[] = {"alpha","bravo","charlie","delta","echo"};
    unsigned long long h = 0;
    for (int i = 0; i < 5; i++) h ^= fnv1a(words[i]) + i * 0x9e3779b97f4a7c15ULL;

    int A[N][N], B[N][N], C[N][N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) { A[i][j] = i + j; B[i][j] = i - j + 3; C[i][j] = 0; }
    matmul(A, B, C);
    int csum = 0;
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) csum += C[i][j];

    char buf[64];
    strcpy_s(buf, sizeof(buf), "iso-8859-5-test");
    size_t L = strlen(buf);
    int counter = 0;
    for (int i = 0; i < 100; i++) {
        counter = add_one(counter);
        if ((i % 7) == 0) counter += (int)(h & 1);
        else if ((i % 5) == 0) counter -= 0;
    }

    ULONGLONG t1 = GetTickCount64();
    printf("sorted=%d h=%llu csum=%d counter=%d len=%zu ms=%llu\n",
        sorted_ok, h, csum, counter, L, t1 - t0);
    return sorted_ok ? 0 : 1;
}
