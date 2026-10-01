#include <stdio.h>
int add_one(int x) { return x + 1; }
int main() {
    int a = 0;
    a = add_one(a);
    int b = 5;
    b = b + 1;
    if (a == 1) printf("a=1 b=%d\n", b);
    return 0;
}
