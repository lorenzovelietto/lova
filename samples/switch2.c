#include <stdio.h>

int classify(int x) {
    switch (x) {
    case 0:  case 1:  return 10;
    case 2:           return 22;
    case 3:  case 4:  case 5:  return 5;
    case 6:           return 77;
    case 7:  case 8:  return 1;
    case 9:           return 90;
    case 10: case 11: return 33;
    default: return -1;
    }
}

int main() {
    int s = 0;
    for (int i = 0; i < 12; i++) s += classify(i);
    printf("s=%d\n", s);
    return 0;
}
