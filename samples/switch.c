#include <stdio.h>

int classify(int x) {
    switch (x) {
    case 0:  return 10;  case 1:  return 22;  case 2:  return 5;
    case 3:  return 77;  case 4:  return 1;   case 5:  return 90;
    case 6:  return 33;  case 7:  return 14;  case 8:  return 66;
    case 9:  return 28;  case 10: return 41;  case 11: return 7;
    default: return -1;
    }
}

int main() {
    int s = 0;
    for (int i = 0; i < 12; i++) s += classify(i) ^ classify((i * 5) % 13);
    printf("s=%d\n", s);
    return 0;
}
