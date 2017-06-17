#include <stdio.h>

using namespace std;

void test1(){
    int tmp = 3;
    int *a = (int*)&tmp;
    *a = *a + 2;
    printf("%d\n", *a);
}

int main(){
    test1();
    return 0;
}