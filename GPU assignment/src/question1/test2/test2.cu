#include <stdio.h>

using namespace std;

void test2(){
    int *a, *b;     // int *a, *b not int *a, b
    a = (int *) malloc(sizeof (int));
    b = (int *) malloc(sizeof (int));
    if (!(a && b)){
        printf("Out of memory\n");
        exit(-1);
    }
    *a = 2;
    *b = 3;
}

int main(){
    test2();
    return 0;
}