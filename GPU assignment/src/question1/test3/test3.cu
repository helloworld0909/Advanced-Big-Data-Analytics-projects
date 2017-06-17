#include <stdio.h>

using namespace std;

void test3(){
    int i, *a = (int *) malloc(1000 * sizeof(int));     //malloc(length * sizeof(type))
    if (!a){
        printf("Out of memory\n");
        exit(-1);
    }
    for (i = 0; i < 1000; i++)
        *(i + a) = i;
}

int main(){
    test3();
}