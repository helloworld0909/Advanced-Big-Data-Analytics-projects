#include <stdio.h>

using namespace std;

void test5(){
    int *a = (int *) malloc(sizeof (int));
    scanf("%d", a);
    if (!*a)    //!*a not !a
        printf("Value is 0\n");
}

int main(){
    test5();
}