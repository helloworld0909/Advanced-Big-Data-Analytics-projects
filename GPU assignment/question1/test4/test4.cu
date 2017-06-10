#include <stdio.h>

using namespace std;

void test4(){
    int **a = (int **) malloc(3 * sizeof (int *));

    for(int i=0;i<3;i++){
        a[i] = (int *) malloc(100 * sizeof(int));
    }
    a[1][1] = 5;
}

int main(){
    test4();
}