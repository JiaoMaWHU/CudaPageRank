#include <iostream>
#include "pagerank.h"

int main(){
    for(int i=0; i<5; i++)
        run_pagerank("1000.txt", 1e-4);
    return 0;
}