#include <iostream>
#include <vector>
#include "fisheryatesshuffle.h"

int main(){

    FisherYatesShuffle shuffler(10);

    std::vector<int> res;

    for(int i=0; i<10; i++){
        res = shuffler.get_random_indexies(3);

        std::cout << "Rolling " << i << " : ";
        for(auto a: res){
            std::cout << a << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
