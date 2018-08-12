#include <iostream>
#include <vector>
#include <unordered_set>
#undef NDEBUG // Do assert always
#include <assert.h>
#include <classifiers/fisheryatesshuffle.h>

int main(){

    FisherYatesShuffle shuffler(50000);

    std::vector<int> res;
    std::unordered_set<int> test_set;

    for(int i=0; i<5; i++){
        res = shuffler.get_random_indexies(500);

        for(auto a: res){
            assert( test_set.find(a) == test_set.end() );
            test_set.insert(a);
        }
    }
    std::cout << "Test passed! " << std::endl;
    return 0;
}
