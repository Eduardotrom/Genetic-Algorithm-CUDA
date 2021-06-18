#include "auxx.hpp"

double mean(std::vector<float> &vec){
    double sum=0;
    for(auto x:vec)
        sum+=x;
    return sum/vec.size();
}