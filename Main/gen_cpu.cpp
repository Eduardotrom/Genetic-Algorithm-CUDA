#include "genetic_cpu.hpp"
#include <iostream>
#include <ctime>

int main(){
    srand(time(NULL));
    int pop=256,vars=2,iters=20;
    vec_f temp(2);    
    temp[0]=-10;temp[1]=10;
    //for(pop=128;pop<=8192;pop*=2)
        //for(vars=2;vars<18;vars+=2)
        {
            mat_f lims(vars,temp);
            Genetic_cpu GA(pop,vars,0.95,0.7,0.1,0.2,lims);
            auto t1=clock();
            t1=0;
            for(int i=0;i<iters;i++){
                auto t2=clock();
                GA.pop_init();
                GA.optimiza(50);
                t1+=clock()-t2;
            }
            GA.print("geneticos stats");
            printf("%4d %4d %e\n",pop,vars,(double)t1/(CLOCKS_PER_SEC*iters));
        }
    return 0;
}