#include <iostream>
#include "gen_gpu.h"
#include<cuda.h>
#include <vector>
#include <ctime>
int main(int argc,char*argv[]){
    std::string temp;
    int nvar=2,pob=10,reps=1,iters=10000;
    for(int i=1;i<argc;i++){
        if(i==1)pob=atoi(argv[i]);
        if(i==2)temp=argv[i];
        if(i==3)reps=atoi(argv[i]);
        if(i==4)nvar=atoi(argv[i]);
    }
    if(nvar<2)nvar=2;
    std::vector<float> tmp(2);
    tmp[0]=-10;tmp[1]=10;
    mat_f lims;
        lims=mat_f(nvar,tmp);
            printf("Se realizaran %d itereaciones con %d pobladores de %d variables\n ",iters,pob,nvar);
            double ftime=0;
            gpu_GA GA(pob,nvar,0.95,0.1,lims);
            for(int i=0;i<reps;i++){
                auto tic=clock();
                GA.generate_population();
                GA.optimiza(iters,0.95);
                ftime+=clock()-tic;
            }
            std::string nom="Resultados.txt";
            GA.print(nom);
            ftime/=reps;
            printf("se hicieron %d repeticiones\n",reps);
            printf("El tiempo de ejecucion promedio para realizar una ejecucion fue de: %e[s]\n",ftime/CLOCKS_PER_SEC);
            cudaDeviceReset();
    return 0;
}