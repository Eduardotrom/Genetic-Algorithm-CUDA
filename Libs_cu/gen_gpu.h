/*
Esta cabecera define la clase de algoritmo genetico para gpu, y la clase individuo para gpu.
La clase de algoritmo genetico almacena todos los parametros a utilizar dentro de la ejecucion, y las envia al gpu.
Maneja la memoria y realiza las llamadas a los kernels segun sea necesario.

La clase De individuo gpu mantiene copias de la informacion de la mejor solucion que se ha encontrado.

*/

#ifndef GEN_GPU_H
#define GEN_GPU_H

#include "gen_kernels.h"
#include "cuda_utils.hpp"
#include <cmath>
#include <fstream>
#include "auxx.hpp"
class best_ind_gpu{
    public:
        cuda_vec<float> gens,data;
        best_ind_gpu(){};
        best_ind_gpu(int dims);
        void resize(int dims);
        float* get_fit();
        float* get_gen();
        float* get_index();
};


class gpu_GA{
    private:
        cudamat<float> parents,children,limits;
        cuda_vec<float> fit,conv;
        std::vector<float> max_m,min_m,hist_m;
        mat_f pop_host;
        gpu_randgen fgen;
        float eta,cp;
        int popSize,varNum;
        dim3 blk,tpb;
        best_ind_gpu best;
    public:
        gpu_GA(){};
        gpu_GA(int populationSize,int varNum,float cp,float eta,mat_f lims);
        void optimiza(int iters,float alpha);
        void generate_population();
        void get_population(mat_f &res);
        void print(std::string nom);
};  




#endif