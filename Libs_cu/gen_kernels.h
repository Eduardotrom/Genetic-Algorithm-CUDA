/*
Encabezado que define las llamadas para todos lso kerneles necesarios en 
el proceso de optimizacion.

Elaboro: Eduardo Tapia Romero.

*/
#ifndef GEN_KERNELS_HPP
#define GEN_KERNELS_HPP
#include<cuda.h>
#include<cuda_runtime.h>
#include "cuda_utils.hpp"
#define GLOBAL __global__
#define DEVICE __device__
#define IDXMC int idx=blockIdx.x*blockDim.x+threadIdx.x;
#define IDYMC int idy=blockIdx.y*blockDim.y+threadIdx.y;
#define BIDX blockIdx.x
#define BIDY blockIdx.y
GLOBAL void population_init(float **population,float *randnum,float **lims,int cols,int rows);
GLOBAL void mutation(float **population,float* randnum,float** lims,int cols, int rows,float CP);
GLOBAL void crossover(float **parents, float **children,float* chosen,float alpha,float * randnum, float**lims,int cols, int rows);
DEVICE void mutation_d(float* hijo,float randn,float upper,float lower,int gen);
GLOBAL void selection(float *randidx,float* chosen,float *fitness,int pob_size);
DEVICE float get_delt(float u );
void GLOBAL Rosembrock_gpu(float** pob,float *results,int cols,int rows);
void GLOBAL shpere(float** pob,float *results,int cols,int rows);
void GLOBAL findbest(float** population,float *fit,float *index,float *bestfit,float*bestgen,int popsize,int varnum);
GLOBAL void hist_up(float * fit,float * conv,int index);
//GLOBAL void statistics
//GLOBAL void selection
//GLOBAL void crossOver
#endif