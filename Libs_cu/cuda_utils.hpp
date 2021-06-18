/*Esta cabecera se utilizo para declarar Clases  que busca hacer que el manejo de las variables
dentro del gpu resulte mas intuitiva, incluye algunas partes de deteccion de errores de asginacion

Las variables declaradas tienen 3 objetivos distintos:
    1) Generar arreglos unidimensionales dentro del gpu, facilitan el manejo de memoria desde el host, asi como 
    las copias entre CPU<=>GPU
    2) Una clase que se hace cargo de la generacion de numeros aleatorios utilizando la libreria curand
    Se genero de esta forma ya que es necesario mantener espacios de memoria especificos para este fin
    asi como mantener el estado del generador en una varuable especial, de esta forma se tiene autocontenido
    tanto el estado del generador aleatorio como el arreglo que le corresponde.
    3) Una clase de tipo matriz que crea y administra un arreglo bidimensional dentro del gpu.
    Se encarga de administrar las direcciones de memoria, asi como de las copias GPU<=>CPU
Elaboro: Eduardo Tapia Romero*/
#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>
#include <curand.h>
#include <vector>
#include <stdio.h>
#define GLOBAL __global__
#define DEVICE __device__
#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    printf("The error is: %s\n",cudaGetErrorString(x));\
    exit(EXIT_FAILURE);}} while(0)
#define CUDA__ERROR_CHK { auto x=cudaGetLastError(); if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    printf("The error is: %s\n",cudaGetErrorString(x));\
    exit(EXIT_FAILURE);}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)
typedef std::vector<float> vec_f;
typedef std::vector<vec_f> mat_f;
template<class T>
class cuda_vec{
    private:
        T *ptr;
        unsigned int size,type_s,bitsize,capacity;
    public:
    cuda_vec();
    cuda_vec(int n);
    ~cuda_vec();
    T* data();
    void copy_from(T *ptr_h,int size,int type_size);
    void resize(int new_size);
    void copy_to(T *ptr_h);
    void to_host(std::vector<T> &a);
    void show();
};
template class cuda_vec<int>;
template class cuda_vec<float>;
template class cuda_vec<double>;   
template class cuda_vec<char>;





template<typename T>
class cudamat{
    public:
        T *vec_d,**mat_d;
        unsigned int rows,cols,type_size,capacity,row_c;
        cudamat();
        cudamat(int r,int c);
        ~cudamat();
        void to_device(const std::vector<T>&a);
        void to_device(const std::vector<std::vector<T>>&a);
        void resize(int n_r,int n_c);
        void to_host(std::vector<T>&a);
        T** get_mat();
        void show();
        void resume();
};

template class cudamat<int>;
template class cudamat<float>;
template class cudamat<double>;
template class cudamat<char>;


class gpu_randgen{
    private:
        int maxs;
        cuda_vec<float> dats;
        curandGenerator_t gen;
    public:
        gpu_randgen();
        gpu_randgen(int max_size);
        ~gpu_randgen();
        void set(int i);
        void generate();
        void to_host(vec_f &datos);
        float* get_dat();
        void show();
};


void swap(cudamat<float> &a,cudamat<float>&b);

void prueba(int a);
//Kernels
template<typename T>
GLOBAL void matrixtest(T **mat,int idim,int jdim);
template<typename T>
GLOBAL void matrixset(T **mat,T* vec_d,int rows, int cols );
template<typename T>
GLOBAL void vectortest(T *vec, int idim);
GLOBAL void set_val(float* val,float x);
#endif