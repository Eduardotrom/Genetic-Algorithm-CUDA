#include"cuda_utils.hpp"
    //constructor de la clase cuda_vec, si no se indica nada, incializa en 0 y apunta a null
    template<typename T>
    cuda_vec<T>::cuda_vec(){
        this->size=0;
        this->type_s=sizeof(T);
        this->ptr=NULL;
        this->bitsize=0;
        this->capacity=0;
        #ifdef DEBUG
            printf("Se alojo un vector en gpu para %d elementos de tamanio %d\n",this->size,this->type_s);
            printf("Su direccion es %p\n",this->ptr);
        #endif
    }
    //Constructor para clase cuda_vec con parametros, asigna la memoria del tamano que se indique.
    template<typename T>
    cuda_vec<T>::cuda_vec(int n){
        this->size=n;
        this->type_s=sizeof(T);
        this->bitsize=this->size*this->type_s;
        this->ptr=NULL;
        #ifdef __NVCC__
        CUDA_CALL(cudaMalloc((void**) &this->ptr,this->bitsize));
        auto err=cudaGetLastError();
        if(err!=cudaSuccess){//en caso de que la asignacion falle, se envia mensaje de error y se cierra la ejecucion
            printf("Could-nt allocate memory on GPU\n");
            printf("Cuda error %s",cudaGetErrorString(err));
            exit(2);
        }
        #ifdef DEBUG
            printf("Se alojo un vector en gpu para %d elementos de tamanio %d\n",this->size,this->type_s);
            printf("Su direccion es %p\n",this->ptr);
        #endif
        #endif
        }
    //Destructor de clase, desaloja la memoria alocada en caso de haber sido inicializada en primer lugar.
    template<typename T>
    cuda_vec<T>::~cuda_vec(){
        this->size=0;
        this->type_s=0;
        #ifdef __NVCC__
        if(this->ptr!=NULL){
            #ifdef DEBUG
                printf("Me voy a morir!, mi direccion es %p, mi tamano es de %d bytes\n",this->data(),this->bitsize);
            #endif
            cudaFree(this->ptr);
        }
        #endif
        this->bitsize=0;
        this->capacity=0;
        this->ptr=NULL;
    }
    //metodo de acceso al puntero de memoria de la clase
    template<typename T>
    T* cuda_vec<T>::data(){
        return this->ptr;
    }
    //Metodo de copia de cpu a host, recibe el puntero dentro de host
    //el numero de elementos, y el tama;o del tipo de variable a almacenar
    template<typename T>
    void cuda_vec<T>::copy_from(T *ptr_h,int size,int type_size){
        //quiebra la ejecucion si se intenta copiar dentro de un puntero vacio
        if(this->ptr==NULL){
            printf("Can-t copy to Device, Memory Not allocated \n");
            exit(2);
        }
        //quierba la ejecucion si se intenta asignar algo con capacidad distinta a la previamente asignada
        if(this->size!=size&&this->type_s!=type_size){
            printf("Non compatible cuda assignation, check memory sizes\n");
            exit(4);
        }
        #ifdef __NVCC__
        //se realiza la copia al device
        cudaMemcpy(this->ptr,ptr_h,size*type_size,cudaMemcpyHostToDevice);
        auto err=cudaGetLastError();
        if(err!=cudaSuccess){//si hay algun error en la copia, se detiene la ejecucion y se envia mensaje
            printf("Could'nt copy memory to device\n");
            printf("Cuda error %s\n",cudaGetErrorString(err));
            exit(2);
        }
        cudaDeviceSynchronize();
        #endif
    }
    //Metodo para cambiar de tamano el vector, se asegura de que el nuevo vector contenga la informacion
    //que contenia previamente y le asigna mas espacio, en caso de buscar reducir el tama;o, no libera memoria en gpu
    template<typename T>
    void cuda_vec<T>::resize(int max_size){
        #ifdef DEBUG
            printf("Entre al resize\n");
        #endif
        if(this->capacity<max_size){
            void*temp_d=NULL;
            this->bitsize=max_size*sizeof(T);
            CUDA_CALL(cudaMalloc(&temp_d,max_size*this->type_s));
            CUDA_CALL(cudaMemcpy(temp_d,(void*)this->ptr,this->size*this->type_s,cudaMemcpyDeviceToDevice));
            CUDA_CALL(cudaFree(this->ptr));
            #ifdef DEBUG
                printf("Se va liberar la direccion %p y la nueva direccion va a ser %p\n",this->ptr,temp_d);
                if(temp_d==NULL)exit(0);
            #endif
            this->ptr=(T*)temp_d;
            this->capacity=max_size;
        }
        this->size=max_size;
        cudaDeviceSynchronize();
    }
//funcion de copia de regreso a host, recibe el puntero en memoria host donde se quiere almacenar el resutlado
    template<typename T>
    void cuda_vec<T>::copy_to(T *ptr_h){
        #ifdef DEBUG
            printf("Se va a copiar un vector a host de %d individuos de %d bytes cada uno\n",this->size,this->type_s);
            printf("Las direcciones origen %p destino %p\n",ptr_h,this->ptr);
        #endif
        #ifdef __NVCC__
        CUDA_CALL(cudaMemcpy((void*)ptr_h,(void*)this->ptr,this->bitsize,cudaMemcpyDeviceToHost));
        /*auto err=cudaGetLastError();
        if(err!=cudaSuccess){
            printf("Could'nt copy memory back to host\n");
            printf("Cuda error %s",cudaGetErrorString(err));
            exit(2);
        }*/
        #endif
    }
    //Metodo de copia de datos al host directo a un vector, se hace cargo completamente del tamano de memoria
    //del vector dentro del host, no es necesario que la variable este inicializada previamente.
    template<typename T>
    void cuda_vec<T>::to_host(std::vector<T> &a){
        a.resize(this->size);
        this->copy_to(a.data());
    }

    //funcion que muestra en pantalla el contenido del vector en gpu
    template<typename T>
    void cuda_vec<T>::show(){
        vectortest<T><<<1,1>>>(this->data(),this->size);
        cudaDeviceSynchronize();
    }
//---------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------
//Constructor por defecto para el generador aleatorio en gpu uniforme 0-1
    gpu_randgen::gpu_randgen(){
        #ifdef __NVCC__
        CURAND_CALL(curandCreateGenerator(&this->gen,CURAND_RNG_PSEUDO_DEFAULT));
        int sd=time(NULL);
        #ifndef DEBUG
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(this->gen,sd));
        #else
        sd=1;
        printf("La semilla aleatoria es %d\n",sd);
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(this->gen,sd));
        cudaDeviceSynchronize();
        this->maxs=0;
        #endif
        #endif
    }

//---------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------
//Seccion para la matriz en cuda:
//constructor por defecto de la matriz
template<typename T>
cudamat<T>::cudamat(){
    this->rows=0;
    this->cols=0;
    this->type_size=sizeof(T);
    this->capacity=0;
    this->mat_d=NULL;
    this->vec_d=NULL;
}
//Constructor de la matriz, recibe numero de filas y columnas
template<typename T>
cudamat<T>::cudamat(int r,int c){
    this->rows=r;
    this->cols=c;
    this->type_size=sizeof(T);
    this->capacity=r*c;
    this->vec_d=NULL;
    this->mat_d=NULL;
    this->row_c=r;
    cudaMalloc((void**)&this->mat_d,r*sizeof(int*));
    cudaMalloc((void**)&this->vec_d,r*c*sizeof(int));
    matrixset<T><<<1,1>>>(this->mat_d,this->vec_d,r,c);
    cudaDeviceSynchronize();
}
//Funcion que cambia las dimensiones de la matriz previamente declarada
//Modifica los valores de los punteros para que sea congruente con la nueva forma de la matriz.
template<typename T>
void cudamat<T>::resize(int n_r,int n_c){
    int ns=n_c*n_r;
    if(ns<=capacity){
        if(n_r<=this->row_c){
            this->rows=n_r;
        }else{
            #ifdef DEBUG
                printf("se realocara el tamanio de las columnas de %p ",this->mat_d);
            #endif
            this->row_c=n_r;
            this->rows=n_r;
            cudaFree(mat_d);
            cudaMalloc((void**)&this->mat_d,this->row_c*sizeof(int*));
            #ifdef DEBUG
                printf("Y quedara en la posicion %p\n",this->mat_d);
            #endif
        }
        this->cols=n_c;
    }else{
        #ifdef DEBUG
            printf("Se realojara la matriz completa, pasara de las posicionees %p y %p ",this->mat_d,this->vec_d);
        #endif
        cudaFree(this->mat_d);
        cudaFree(this->vec_d);
        this->row_c=2*n_r;
        this->rows=n_r;
        this->cols=n_c;
        this->capacity=2*rows*cols;
        cudaMalloc((void**)&this->mat_d,rows*sizeof(int*));
        cudaMalloc((void**)&this->vec_d,capacity*sizeof(int));
        matrixset<T><<<1,1>>>(this->mat_d,this->vec_d,this->rows,this->cols);
        #ifdef DEBUG
            cudaDeviceSynchronize();
            printf("a las posiciones %p y %p \n",this->mat_d,this->vec_d);
        #endif
    }
}
//Destructor de la matriz, libera la memoria en GPU 
template<typename T>
cudamat<T>::~cudamat(){
    #ifdef DEBUG
        printf("Se liberara la memoria en GPU de la posicion %p y %p\n",this->vec_d,this->mat_d);
    #endif
    if(this->vec_d!=NULL)cudaFree(this->vec_d);
    if(this->mat_d!=NULL)cudaFree(this->mat_d);
    this->vec_d=NULL;
    this->mat_d=NULL;
}
//Metodo de copia de datos CPU->GPU, los datos de la matriz deben ser ingresados en un vector organizado en forma row-major
template<typename T>
void cudamat<T>::to_device(const std::vector<T>&a){
    if(a.size()==this->capacity){
        cudaMemcpy(this->vec_d,a.data(),this->capacity*sizeof(T),cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }
}
//Metodo que imrpime en pantalla la informacion contenida dentro de la clase.
//Se utiliza principalmente para hacer debug.
template<typename T>
void cudamat<T>::resume(){
    printf("La matriz es de dimensiones (%d,%d)\n",this->rows,this->cols);
    printf("la capacidad del arreglo es de %d\n",this->capacity);
    printf("Mi direccion de memoria es %p\n",this->mat_d);  
}
//Funcion que copia el contenido de una matriz dentro del host al gpu, se hace cargo del manejo de memoria en su totalidad.
//solo fuciona correctamente si todas las filas del arreglo son del mismo tamano
template<typename T>
void cudamat<T>::to_device(const std::vector<std::vector<T>>&a){
    int n=a.size()*a[0].size();
    if(n>this->capacity)this->resize(a.size(),a[0].size());
    std::vector<T> tmp(n,0);
    int cont=0;
    #ifdef DEBUG
        printf("Los valores a copiar al gpu son:\n");
    #endif
    for(int i=0;i<a.size();i++)
        for(int j=0;j<a[i].size();j++){
            tmp[cont]=a[i][j];
            #ifdef DEBUG
                printf("%f ",(float)tmp[cont]);
            #endif
            cont++;
        }
    cudaMemcpy(this->vec_d,tmp.data(),this->capacity*sizeof(T),cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}
//Metodo de copia de datos GPU->CPU, regresa los datos en un vector organizado en forma row-major
template<typename T>
void cudamat<T>::to_host(std::vector<T>&a){
    a.resize(this->capacity);
    cudaMemcpy(a.data(),this->vec_d,this->capacity*sizeof(int),cudaMemcpyDeviceToHost);
}
//Metodo que regresa la direccion en memoria de la matriz.
template<typename T>
T** cudamat<T>::get_mat(){
    return this->mat_d;
}
//Metodo que imprime en pantalla el contenido de la matriz.
template<typename T>
void cudamat<T>::show(){
    matrixtest<T><<<1,1>>>(this->mat_d,this->rows,this->cols);
    cudaDeviceSynchronize();
}

//Estos son kernels

//Kernel para inicializar Los punteros dentro de la matriz para facilitar el acceso
template<typename T>
__global__ void matrixset(T **mat,T* vec_d,int rows, int cols ){
    if(threadIdx.x==0){
        for(int i=0;i<rows;i++){
            mat[i]=vec_d+cols*i;
        }
    }
}
//Imprime el contenido de la matriz en pantalla desde el GPU
template<typename T>
__global__ void matrixtest(T **mat,int idim,int jdim){
    if(threadIdx.x==0){
        printf("La matriz contiene los siguientes numeros:\n");
        for(int i=0;i<idim;i++){
            for(int j=0;j<jdim;j++)
                printf("%f ",(float)mat[i][j]);
            printf("\n");
        }
    }
}


//Imprime el conteenido del vector en pantalla
template<typename T>
GLOBAL void vectortest(T *vec, int idim){
    if(threadIdx.x==0){
        printf("La El vector contiene los siguientes numeros:\n");
        for(int i=0;i<idim;i++)
                printf("%f ",(float)vec[i]);
        printf("\n");
    }
}

//Terminan los kernels




//---------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------------------------------------



//Constructor del generador aleatorio uniforme 0-1,asigna tamaño maximo de muestras a tomar
//en la memoria del gpu
    gpu_randgen::gpu_randgen(int max_size){
        int sd=time(NULL);
        #ifdef __NVCC__
        CURAND_CALL(curandCreateGenerator(&this->gen,CURAND_RNG_PSEUDO_DEFAULT));
        #ifdef DEBUG
            printf("La semilla aleatoria es %d\n",sd);
            CURAND_CALL(curandSetPseudoRandomGeneratorSeed(this->gen,sd));
        #else
            CURAND_CALL(curandSetPseudoRandomGeneratorSeed(this->gen,sd));
        #endif
        #endif
        this->dats.resize(max_size);   
        //this->dats=cuda_vec<float>(max_size);
        this->maxs=max_size;
    }
    gpu_randgen::~gpu_randgen(){ 
        CURAND_CALL(curandDestroyGenerator(this->gen));
        this->dats.~cuda_vec();
    }
    //Genera una muestra de numeros aleatorios segun el tamaño predefinido en el constructor
    void gpu_randgen::generate(){
        CURAND_CALL(curandGenerateUniform(this->gen,this->dats.data(),this->maxs));
        cudaDeviceSynchronize();
    }
    void gpu_randgen::to_host(vec_f &datos){
        datos.resize(this->maxs);
        this->dats.copy_to(dats.data());
    }   

    float* gpu_randgen:: get_dat(){
        return this->dats.data();
    }

    void gpu_randgen::set(int i){
        if(i>maxs){
            #ifdef DEBUG
                printf("\nResize of randgen\n");
            #endif
            dats.resize(i);
            this->maxs=i;
        }
    }

//Metodo para imrpimir el contenido del generador de numeros aleatorios
void gpu_randgen::show(){
    this->dats.show();
}
//Metodo que sirve para intercambiar el contenido entre 2 variables de tipo cuda_mat.
void swap(cudamat<float> &a,cudamat<float>&b){
    float **t_m,*t_v;
    #ifdef DEBUG
    printf("A contiene %p %p antes de\n",a.mat_d,a.vec_d);
    printf("B contiene %p %p antes de\n",b.mat_d,b.vec_d);
    #endif
        t_v=a.vec_d;a.vec_d=b.vec_d;b.vec_d=t_v;
        t_m=a.mat_d;a.mat_d=b.mat_d,b.mat_d=t_m;
    #ifdef DEBUG
        printf("A contiene %p %p Despues\n",a.mat_d,a.vec_d);
        printf("B contiene %p %p Despues\n",b.mat_d,b.vec_d);
    #endif
}
//Kernel que se utiliza para asignar valor a una variable dentro del gpu 
void GLOBAL set_val(float* val,float x){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx==0){
        *val=x;
    }
}