#include "gen_gpu.h"
#include <vector>
//Constructor para el algoritmo genetico, almacena los valores necesarios para la ejecucion
//Y asigna los espacios necesarios dentro del gpu para poder realizar las ejecuciones.
gpu_GA::gpu_GA(int populationSize,int varNum,float cp,float eta,mat_f lims){
    this->popSize=populationSize;
    this->varNum=varNum;
    this->cp=cp;
    this->eta=eta;
    this->fit.resize(popSize);  
    this->parents.resize(popSize,varNum); 
    this->children.resize(popSize,varNum);
    this->best.resize(this->varNum);
    limits.resize(varNum,2);
    limits.to_device(lims);
    fgen.set(3*varNum*popSize);
    cudaDeviceSynchronize();
    #ifdef DEBUG
        printf("Los limites son:\n");
        limits.show();
        printf("Constructor del genetico completado\n");
        printf("\nLos datos de parents son\n");
        this->parents.resume();
        printf("\nLos datos de children son\n");
        this->children.resume();
    #endif
}
//Metodo que llama al kernel para generar la poblacion inicial dentro de los limites establecidos al crear la clase genetico.
void gpu_GA::generate_population(){
    this->fgen.generate();
    CUDA__ERROR_CHK;
    blk=dim3(std::ceil(this->popSize/(float)32.0),this->varNum);
    tpb=dim3(32);
    #ifdef DEBUG
        printf("Se generara una poblacion de %d pobladores de %d variables\n",this->popSize,this->varNum);
    #endif
    population_init<<<blk,tpb>>>(parents.get_mat(),fgen.get_dat(),\
                this->limits.get_mat(),this->varNum,this->popSize);
    cudaDeviceSynchronize();
    CUDA__ERROR_CHK;
    #ifdef DEBUG
        printf("El contenido de la poblacion inicial es:\n");
        this->parents.show();
    #endif
}
//Funcion que se encarga de realizar la optimizacion, llama los kernels y hace los movimientos en memoria pertientes
//asi como calcular el numero necesario de invoccaciones de parametros para cada kernel.
void gpu_GA::optimiza(int iters,float alpha){
    conv.resize(iters);
    std::vector<std::vector<float>> max(iters),min(iters),hist(iters);
    float btemp=MAXFLOAT;
    float threads=32;
    this->blk=dim3(std::ceil(popSize/threads),varNum);
    this->tpb=dim3(threads);
    //Se evalua la funcion objetivo en gpu para la poblacion inicial
    //Rosembrock_gpu<<<blk,tpb>>>(this->parents.get_mat(),this->fit.data(),this->varNum,this->popSize);
    shpere<<<blk,tpb>>>(this->parents.get_mat(),this->fit.data(),this->varNum,this->popSize);
    cudaDeviceSynchronize();
    CUDA__ERROR_CHK;
    std::vector<float> temp;
    for(int gen=0;gen<iters;gen++){
        this->blk=dim3(std::ceil(popSize/threads),1);
        this->tpb=dim3(threads,varNum);
        //PArte de la evaluacion dentro del ciclo a optimizar 
        //Rosembrock_gpu<<<blk,tpb>>>(this->parents.get_mat(),this->fit.data(),this->varNum,this->popSize);
        shpere<<<blk,tpb>>>(this->parents.get_mat(),this->fit.data(),this->varNum,this->popSize);
        cudaDeviceSynchronize();
        CUDA__ERROR_CHK;
        //Se busca el elemento con mejor fitness
        findbest<<<1,1>>>(this->parents.get_mat(),this->fit.data(),this->best.get_index(),this->best.get_fit(),this->best.get_gen(),this->popSize,this->varNum);
        cudaDeviceSynchronize();
        CUDA__ERROR_CHK;
    #ifdef STATISTICS
        //En esta parte se reailzan las operaciones para obtener informacion soble los resultados.
        //se evita en ejecuciones normales ya que realiza muchas copias a cpu.
        hist_up<<<1,1>>>(this->best.get_fit(),this->conv.data(),gen);
        cudaDeviceSynchronize();
        fit.to_host(temp);
        auto it=std::max_element(temp.begin(),temp.end());
//        btemp=maxx(temp);
        max[gen].push_back(*it);
        //btemp=minn(temp);
        it=std::min_element(temp.begin(),temp.end());
        min[gen].push_back(*it);
        btemp=mean(temp);
        hist[gen].push_back(btemp);
    #endif
        fgen.generate();
        blk=dim3(std::ceil(popSize/(threads)));
        tpb=dim3(threads);
        //Se realiza la seleccion por toreno binario.
        selection<<<blk,tpb>>>(fgen.get_dat(),fgen.get_dat()+popSize,fit.data(),this->popSize);
        cudaDeviceSynchronize();
        blk=dim3(std::ceil(popSize/(2.0*threads)),this->varNum);
        tpb=dim3(threads,varNum);
        int arrsz=this->varNum*sizeof(float);
        //Se realiza el corssover.
        crossover<<<blk,tpb,(arrsz)>>>(this->parents.get_mat(),this->children.get_mat(),this->fgen.get_dat()+popSize,alpha,\
                    this->fgen.get_dat(),this->limits.get_mat(),this->varNum,this->popSize);
        cudaDeviceSynchronize();
        #ifdef DEBUG
            printf("La poblacion de hijos es:\n");
            this->children.resume();
            this->children.show();
            printf("\n");
        #endif
        fgen.generate();
        blk=dim3(std::ceil(popSize/(threads)),1);
        tpb=dim3(threads,1);
        //Se realiza la mutacion de los hijos
        mutation<<<blk,tpb>>>(this->children.get_mat(),fgen.get_dat(),this->limits.get_mat(),this->varNum,this->popSize,this->cp);
        cudaDeviceSynchronize();
        #ifdef DEBUG
            printf("La poblacion de hijos mutados es:\n");
            this->children.show();
            printf("\n");
        #endif
        //Se swapea la poblacion para utilizar la nueva.
        swap(this->parents,this->children);
        #ifdef DEBUG
            printf("Los padres ahora contienen:\n");
            this->parents.show();
        #endif
    }

    std::vector<float> tst(this->varNum);
    this->best.gens.copy_to(tst.data());
    #ifdef STATISTICS
    printf("El mejor individuo al final es: \n");
        for(float x:tst)
            printf("%f ",x);
        printf("\n");

        this->max_m.resize(iters);
        this->min_m.resize(iters);
        this->hist_m.resize(iters);
        for(int i=0;i<iters;i++){
            //this->max_m[i]=maxx(max[i]);
            //this->min_m[i]=minn(min[i]);
            //this->hist_m[i]=mean(hist[i]);
            this->max_m[i]=*std::max_element(max[i].begin(),max[i].end());
            this->min_m[i]=*std::min_element(min[i].begin(),min[i].end());
            this->hist_m[i]=mean(hist[i]);
        }
    #endif
}
//Metodo que imprime en 2 archivos distintos los resultados del genetico.
void gpu_GA::print(std::string nom){
    std::ofstream f(nom);
    int n=this->hist_m.size();
    for(int i=0;i<n;i++){
        f<<this->min_m[i]<<" "<<this->hist_m[i]<<" "<<this->max_m[i]<<std::endl;
    }
    f.close();
    for(int i=0;i<4;i++)
        nom.pop_back();
    f.open(nom+"_convergencia.txt");
    vec_f tmp;
    this->conv.to_host(tmp);
    for(int i=0;i<tmp.size();i++){
        f<<i<<" "<<tmp[i]<<std::endl;
    }
    f.close();
}

//Constructor de la clase best ind, asigna la memoria necesaria
best_ind_gpu::best_ind_gpu(int dims){
    this->gens.resize(dims);
    this->data.resize(5);
    set_val<<<1,1>>>(this->get_fit(),MAXFLOAT);
    cudaDeviceSynchronize();
    CUDA__ERROR_CHK;
}
//Funcion que cambia el tamanio del individo
void best_ind_gpu::resize(int dims){
    this->gens.resize(dims);
    this->data.resize(5);
    set_val<<<1,1>>>(this->get_fit(),MAXFLOAT);
    cudaDeviceSynchronize();
    CUDA__ERROR_CHK;
}
//metodo que devuelve el puntero del fitness del individuo en gpu
float* best_ind_gpu::get_fit(){
    return this->data.data();
}
//Metodo que devuelve el numero del poblador de que tomo el valor
float* best_ind_gpu::get_index(){
    return this->data.data()+1;
}
//Metodo que devuelve puntero a los genes del individuo.
float* best_ind_gpu::get_gen(){
    return this->gens.data();
}

