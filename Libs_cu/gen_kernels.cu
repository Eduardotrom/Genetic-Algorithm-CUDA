#include"gen_kernels.h"

GLOBAL void population_init(float **population,float *randnum,float **lims,int cols,int rows){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int idy=blockIdx.y*blockDim.y+threadIdx.y;
    if(idx<rows&&idy<cols){
        #ifdef DEBUG
            printf("Soy el hilo (%d,%d) y chambeare \n",idx,idy);
        #endif
        population[idx][idy]=lims[idy][0]+randnum[idx*cols+idy]*(lims[idy][1]-lims[idy][0]);
    }
}
//funcion de seleccion, se debe llamar con bloque de tamanio pob_size/(2*32) y 32 hilos por bloque
GLOBAL void selection(float *randidx,float* chosen,float *fitness,int pob_size){
    IDXMC IDYMC
    if(idx<pob_size&&idy==0){;
        int ind1=randidx[idx]*pob_size,ind2=randidx[idx+1]*pob_size;
        chosen[idx]=(fitness[ind1]<=fitness[ind2])?ind1:ind2;
        #ifdef DEBUG
            printf("entre %d y %d seleccione a %f\n",ind1,ind2,chosen[idx]);
        #endif
    }
}
//Kernel de crossover
GLOBAL void crossover(float **parents, float **children,float * chosen,float alpha,float * randnum, float**lims,int cols, int rows){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    int idy=blockIdx.y*blockDim.y+threadIdx.y;
    int bidy= threadIdx.y,bidx=threadIdx.x;
    extern __shared__ float limss[];
    float *upper,*lower,*h1,*h2;
    upper=(limss+2*cols);
    lower=(limss);
    if(bidx==0&&bidy<cols){
        //Se cargan a memoria compartida los limites de cada indice en el bloque
        upper[bidy]=lims[bidy][1];
        lower[bidy]=lims[bidy][0];
        #ifdef DEBUG
            printf("lims[%d][0]=%f lower[%d]=%f\n",bidy,lims[bidy][0],bidy,lower[bidy]);
            printf("lims[%d][1]=%f upper[%d]=%f\n",bidy,lims[bidy][1],bidy,upper[bidy]);
        #endif
    }
    __syncthreads();
    if(idx<rows/2&&idy<cols){
        h1=children[2*idx];
        h2=children[2*idx+1];

        //se convierten de numeros aleatorios uniformes [0,1] a enteros.
        int p1=chosen[2*idx],p2=chosen[2*idx+1];
        #ifdef DEBUG
            printf("Soy el hilo (%d,%d) y entre a trabajar\n",idx,idy);
            printf("Los valores que leeo de la memoria compartida son:\n l[%d]=%f u[%d]=%f\n",bidy,lower[bidy],bidy,upper[bidy]);
        #endif
        float t1,t2,tr1,tr2;
        //Se generan posiciones sustitutas posibles en caso de que el algoritmo genere numeros fuera de rango
        tr1=lower[bidy]+randnum[2*idx]*(upper[bidy]-lower[bidy]);
        tr2=lower[bidy]+randnum[2*idx+1]*(upper[bidy]-lower[bidy]);
        alpha=-0.5+randnum[2*idx]*(2.0);
        //Se utiliza el operador de cruza para geneticos reales en lugar de SBX
        t1=parents[p1][idy]*alpha+parents[p2][idy]*(1.0-alpha);
        t2=parents[p2][idy]*alpha+parents[p1][idy]*(1.0-alpha);
        #ifdef DEBUG
            printf("Los padres a leer son: p1=%d p2=%d\n",p1,p2);
            printf("t1=%f t2=%f\n",t1,t2);
        #endif
        //Se asigna a los hijos el valor calculado o bien el nuevo aleatorio si es que el generado sale de rango.
        h1[bidy]=((t1>=lower[bidy])&&(t1<=upper[bidy]))?t1:tr1;
        h2[bidy]=((t2>=lower[bidy])&&(t2<=upper[bidy]))?t2:tr2;
        #ifdef DEBUG
            printf("children[%d][%d]=%f\n",2*idx,bidy,children[2*idx][bidy]);
            printf("children[%d][%d]=%f\n",2*idx+1,bidy,children[2*idx+1][bidy]);
        #endif
    }
}

//Kernel de mutacion polinomial
GLOBAL void mutation(float **population,float* randnum,float** lims,int cols, int rows,float CP){
    IDXMC IDYMC
    if(idx<rows&&idy<1){
        float f=cols*randnum[3*rows+idx];
        int gen=f;
        gen=(f-gen>0)?gen+1:gen;
        gen--;
        #ifdef DEBUG
            printf("voy a ver si muta el hijo %d del gen %d\n",idx,gen);
        #endif
            mutation_d(population[idx],randnum[2*rows+idx],lims[gen][1],lims[gen][0],gen);
        }

}

DEVICE float get_delt(float u ){
    float delta;
    int sign=1;
    if(u<=-1.0)u=-1.0;
    if(u>1.0)u=1.0;
    if(u<0){
        u=-u;
        sign=-1;
    }
    delta=1.0-powf((1.0-u),(1.0+0.5));
    return sign*delta;
}

//Funcion de mutacion de una implementacion que ya no se utilizo
DEVICE void mutation_d(float* hijo,float randn,float upper,float lower,int gen){
    float x,d1,d2,deltal,deltar,eta=0.7;
    x=hijo[gen];
    d1=x-lower;
    d2=upper-x;
    deltal=powf(2*randn,1/(1+eta));
    deltar=1-powf((2*(1-randn)),1/(1+eta));
    deltal*=d1;
    deltar*=d2;
    deltal=(deltal>=lower&&deltal<=upper)?deltal:deltal/2;
    deltar=(deltar>=lower&&deltar<=upper)?deltar:deltar/2;
    hijo[gen]+=(randn<=0.5)?deltal:deltar;
}
//Kernel para evaluaro rosembrock en gpu
void GLOBAL Rosembrock_gpu(float** pob,float *results,int cols,int rows){
    IDXMC IDYMC
    results[idx]=0;
    __syncthreads();
    float psum=0;
    if(idx<rows&&idy<cols-1){
        float t1=(pob[idx][idy]*pob[idx][idy]-pob[idx][idy+1]);
        float t2=(pob[idx][idy]-1.0);
        psum=100*t1*t1+t2*t2;
    }
    atomicAdd(results+idx,psum);
}

void GLOBAL shpere(float** pob,float *results,int cols,int rows){
    IDXMC IDYMC
    results[idx]=0;
    __syncthreads();
    float psum=0;
    if(idx<rows&&idy<cols){
        psum=(pob[idx][idy]*pob[idx][idy]);
    }
    atomicAdd(results+idx,psum);
}


void GLOBAL findbest(float** population,float *fit,float *index,float *bestfit,float*bestgen,int popsize,int varnum){
    IDXMC IDYMC
    if(idx==0){
        for(int i=0;i<popsize;i++){
            if(fit[i]<*bestfit){
                *bestfit=fit[i];
                *index=i;
                for(int j=0;j<varnum;j++){
                    bestgen[j]=population[i][j];
                }
            }
        }
        #ifdef DEBUG
            printf("El mejor hasta ahora es: %e\n",*bestfit);
        #endif
        /*for(int j=0;j<varnum;j++){
           population[0][j]=bestgen[j];
       }
       fit[0]=*bestfit;*/
    }
}


GLOBAL void hist_up(float * fit,float * conv,int index){
    IDXMC
    if(idx==0){
        conv[index]=*fit;
    }
}