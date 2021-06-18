#include"genetic_cpu.hpp"
individuo::individuo(int n){
    this->gen.resize(n);
}

individuo::individuo(vec_f &v){
    this->gen=v;
}
void individuo::set_fit(float nf){
    this->fit=nf;
}
void individuo::set_gen(vec_f &v){
    this->gen=v;
}
float individuo::get_fit(){
    return this->fit;
}
vec_f individuo::get_vec(){
    return this->gen;
}

bool individuo::operator<(individuo &b){
    return this->fit<b.get_fit();
}
void individuo::show(){
    printf("El individuo tiene un fitness de %f ",this->fit);
    printf("y contiene los valores:\n");
    for(auto x:gen)
        printf("%f ",x);
    printf("\n");
}
//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------


Genetic_cpu::Genetic_cpu(int popSize,int nVars,float cp_i, float pc_i,float pm_i,float eta_i,mat_f lims){
    this->popSize=popSize;
    this->nVars=nVars;
    this->cp=cp_i;
    this->pc=pc_i;
    this->alpha=pc;
    this->pm=pm_i;
    individuo tmp(nVars);
    this->poblacion.resize(popSize,tmp);
    this->hijos=poblacion;
    this->best=tmp;
    this->lims=lims;
    best.set_fit(1e30);
}
void Genetic_cpu::pop_init(){
    vec_f t(nVars);
    for(int i=0;i<popSize;i++){
        for(int j=0;j<nVars;j++){
            this->poblacion[i].gen[j]=lims[j][0]+URAND*(lims[j][1]-lims[j][0]);
        }
    }
    evalua(this->poblacion);
    
}

void Genetic_cpu::reproduce(pob_v &padres,pob_v &hijos,float alpha){
    vec_f t1(this->nVars),t2(this->nVars);
    int p1,p2,p3,p4;
    this->hijos=padres;
    for(int i=0;i<popSize;i+=2){
        if(URAND>this->pc)continue;
        p1=rand()%this->popSize;p2=rand()%this->popSize;
        p1=(padres[p1]<padres[p2])?p1:p2;
        p3=rand()%this->popSize;p4=rand()%this->popSize;
        p2=(padres[p3]<padres[p4])?p3:p4;
        for(int j=0;j<this->nVars;j++){
            alpha=-0.5+URAND*(1.5-(-1.5));
            t1[j]=padres[p1].gen[j]*alpha+padres[p2].gen[j]*(1.0-alpha);
            t2[j]=padres[p2].gen[j]*alpha+padres[p1].gen[j]*(1.0-alpha);
        }
        hijos[i].set_gen(t1);
        hijos[i+1].set_gen(t2);
    }   
}
void Genetic_cpu::muta(pob_v &pop){
    for(int p=0;p<this->popSize;p++){
        if(URAND>this->pm)continue;
        float u=URAND,delta=0,dist=0;
        float gen=std::floor(URAND*this->nVars),temp=pop[p].gen[gen];
        if(u<=0.5){
            delta=std::pow(2*u,1.0/(1.0+this->eta));
            dist=temp-this->lims[gen][0];
            dist*=delta;
        }else{
            delta=1-std::pow(2*(1-u),1.0/(1.0+this->eta));
            dist=temp-this->lims[gen][0];
            dist*=delta;
        }
        temp+=dist;
        pop[p].gen[gen]=(temp>=this->lims[gen][0]&&temp<=this->lims[gen][1])?\
                        temp:lims[gen][0]+URAND*(lims[gen][1]-lims[gen][0]);
    }
}
void Genetic_cpu::evalua(pob_v &pop){
    double temp=0;
    for(int i=0;i<pop.size();i++){
        temp=rosembrock(pop[i].gen);
        pop[i].set_fit(temp);
    }
}
void Genetic_cpu::optimiza(int gen){
    this->best=this->poblacion[0];
    max.resize(gen);min.resize(gen);avg.resize(gen);
    for(int g=0;g<gen;g++){
        std::sort(poblacion.begin(),poblacion.end());
        if(this->poblacion[0]<best)best=poblacion[0];
        conv.push_back(best.get_fit());
        max[g]=poblacion[popSize-1].get_fit();
        min[g]=poblacion[0].get_fit();
        float t=0;
        for(auto x:poblacion)
            t+=x.get_fit();
        avg[g]=t/popSize;
        reproduce(poblacion,hijos,this->alpha);
        muta(hijos);
        evalua(hijos);
        poblacion=hijos;
    }
    #ifdef VERBOSE
        printf("el mejor elemento es:");
        this->best.show();
    #endif
}


void Genetic_cpu::print(std::string nom){
    std::ofstream f(nom);
    for(int i=0;i<avg.size();i++){
        f<<min[i]<<" "<<avg[i]<<" "<<max[i]<<std::endl;
    }
    f.close();
    f.open(nom+"convergencia.dat");
    for(int i=0;i<conv.size();i++){
        f<<i<<" "<<conv[i]<<std::endl;
    }
    f.close();
}


float rosembrock(vec_f &v){
    int n=v.size();
    float sum=0;
    for(int i=0;i<n-1;i++){
        sum+=100*std::pow(v[i+1]-std::pow(v[i],2),2);
        sum+=std::pow(v[i]-1.0,2);
    }
    return sum;
}