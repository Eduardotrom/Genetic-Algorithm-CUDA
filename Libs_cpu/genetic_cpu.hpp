#ifndef GENETIC_CPU
#define GENETIC_CPU
#include <vector>
#include<cmath>
#include<iostream>
#include<algorithm>
#include<fstream>
#define URAND rand()/(float)RAND_MAX
typedef std::vector<float> vec_f;
typedef std::vector<vec_f> mat_f;
class individuo{
    private:    
        float fit;
    public:
        std::vector<float> gen;
        individuo(){}
        individuo(vec_f &v);
        individuo(int n);
        void set_fit(float nf);
        void set_gen(vec_f &v);
        float get_fit();
        vec_f get_vec();
        bool operator<(individuo &b);
        void show();
};
typedef std::vector<individuo> pob_v;

class Genetic_cpu{
    private:
        std::vector<individuo> poblacion,hijos;
        std::vector<vec_f> lims;
        vec_f max,min,avg,conv;
        individuo best;
        float cp,pc,pm,eta,alpha;
        int popSize,nVars;
    public:
        Genetic_cpu(){};
        Genetic_cpu(int popSize,int nVars,float cp_i, float pc_i,float pm_i,float eta_i,mat_f lims);
        void optimiza(int gen);
        void reproduce(pob_v &padres,pob_v &hijos,float alpha);
        void muta(pob_v &pop);
        void evalua(pob_v &pop);
        void pop_init();
        void print(std::string nom);
};


float rosembrock(vec_f &v);

#endif