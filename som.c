#include "include/som.h"
#include "include/common.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

//Uncomment this if you want to write history files for the stabilization process
//som1D.data will store history for 1D map
//som2D.data will store history for 2D map
//som3D.data will store history for 3D map
//#define TRACE_SOM 1

#pragma region Config Section
somConfig* getsomDefaultConfig(){
    somConfig* config = malloc(sizeof(somConfig));
    
#ifdef TRACE_SOM
    config->normalize = 0;
    config->stabilizationTrigger = 0.01;
#else
    config->normalize = 1;
    config->stabilizationTrigger = 0.0001;
#endif
    config->dimension = twoD;
    config->alpha = 0.99;
    config->alphaDecreaseRate=0.99;
    config->sigma = 0.99;
    config->sigmaDecreaseRate=0.95;
    config->radiusDecreaseRate = 5;
    config->initialPercentCoverage = 0.6;
    config->maxEpisodes = 1000;
}


void resetConfig(somConfig* config)
{
    config->map_b=0;
    config->map_c=0;
    config->map_r=0;
    config->nw=0;
    config->radius =0;
}
#pragma endregion

#pragma region Init Section
//Return a random value between a boundary
double getRandom(dataBoundary boundary){
    return (((double)rand()/RAND_MAX)*(boundary.max - boundary.min)) + boundary.mean;
}

void initNeuron(somNeuron*n, somConfig* config, dataBoundary *boundaries, int b, int r, int c)
{
    int p = config->p;
    n->b=b;
    n->r=r;
    n->c=c;
    n->v=(double*)malloc(p * sizeof(double));
    n->neighbours = NULL;
    n->nc = 0;
    n->updates = (double*)calloc(p , sizeof(double));
    n->entries = (int*)malloc(sizeof(int));
    n->ec=0;
    n->isStabilized = 0;
    for(int j=0;j<p;j++)
    {
        n->v[j]= getRandom(boundaries[j]);
    }
}

//Initialize SOM 1D weights with random values based on parameters boundaries
void initialize1D(somNeuron *weights, somConfig* config, dataBoundary *boundaries){
    for(int i=0;i<config->nw; i++){
        initNeuron(&weights[i], config, boundaries,-1,-1,i);
    }
}

//Initialize SOM 2D weights with random values based on parameters boundaries
void initialize2D(somNeuron** weights, somConfig* config, dataBoundary *boundaries){
    for(int i=0;i<config->map_r; i++)
    {
        weights[i] = (somNeuron*)malloc(config->map_c * sizeof(somNeuron));
        for(int j=0;j<config->map_c;j++)
        {
            initNeuron(&weights[i][j], config, boundaries,-1,i,j);
        }
    }
}

//Initialize SOM 3D weights with random values based on parameters boundaries
void initialize3D(somNeuron*** weights, somConfig* config, dataBoundary *boundaries){
    for(int i=0;i<config->map_b; i++)
    {
        weights[i] = (somNeuron**)malloc(config->map_r * sizeof(somNeuron*));
        for(int j=0;j<config->map_r;j++)
        {
            weights[i][j] = (somNeuron*)malloc(config->map_c * sizeof(somNeuron));
            for(int k=0;k<config->map_c;k++){
               initNeuron(&weights[i][j][k], config, boundaries,i,j,k);
            }
        }
    }
}

void setMap1DSize(somConfig *config)
{
    if(!config->map_c)
    {
        config->map_c = config->nw; 
    }
    else
    {
        config->nw = config->map_c;
    }
    if(!config->radius)
    {
        config->radius = ((config->initialPercentCoverage*config->nw)-1)/2;
    }
}

void set2DMapSize(somConfig *config)
{
    if(!config->map_r|| !config->map_c)
    {
        //Arbitrary 4/3 ratio (16/9 didn't exist in the eighties ;))
        double ratio = 4.0/3;
        int y = floor(sqrt(config->nw / ratio));
        y-=config->nw%y;
        int x = config->nw/y;
        config->map_r=y;
        config->map_c=x;      
    }

    config->nw = config->map_r*config->map_c;
    if(!config->radius){
        config->radius = ceil((sqrt(config->initialPercentCoverage*config->nw) - 1)/2);
    }
}

void set3DMapSize(somConfig *config)
{
    if(!config->map_b || !config->map_r|| !config->map_c)
    {
        int size = ceil(cbrt(config->nw));
        config->map_r=size;
        config->map_c=size;
        config->map_b=size;       
    }
    config->nw = config->map_b*config->map_r*config->map_c;
   
    if(!config->radius){
        config->radius = ceil((cbrt(config->initialPercentCoverage*config->nw)-1)/2);
    }
}

void* getsom1D(dataVector* data, somConfig *config, dataBoundary* boundaries)
{
    setMap1DSize(config);
    somNeuron *weights = (somNeuron*)malloc(config->nw *sizeof(somNeuron));
    initialize1D(weights, config, boundaries);
    return weights;
}

void* getsom2D(dataVector* data, somConfig *config, dataBoundary* boundaries)
{
    set2DMapSize(config);
    somNeuron **weights = (somNeuron**)malloc(config->map_r * sizeof(somNeuron*));
    initialize2D(weights, config, boundaries);
    return weights;
}

void* getsom3D(dataVector* data, somConfig *config, dataBoundary* boundaries)
{
    set3DMapSize(config);
    somNeuron ***weights = (somNeuron***)malloc(config->map_b * sizeof(somNeuron**));
    initialize3D(weights, config, boundaries);
    return weights;
}

#pragma endregion

#pragma region  Learning Section

#pragma region Find Section
//Calculate the euclidian distance between 2 vectors using p dimensions
double distance_function(double *v, double *w, int p)
{
    double sum=0;
    for(int i=0;i<p;i++){
        sum+= pow((v[i] - w[i]),2);
    }
    return sqrt(sum);
}

void updateWinnerResults(somNeuron** results, dataVector* v, somNeuron* n, int p, int* count, double* minValue)
{
    double min = *minValue;
    int c = *count;
    double distance = distance_function(v->v, n->v, p);
    if(distance<min){
        results[c-1]= n;
        min = distance;
    }
    else if(distance == min){
        results = realloc(results, ++c*sizeof(int));
        results[c-1]= n;
    }
    *minValue = min;
    *count = c;
}

somNeuron* getRandomNeuron1D(void* weights, somConfig* config)
{
    int c = rand()%config->map_c;
    return &((somNeuron*)weights)[c];
}

somNeuron* getRandomNeuron2D(void* weights, somConfig* config)
{
    int r = rand()%config->map_r;
    int c = rand()%config->map_c;
    return &((somNeuron**)weights)[r][c];
}

somNeuron* getRandomNeuron3D(void* weights, somConfig* config)
{
    int b = rand()%config->map_b;
    int r = rand()%config->map_r;
    int c = rand()%config->map_c;
    return &((somNeuron***)weights)[b][r][c];
}

somNeuron* getNearest(dataVector* v, void* weights, somConfig* config, void (*getnbfp)(somNeuron*, void*, somConfig*), somNeuron*(*getrndfp)(void*, somConfig*))
{
    if(!v->lastWinner)
    {
        v->lastWinner = getrndfp(weights, config);
    }
    somNeuron* neuron = v->lastWinner;
    if(neuron->isStabilized)
    {
        return neuron;
    }
    if(!neuron->neighbours)
    {
        getnbfp(neuron, weights, config);
    }
    somNeuron **results = malloc(sizeof(somNeuron*));
    int count = 1;
    double minValue = __DBL_MAX__;
    int p = config->p;
    for(int i = 0; i<neuron->nc; i++)
    {
        somNeuron* n = neuron->neighbours[i];
        updateWinnerResults(results, v,  n, p, &count, &minValue);
    }
    int selectedIndex = count > 1 ? rand()%count : 0;
    somNeuron* result = results[selectedIndex];
    free(results);
    return result;
}

void find_winner_fromNeighbours(dataVector* v, void * weights, somConfig* config, void (*getnbfp)(somNeuron*, void*, somConfig*), somNeuron*(*getrndfp)(void*, somConfig*))
{
    somNeuron* nearest = getNearest(v, weights, config, getnbfp, getrndfp);
    while(nearest!=v->lastWinner)
    {
        v->lastWinner = nearest;
        nearest = getNearest(v, weights, config, getnbfp, getrndfp);
    }
}

void getNeighbours1D(somNeuron* n, void* weights, somConfig* config)
{
    somNeuron* som = (somNeuron*)weights;
    int start = max(0, n->c - config->radius);
    int end = min(config->map_c, n->c + config->radius);
    int in=0;
    n->neighbours = (somNeuron**)malloc(sizeof(somNeuron*));
    for(int i=start;i<end;i++)
    {
        somNeuron* nb = &som[i];
        n->neighbours = realloc(n->neighbours, (in+1)*sizeof(somNeuron*));
        n->neighbours[in]= nb;
        in++;
    }
    n->nc=in;
}

void getNeighbours2D(somNeuron* n, void* weights, somConfig* config)
{
    somNeuron** som = (somNeuron**)weights;
    int start_r = max(0, n->r - config->radius);
    int end_r = min(config->map_r, n->r + config->radius);
    int start_c = max(0, n->c - config->radius);
    int end_c = min(config->map_c, n->c + config->radius);
    int rc = (end_r - start_r);
    int cc = (end_c - start_c);
    int in=0;
    n->neighbours = (somNeuron**)malloc(sizeof(somNeuron*));
    for(int i=start_r;i<end_r;i++)
    {
        for(int j=start_c;j<end_c;j++)
        {
            somNeuron* nb = &som[i][j];
            n->neighbours = realloc(n->neighbours, (in+1)*sizeof(somNeuron*));
            n->neighbours[in]= nb;
            in++;
        }
    }
    n->nc = in;
}

void getNeighbours3D(somNeuron* n, void* weights, somConfig* config)
{
    somNeuron*** som = (somNeuron***)weights;
    int start_r = max(0, n->r - config->radius);
    int end_r = min(config->map_r, n->r + config->radius);
    int start_c = max(0, n->c - config->radius);
    int end_c = min(config->map_c, n->c + config->radius);
    int start_b = max(0, n->b - config->radius);
    int end_b = min(config->map_b, n->b + config->radius);
    int rc = end_r - start_r;
    int cc = end_c - start_c;
    int bc = end_b - start_b;
    int in=0;
    n->neighbours = (somNeuron**)malloc(sizeof(somNeuron*));
    for(int i=start_b;i<end_b;i++)
    {
        for(int j=start_r;j<end_r;j++)
        {
            for(int k=start_c;k<end_c;k++)
            {
                somNeuron* nb = &som[i][j][k];
                n->neighbours = realloc(n->neighbours, (in+1)*sizeof(somNeuron*));
                n->neighbours[in]= nb;
                in++;
            }

        }
    }
    n->nc = in;
}

//Return the index of the neuron closest to a entry vector using the distance_function for 1D map
void find_winner1D(dataVector* v, somNeuron *weights, somConfig* config)
{
    find_winner_fromNeighbours(v,weights,config, getNeighbours1D, getRandomNeuron1D);
}

//Return the index of the neuron closest to a entry vector using the distance_function for 2D map
void find_winner2D(dataVector* v, somNeuron* lastKnownWinner, somNeuron** weights, somConfig *config)
{
    find_winner_fromNeighbours(v,weights,config, getNeighbours2D, getRandomNeuron2D);
}

//Return the index of the neuron closest to a entry vector using the distance_function for 3D map
void find_winner3D(dataVector* v, somNeuron*** weights, somConfig *config)
{
    find_winner_fromNeighbours(v,weights,config, getNeighbours3D, getRandomNeuron3D);
}
#pragma endregion

#pragma region Update Section

//Return a value between 0 and 1 according to the 1D distance between the winner neuron and another neuron and a neighborhood factor using p parameters
double neighborhood_function1d(somNeuron* winner, somNeuron* n, double sigma)
{
    double wp[1]={winner->c};
    double np[1]={n->c};
    return exp(-(distance_function(wp, np, 1)/(sigma * sigma)));
}

//Return a value between 0 and 1 according to the 2D distance between the winner neuron and another neuron and a neighborhood factor using p parameters
double neighborhood_function2d(somNeuron* winner, somNeuron* n, double sigma)
{
    double wp[2]={winner->c, winner->r};
    double np[2]={n->c, n->r};
    return exp(-(distance_function(wp, np, 2)/(sigma * sigma)));
}

//Return a value between 0 and 1 according to the 3D distance between the winner neuron and another neuron and a neighborhood factor using p parameters
double neighborhood_function3d(somNeuron* winner, somNeuron* n, double sigma)
{
    double wp[3]={winner->c, winner->r, winner->b};
    double np[3]={n->c, n->r, n->b};
    return exp(-(distance_function(wp, np, 3)/(sigma * sigma)));
}

double absd(double v)
{
    return v>0?v:-v;
}

void updateNeuron(dataVector* v, somNeuron* n, double h, somConfig* config)
{
    for(int i=0;i<config->p;i++){
        double delta = config->alpha * h *(v->v[i] - n->v[i]);
        n->updates[i]+= delta;
        n->v[i] += delta;
    }
}

//Update the 1D weight vector of a neuron adding for each parameter the difference between entry vector parameter and current vector parameter
//The difference is multiplied by a learning factor (epsilon) and a the result (value between 0 and 1) of neighborhood function
//If no parameter has been updated significantly return 0, 1 otherwise
void updateNeuron1D(dataVector* v, somNeuron* winner, somNeuron* n, somConfig* config)
{   
    double h = neighborhood_function1d(winner, n, config->sigma);
    updateNeuron(v, n, h, config);
}

//Update the 2D weight vector of a neuron adding for each parameter the difference between entry vector parameter and current vector parameter
//The difference is multiplied by a learning factor (epsilon) and a the result (value between 0 and 1) of neighborhood function
//If no parameter has been updated significantly return 0, 1 otherwise
void updateNeuron2D(dataVector* v, somNeuron* winner, somNeuron* n, somConfig* config){
    double h = neighborhood_function2d(winner, n, config->sigma);
    updateNeuron(v, n, h, config);
}

//Update the 3D weight vector of a neuron adding for each parameter the difference between entry vector parameter and current vector parameter
//The difference is multiplied by a learning factor (epsilon) and a the result (value between 0 and 1) of neighborhood function
//If no parameter has been updated significantly return 0, 1 otherwise
void updateNeuron3D(dataVector* v, somNeuron* winner, somNeuron* n, somConfig* config){
    double h = neighborhood_function3d(winner, n, config->sigma);
    updateNeuron(v, n, h, config);
}

void clear_neigbhoursAtom(somNeuron* n)
{
    free(n->neighbours);
    n->neighbours = NULL;
    n->nc = 0;
}

void clear_neighbours1D(void* weights, somConfig* config)
{
    somNeuron* som = (somNeuron*)weights;
    for(int i=0; i< config->map_c; i++)
    {
        clear_neigbhoursAtom(&som[i]);
    }
}

void clear_neighbours2D(void* weights, somConfig* config)
{
    somNeuron** som = (somNeuron**)weights;
    for(int i=0; i< config->map_r; i++)
    {
        clear_neighbours1D(som[i], config);
    }
}

void clear_neighbours3D(void* weights, somConfig* config)
{
    somNeuron*** som = (somNeuron***)weights;
    for(int i=0; i< config->map_b; i++)
    {
        clear_neighbours2D(som[i], config);
    }
}

//Update winner neuron and neighbours for 1D map return 1 if at least one neuron was updated 0 otherwise
void updateNeurons1D(dataVector* v, somNeuron  *weights, somConfig* config)
{
    somNeuron* winner = v->lastWinner;
    for(int i=0; i<winner->nc;i++)
    {
      updateNeuron1D(v, winner, winner->neighbours[i], config);      
    };
}

//Update winner neuron and neighbours for 2D map return 1 if at least one neuron was updated 0 otherwise
void updateNeurons2D(dataVector* v, somNeuron  **weights, somConfig* config)
{
    somNeuron* winner = v->lastWinner;
    for(int i=0; i<winner->nc;i++)
    {
      updateNeuron2D(v, winner, winner->neighbours[i], config);      
    };
}

//Update winner neuron and neighbours for 2D map return 1 if at least one neuron was updated 0 otherwise
void updateNeurons3D(dataVector* v, somNeuron  ***weights, somConfig* config)
{
    somNeuron* winner = v->lastWinner;
    for(int i=0; i<winner->nc;i++)
    {
      updateNeuron3D(v, winner, winner->neighbours[i], config);      
    };
}
#pragma endregion

void updateWinner(somNeuron* winner, int vectorIndex)
{
    winner->entries[winner->ec++]=vectorIndex;
    winner->entries=realloc(winner->entries, (winner->ec+1)*sizeof(int));
}


void learn1D(int vectorIndex, dataVector* v, void* weights, somConfig* config)
{
    find_winner_fromNeighbours(v,weights,config, getNeighbours1D, getRandomNeuron1D);
    updateWinner(v->lastWinner, vectorIndex);
    updateNeurons1D(v, (somNeuron*)weights, config);
}

void learn2D(int vectorIndex, dataVector* v, void* weights, somConfig* config)
{
    find_winner_fromNeighbours(v,weights,config, getNeighbours2D, getRandomNeuron2D);
    updateWinner(v->lastWinner, vectorIndex);
    updateNeurons2D(v, (somNeuron**)weights, config);
}

void learn3D(int vectorIndex, dataVector* v, void* weights, somConfig* config)
{
   find_winner_fromNeighbours(v,weights,config, getNeighbours3D,  getRandomNeuron3D);
   updateWinner(v->lastWinner, vectorIndex);
   updateNeurons3D(v, (somNeuron***)weights, config);
}
#pragma endregion

#pragma region Clear Section
void clear_score1D(void* score, somConfig* config)
{
    somScore* sc = score ?  (somScore*)score : NULL;
    for(int i = 0; i < config->map_c; i++)
    {
        free(sc[i].scores);
        free(sc[i].entries);
    }
    free(sc);
}

void clear_score2D(void* score, somConfig* config)
{
    somScore** sc = (somScore**)score;
    for(int i = 0; i < config->map_r; i++)
    {
        clear_score1D(sc[i], config);
    }
    free(sc);
}

void clear_score3D(void* score, somConfig* config)
{
    somScore*** sc = (somScore***)score;
    for(int i = 0; i < config->map_b; i++)
    {
        clear_score2D(sc[i], config);
    }
    free(sc);
}

void clear_score(somScoreResult* score, somConfig* config)
{
    void (*clearfp)(void*, somConfig*);
    switch(config->dimension)
    {
        case oneD: clearfp = clear_score1D; break;
        case threeD: clearfp = clear_score3D; break;
        default: clearfp = clear_score2D; break;
    }
    clearfp(score->scores, config);
    free(score);   
}

void clear_mem1D(void* weights, void* score, somConfig* config)
{
    somScore* sc = score ?  (somScore*)score : NULL;
    somNeuron* som = (somNeuron*)weights;
    for(int i = 0; i < config->map_c; i++)
    {
        free(som[i].v);
        if(som[i].neighbours)
        {
            free(som[i].neighbours);
        }
        free(som[i].updates);
        free(som[i].entries);
        if(sc)
        {
            free(sc[i].scores);
        }
    }
    free(som);
    if(sc)
    {
        free(sc);
    }

}

void clear_mem2D(void* weights, void* score, somConfig* config)
{
    somScore** sc = (somScore**)score;
    somNeuron** som = (somNeuron**)weights;
    for(int i = 0; i < config->map_r; i++)
    {
        clear_mem1D(som[i], sc? sc[i]:NULL, config);
    }
    free(som);
    if(sc)
    {
        free(sc);
    }
}

void clear_mem3D(void* weights, void* score, somConfig* config)
{
    somScore*** sc = (somScore***)score;
    somNeuron** som = (somNeuron**)weights;
    for(int i = 0; i < config->map_b; i++)
    {
        clear_mem2D(som[i], sc? sc[i] : NULL, config);
    }
    free(som);
    if(sc)
    {
        free(sc);
    }
}

void clear_config(somConfig* config)
{
    free(config);
}

void clear_mem(void* weights, somScoreResult* score, somConfig* config)
{
    void (*clearfp)(void*, void*, somConfig*);
    switch(config->dimension)
    {
        case oneD: clearfp = clear_mem1D; break;
        case threeD: clearfp = clear_mem3D; break;
        default: clearfp = clear_mem2D; break;
    }
    clearfp(weights, score ? score->scores : NULL, config);
    if(score)
    {
        free(score);
    }
    
}

#pragma endregion
short getIsStabilized(double* updates, somConfig* config)
{
   for(int i=0;i<config->p;i++)
   {
       if(absd(updates[i]) > config->stabilizationTrigger)
       {
           return 0;
       }
   }
   return 1;
}

int indexOf(int array[], int count, int value)
{
    for(int i = 0;i<count;i++)
    {
        if(array[i]==value)
        {
            return i;
        }
    }
    return -1;
}

void updateStabilizedVectors(int* stabilizedVectors, somConfig* config, somNeuron*n)
{
    int index = config->n-1;
    for(int j=0;j<n->ec && index>=0;j++)
    {

        int e = n->entries[j];
        int ep = stabilizedVectors[e];
        int pos = ep == e ? e : indexOf(stabilizedVectors, index+1, e);
        if(pos>=0)
        {
            stabilizedVectors[pos]=stabilizedVectors[index];
            stabilizedVectors[index]=e;
            index--;
        }
    }
    config->n=index+1;
}

int updateStabilized1D(void* weights, somConfig* config, int* stabilizedVectors)
{
    int nNewStabilized = 0;
    somNeuron* som = (somNeuron*)weights;
    for(int i=0;i<config->map_c;i++)
    {
        somNeuron* n = &som[i];
        if(!n->isStabilized)
        {
            n->isStabilized = getIsStabilized(n->updates, config);
            n->updates = (double*)calloc(config->p, sizeof(double));
            if(n->isStabilized)
            {
                nNewStabilized++;
                updateStabilizedVectors(stabilizedVectors, config, n);
            }
        }
    }
    return nNewStabilized;
}

int updateStabilized2D(void* weights, somConfig* config, int* stabilizedVectors)
{
    int nNewStabilized = 0;
    somNeuron** som = (somNeuron**)weights;
    for(int i=0;i<config->map_r;i++)
    {
       nNewStabilized += updateStabilized1D(som[i], config, stabilizedVectors);
    }
    return nNewStabilized;
}

int updateStabilized3D(void* weights, somConfig* config, int* stabilizedVectors)
{
    int nNewStabilized = 0;
    somNeuron*** som = (somNeuron***)weights;
    for(int i=0;i<config->map_b;i++)
    {
        nNewStabilized += updateStabilized2D(som[i], config, stabilizedVectors);
    }
    return nNewStabilized;
}

//Get stabilized som neurons that has been train using provided data and config
void* getsom(dataVector* data, somConfig *config, dataBoundary* boundaries)
{
    
    if(!config->nw){
        config->nw = floor(5 *sqrt(config->n *1.0));
        config->nw -= config->nw%12;
    }
   
    printf("Calculating  %dD SOM for %d entries and %d parameters :", config->dimension, config->n, config->p);
    void* (*initfp)(dataVector*, somConfig*, dataBoundary*);
    void (*learnfp)(int, dataVector*, void*, somConfig*);
    void (*clearnbfp)(void*, somConfig*);
    int (*updatestfp)(void*, somConfig*, int*);
#ifdef TRACE_SOM
    void (*clearscorefp)(void*, somConfig*);
#endif
    switch (config->dimension){
        case oneD:
         initfp = getsom1D;
         learnfp = learn1D;
         clearnbfp = clear_neighbours1D;
         updatestfp = updateStabilized1D;
#ifdef TRACE_SOM
         clearscorefp = clear_score1D;
#endif
         break;
        case threeD:
         initfp = getsom3D;
         learnfp = learn3D;
         clearnbfp = clear_neighbours3D;
         updatestfp = updateStabilized3D;
#ifdef TRACE_SOM
         clearscorefp = clear_score3D;
#endif
         break;
        default:
        config->dimension = twoD;
         initfp = getsom2D;
         learnfp = learn2D;
         clearnbfp = clear_neighbours2D;
         updatestfp = updateStabilized2D;
#ifdef TRACE_SOM
         clearscorefp = clear_score2D;
#endif
    }
    void* weights = initfp(data, config, boundaries);
    if(config->nw == 0)
    {
        printf("Aborted, need at least one neuron\n");
        return weights;
    }
    somConfig cfg = *config;
    int stabilizedVectors[cfg.n];
    int vectorsToPropose[cfg.n];
    for(int i=0;i<cfg.n;i++){
        data[i].lastWinner = NULL;
        vectorsToPropose[i]=i;
        stabilizedVectors[i]=i;
    }
    int episode = 0;
    int again = 1;
    int nStabilized = 0;
#ifdef TRACE_SOM
    long stepId = 0;
    long time = 0;
    write(weights, config);
#endif
    while(again)
    {
        for(int i=cfg.n-1;i>=0;i--)
        {
            int ivector = ((double)rand()/RAND_MAX)*i;
            int proposed = vectorsToPropose[ivector];
            learnfp(proposed,  &data[proposed], weights, &cfg);
            vectorsToPropose[ivector] = vectorsToPropose[i];
#ifdef TRACE_SOM
                if(time++%TRACE_SOM == 0)
                {
                    somScoreResult* result = getscore(data, weights, config);
                    writeAppend(stepId++, weights, config, result);
                    clearscorefp(result->scores, config);
                    free(result);
                }
#endif      
        }
        int n = cfg.n;
        int nNewStabilized = updatestfp(weights, &cfg, stabilizedVectors);
        for(int i=cfg.n-1;i>=0;i--)
        {
            vectorsToPropose[i]=stabilizedVectors[i];
        }
        cfg.alpha*= cfg.alphaDecreaseRate;
        cfg.sigma*= cfg.sigmaDecreaseRate;
        short flagCleared = 0;
        if(episode>0 && episode%cfg.radiusDecreaseRate == 0){  
            if(cfg.radius > 1)
            {
                cfg.radius--;
                clearnbfp(weights, &cfg);
                flagCleared = 1;
            }         
        }
        if(nNewStabilized > 0 && !flagCleared)
        {
            clearnbfp(weights, &cfg);
        }
        nStabilized += nNewStabilized;
        episode++;
        again = nStabilized != config->nw && episode <cfg.maxEpisodes;
    }
#ifdef TRACE_SOM
    somScoreResult* result = getscore(data, weights, config);
    writeAppend(stepId++, weights, config, result);
    clearscorefp(result->scores, config);
    free(result);
#endif   
    if(nStabilized == config->nw)
    {
        printf("Stabilized after %d episodes\n", episode);
    }
    else
    {
        printf("Stopped unstabilized after %d ", episode);
    }
    
    return weights;
}

#pragma region Scoring Section
int getClassesCount(dataVector* data, int n)
{
    int result = 0;
    for(int i=0; i<n;i++)
    {
        int c = data[i].class;
        if(c>=result)
        {
            result = c+1;
        }
    }
    return result;
}

int argMax(int* values, int n, int skipIndex)
{
    int result=-1;
    int maxValue=0;
    for(int i=0;i<n;i++)
    {
        if(i==skipIndex)
        {
            continue;
        }
        if(values[i]>maxValue)
        {
            result = i;
            maxValue=values[i];
        }
    }
    return result;
}

short hasMultipleResult(int* values, int n)
{
    int count =0;
    for(int i=0;i<n;i++)
    {
        if(values[i]>0)
        {
            count++;
        }
    }
    return count ==0 ? -1: count >1;   
}

void initScore(somScore* s, int nClasses)
{
    s->scores = malloc(sizeof(int)*nClasses);
    s->entries = malloc(sizeof(int));
    s->totalEntries = 0;
    for(int i=0;i<nClasses;i++)
    {
        s->scores[i]=0;
    }
}

void updateScore(somScore* s, int class, int entry)
{
    s->scores[class]++;
    s->entries[s->totalEntries]=entry;
    s->totalEntries++;
    s->entries = realloc(s->entries, (s->totalEntries+1)*sizeof(int));
}

void updateScoreStats(somScore* s, int nClasses, somScoreResult* scoreResult)
{
    s->maxClass = s->secondClass = -1;
    s->status = hasMultipleResult(s->scores, nClasses);
    if(s->status >=0)
    {
        scoreResult->nActivatedNodes++;
        s->maxClass = argMax(s->scores, nClasses, -1);
        if(s->status >0)
        {
            s->secondClass = argMax(s->scores, nClasses, s->maxClass);
        }
        
    }
}

void score1D(dataVector* data, void* weights, somConfig* config, somScoreResult* scoreResult)
{
    somScore* score = (somScore*)malloc(sizeof(somScore) * config->map_c);
    somNeuron* som = (somNeuron*)weights;
    scoreResult->nActivatedNodes=0;
    int nClasses = scoreResult->nClasses;
    for(int i=0;i<config->map_c;i++)
    {
        initScore(&score[i], nClasses);
    }
    for(int i=0;i<config->n;i++)
    {
        dataVector* v = &data[i];
        find_winner_fromNeighbours(v,weights,config, getNeighbours1D, getRandomNeuron1D);
        somNeuron* winner = v->lastWinner;
        updateScore(&score[winner->c],v->class, i);

    }
    for(int i=0;i<config->map_c;i++)
    {
        updateScoreStats(&score[i], nClasses, scoreResult);
    }
    scoreResult->scores = score;
}

void score2D(dataVector* data, void* weights, somConfig* config, somScoreResult* scoreResult)
{
    somScore** score = (somScore**)malloc(sizeof(somScore*) * config->map_r);
    somNeuron** som = (somNeuron**)weights;
    scoreResult->nActivatedNodes=0;
    int nClasses = scoreResult->nClasses;
    for(int i=0;i<config->map_r;i++)
    {
        score[i] = (somScore*)malloc(sizeof(somScore)* config->map_c);
        for(int j=0;j<config->map_c;j++)
        {
            initScore(&score[i][j], nClasses);
        }
    }
    for(int i=0;i<config->n;i++)
    {
        dataVector* v = &data[i];
        find_winner_fromNeighbours(v,weights,config, getNeighbours2D, getRandomNeuron2D);
        somNeuron* winner = v->lastWinner;
        updateScore(&score[winner->r][winner->c],v->class, i);
    }
    for(int i=0;i<config->map_r;i++)
    {
        for(int j=0;j<config->map_c;j++)
        {
            updateScoreStats(&score[i][j], nClasses, scoreResult);
        }
        
    }
    scoreResult->scores = score;
}

void score3D(dataVector* data, void* weights, somConfig* config, somScoreResult* scoreResult)
{
    somScore*** score = (somScore***)malloc(sizeof(somScore**) * config->map_b);
    somNeuron*** som = (somNeuron***)weights;
    scoreResult->nActivatedNodes=0;
    int nClasses = scoreResult->nClasses;
    for(int i=0;i<config->map_b;i++)
    {
        score[i] = (somScore**)malloc(sizeof(somScore*)* config->map_r);
        for(int j=0;j<config->map_r;j++)
        {
            score[i][j] = malloc(sizeof(somScore)*config->map_c);
            for(int k=0;k<config->map_c;k++)
            {
                initScore(&score[i][j][k], nClasses);
            }
        }
    }
    for(int i=0;i<config->n;i++)
    {
        dataVector* v = &data[i];
        find_winner_fromNeighbours(v,weights,config, getNeighbours3D, getRandomNeuron3D);
        somNeuron* winner = v->lastWinner;
        updateScore(&score[winner->b][winner->r][winner->c],v->class, i);
    }
    for(int i=0;i<config->map_b;i++)
    {
        for(int j=0;j<config->map_r;j++)
        {

            for(int k=0;k<config->map_c;k++)
            {
                updateScoreStats(&score[i][j][k],nClasses, scoreResult);
            }
            
        }
        
    }
    scoreResult->scores = score;
}

somScoreResult* getscore(dataVector* data, void* weights, somConfig* config)
{
    somScoreResult* result = malloc(sizeof(somScoreResult));
    void (*scorefp)(dataVector*,  void* , somConfig*, somScoreResult*);
    switch (config->dimension)
    {
        case oneD: scorefp = score1D; break;
        case threeD: scorefp = score3D; break;
        default: scorefp = score2D; break;
    }
    result->nClasses = getClassesCount(data, config->n);
    scorefp(data, weights, config, result);
    return result;
}
#pragma endregion

#pragma region  Display Section
int getTerminalColorCode(int index){
    if(index>0)
    {
        int code = index%7;
        if(code>0)
        {
            switch(code)
            {
                //red
                case 1: return 31;
                //yellow
                case 2: return 33;
                //blue
                case 3: return 34;
                //green
                case 4: return 32;
                //magenta
                case 5: return 35;
                //cyan
                case 6: return 36;
            }
        }
    }
    //default
    return 39;
}

int getTerminalBGColorCode(int index){
    if(index>0)
    {
        int code = index%7;
        if(code>0)
        {
            switch(code)
            {
                //red
                case 1: return 41;
                //yellow
                case 2: return 43;
                //blue
                case 3: return 44;
                //green
                case 4: return 42;
                //magenta
                case 5: return 45;
                //cyan
                case 6: return 46;
            }
        }
    }
    //default
    return 49;
}

void displayConfig(somConfig* config)
{
    printf("SOM Settings:\n%21s: %d\n%21s: %d\n%21s: %d\n%21s: %d\n%21s: %d\n%21s: %.2f\n%21s: %.2f\n%s: %.f%%\n", "Entries", config->n,
                        "Parameters", config->p, "Dimension", config->dimension, "Neurons", config->nw , "Radius", config->radius,
                        "Learning rate", config->alpha, "Neighborhood factor", config->sigma,
                        "Neighborhood coverage", config->initialPercentCoverage *100);
    if(config->dimension == threeD)
    {
        printf("%21s: %d\n", "Map blocks", config->map_b);  
    }
    if(config->dimension != oneD)
    {
        printf("%21s: %d\n","Map rows", config->map_r);
    }                                      
    
    printf("%21s: %d\n\n", "Map columns", config->map_c); 
}

void displayScoreAtom(somScore* score)
{
    somScore s =*score;
    int c = s.maxClass + 1;
    if(s.status<1)
    {
        printf("\033[0;%dm", getTerminalColorCode(c));
    }
    else
    {
        printf("\033[0;%d;%dm", getTerminalColorCode(c), getTerminalBGColorCode(s.secondClass+1));
    }

    printf("%d", c);
    printf("\033[0m ");
}

void displayScore1D(void* scores, somConfig* config)
{
    somScore* score = (somScore*)scores;
    for(int i=0;i<config->map_c;i++)
    {
        displayScoreAtom(&score[i]);
    }
}

void displayScore2D(void* scores, somConfig* config)
{
    somScore** score = (somScore**)scores;
    for(int i=0;i<config->map_r;i++)
    {
        displayScore1D(score[i], config);
        printf("\n");
    }
}

void displayScore3D(void* scores, somConfig* config)
{
    somScore*** score = (somScore***)scores;
    for(int i=0;i<config->map_b;i++)
    {
        printf("Block %d\n", i);
        displayScore2D(score[i], config);
    }
}

void displayScore(somScoreResult* scoreResult, somConfig* config)
{
    void (*displayScorefp)(void*, somConfig*);
    switch(config->dimension)
    {
        case oneD: displayScorefp = displayScore1D;break;
        case threeD: displayScorefp = displayScore3D;break;
        default: displayScorefp = displayScore2D;break;
    }
    printf("SOM Map:\n");
    displayScorefp(scoreResult->scores, config);
    printf("\nActivated nodes:%d\n\n", scoreResult->nActivatedNodes);
}
#pragma endregion

#pragma region I/O Section
void writeNeuron(FILE* fp, somNeuron* n, somScore* score, long stepId, int p)
{
    fputs("[", fp);
    int x = n->c;
    int y = n->r;
    int z = n->b;
    int s = -1;
    int class = -1;
    int class2 = -1;
    short stabilized = n->isStabilized;
    if(score)
    {
        s = score->totalEntries;
        class = score->maxClass;
        class2 = score->secondClass;
    }
    for(int i=0;i<p;i++){
        fprintf(fp, "%f", n->v[i]);
        if(i<p-1)
        {
            fputs(",", fp);
        }
    }
    fprintf(fp, "];%d;%d;%d;%ld;%d;%d;%d;%d", x,y,z , stepId, s, class,  class2, stabilized);
    fputs(";[", fp);
    int limit = s-1;
    for(int i=0;i<s;i++)
    {
       fprintf(fp, "%d", score->entries[i]);
        if(i<limit)
        {
            fputs(",", fp);
        } 
    }
    fputs("]\n", fp);
}

void writeNeurons1D(FILE* fp, void* weights, void* scores, long stepId, somConfig* config)
{
    somNeuron* som = (somNeuron*)weights;
    somScore* s = (somScore*)scores;
    for(int i=0;i<config->map_c;i++)
    {
        writeNeuron(fp, &som[i], s? &s[i] : NULL, stepId, config->p);
    }
}

void writeNeurons2D(FILE* fp, void* weights, void* scores, long stepId, somConfig* config)
{
    somNeuron** som = (somNeuron**)weights;
    somScore** s = (somScore**)scores;
    for(int i=0;i<config->map_r;i++)
    {
        writeNeurons1D(fp, som[i], s? s[i] : NULL, stepId, config);
    }  
}

void writeNeurons3D(FILE* fp, void* weights, void* scores, long stepId, somConfig* config)
{
    somNeuron*** som = (somNeuron***)weights;
    somScore*** s = (somScore***)scores;
    for(int i=0;i<config->map_b;i++)
    {
        writeNeurons2D(fp, som[i], s? s[i] : NULL, stepId, config);
    }      
}

void wirteNeurons(FILE *fp, void *weights, somConfig* config, long stepid, somScoreResult* scoreResult)
{
    void (*writefp)(FILE*,void*, void*, long, somConfig*);
    switch(config->dimension){
        case oneD: writefp = writeNeurons1D;break;
        case threeD: writefp = writeNeurons3D;break;
        default: writefp = writeNeurons2D;break;
    }
    writefp(fp, weights, scoreResult ? scoreResult->scores : NULL, stepid,config);
}

char* getsomFileName(mapDimension dimension)
{
    switch (dimension)
    {
        case oneD: return "som1D.data";
        case twoD: return "som2D.data";
        case threeD: return "som3D.data";
    }
}

void writeAppend(long stepid, somNeuron *weights, somConfig* config, somScoreResult* scoreResult)
{
    FILE * fp;
    fp = fopen(getsomFileName(config->dimension), "a");
    if(fp != NULL)
    {
        wirteNeurons(fp, weights, config, stepid, scoreResult);
    }
    fclose(fp);
}

void write(somNeuron* weights, somConfig* config){
    FILE * fp;
    fp = fopen(getsomFileName(config->dimension), "w");
    if(fp != NULL)
    {
        wirteNeurons(fp, weights, config, -1, NULL);
    }
    fclose(fp);
}
#pragma endregion



