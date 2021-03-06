#include "include/som.h"
#include "include/common.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

//Calculate the euclidian distance between 2 vectors using p dimensions
double distance_function(double *v, double *w, int p)
{
    double sum=0;
    for(int i=0;i<p;i++){
        sum+= pow((v[i] - w[i]),2);
    }
    return sqrt(sum);
}

//Return the index of the neuron closest to a entry vector using the distance_function for 1D map
somNeuron* find_winner1D(dataVector* v, void *weights, int nw, int p)
{
    somNeuron* som = (somNeuron*)weights;
    somNeuron **results = malloc(sizeof(somNeuron*));
    int count = 1;
    double minValue = __DBL_MAX__;
    for(int i = 0; i<nw; i++){
        double distance = distance_function(v->v, som[i].v, p);
        if(distance<minValue){
            results[count-1]= &som[i];
            minValue = distance;
        }
        else if(distance == minValue){
            results = realloc(results, ++count*sizeof(int));
            results[count-1]= &som[i];
        }
    }
    int selectedIndex = count > 1 ? rand()%count : 0;
    somNeuron* result = results[selectedIndex];
    free(results);
    return result;
}

//Return the index of the neuron closest to a entry vector using the distance_function for 2D map
somNeuron* find_winner2D(dataVector* v, somNeuron** weights, somConfig *config)
{
    somNeuron **results = malloc(sizeof(somNeuron*));
    int count = 1;
    double minValue = __DBL_MAX__;
    for(int i = 0; i<config->map_r; i++)
    {
        for(int j=0;j<config->map_c;j++)
        {
            double distance = distance_function(v->v, weights[i][j].v, config->p);
            if(distance<minValue){
                results[count-1]= &weights[i][j];
                minValue = distance;
            }
            else if(distance == minValue){
                results = realloc(results, ++count*sizeof(int));
                results[count-1]= &weights[i][j];
            }
        }

    }
    int selectedIndex = count > 1 ? rand()%count : 0;
    somNeuron* result = results[selectedIndex];
    free(results);
    return result;
}

//Return the index of the neuron closest to a entry vector using the distance_function for 3D map
somNeuron* find_winner3D(dataVector* v, void *weights, somConfig *config)
{
    somNeuron*** som = (somNeuron***)weights;
    somNeuron **results = malloc(sizeof(somNeuron*));
    int count = 1;
    double minValue = __DBL_MAX__;
    for(int i = 0; i<config->map_b; i++)
    {
        for(int j=0;j<config->map_r;j++)
        {
            for(int k=0;k<config->map_c;k++)
            {
                double distance = distance_function(v->v, som[i][j][k].v, config->p);
                if(distance<minValue){
                    results[count-1]= &som[i][j][k];
                    minValue = distance;
                }
                else if(distance == minValue){
                    results = realloc(results, ++count*sizeof(int));
                    results[count-1]= &som[i][j][k];
                }
            }
        }

    }
    int selectedIndex = count > 1 ? rand()%count : 0;
    somNeuron* result = results[selectedIndex];
    free(results);
    return result;
}

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

short updateNeuron(dataVector* v, somNeuron* n, double h, somConfig* config)
{
    short stabilized = 1;
    for(int i=0;i<config->p;i++){
        double delta = config->alpha * h *(v->v[i] - n->v[i]);
        n->v[i] += delta;
        double d = absd(delta);
        stabilized = (d < config->stabilizationTrigger) & stabilized;
    }
    return stabilized;
}

//Update the 1D weight vector of a neuron adding for each parameter the difference between entry vector parameter and current vector parameter
//The difference is multiplied by a learning factor (epsilon) and a the result (value between 0 and 1) of neighborhood function
//If no parameter has been updated significantly return 0, 1 otherwise
short updateNeuron1D(dataVector* v, somNeuron* winner, somNeuron* n, somConfig* config)
{   
    double h = neighborhood_function1d(winner, n, config->sigma);
    return updateNeuron(v, n, h, config);
}

//Update the 2D weight vector of a neuron adding for each parameter the difference between entry vector parameter and current vector parameter
//The difference is multiplied by a learning factor (epsilon) and a the result (value between 0 and 1) of neighborhood function
//If no parameter has been updated significantly return 0, 1 otherwise
short updateNeuron2D(dataVector* v, somNeuron* winner, somNeuron* n, somConfig* config){
    double h = neighborhood_function2d(winner, n, config->sigma);
    return updateNeuron(v, n, h, config);
}

//Update the 3D weight vector of a neuron adding for each parameter the difference between entry vector parameter and current vector parameter
//The difference is multiplied by a learning factor (epsilon) and a the result (value between 0 and 1) of neighborhood function
//If no parameter has been updated significantly return 0, 1 otherwise
short updateNeuron3D(dataVector* v, somNeuron* winner, somNeuron* n, somConfig* config){
    double h = neighborhood_function3d(winner, n, config->sigma);
    return updateNeuron(v, n, h, config);
}

void getNeighbours1D(somNeuron* n, somNeuron* weights, somConfig* config)
{
    //TODO: Add cosed map handling
    int start = max(0, n->c - config->radius);
    int end = min(config->map_c, n->c + config->radius);
    n->nc = end - start;
    int in=0;
    n->neighbours = (somNeuron**)malloc((n->nc)*sizeof(somNeuron*));
    for(int i=start;i<end;i++)
    {
        n->neighbours[in]= &weights[i];
        in++;
    }
}

void getNeighbours2D(somNeuron* n, somNeuron** weights, somConfig* config)
{
    //TODO: Add cosed map handling
    int start_r = max(0, n->r - config->radius);
    int end_r = min(config->map_r, n->r + config->radius);
    int start_c = max(0, n->c - config->radius);
    int end_c = min(config->map_c, n->c + config->radius);
    int rc = (end_r - start_r);
    int cc = (end_c - start_c);
    n->nc = rc * cc;
    int in=0;
    n->neighbours = (somNeuron**)malloc((n->nc)*sizeof(somNeuron*));
    for(int i=start_r;i<end_r;i++)
    {
        for(int j=start_c;j<end_c;j++)
        {
            n->neighbours[in]= &weights[i][j];
            in++;
        }
    }
}

void getNeighbours3D(somNeuron* n, somNeuron*** weights, somConfig* config)
{
    //TODO: Add cosed map handling
    int start_r = max(0, n->r - config->radius);
    int end_r = min(config->map_r, n->r + config->radius);
    int start_c = max(0, n->c - config->radius);
    int end_c = min(config->map_c, n->c + config->radius);
    int start_b = max(0, n->b - config->radius);
    int end_b = min(config->map_b, n->c + config->radius);
    int rc = end_r - start_r;
    int cc = end_c - start_c;
    int bc = end_b - start_b;
    n->nc = rc * cc;
    int in=0;
    n->neighbours = (somNeuron**)malloc((n->nc)*sizeof(somNeuron*));
    for(int i=start_b;i<end_b;i++)
    {
        for(int j=start_r;j<end_r;j++)
        {
            for(int k=start_c;k<end_c;k++)
            {
                n->neighbours[in]= &weights[i][j][k];
                in++;
            }

        }
    }
}

//Update winner neuron and neighbours for 1D map return 1 if at least one neuron was updated 0 otherwise
short updateNeurons1D(dataVector* v, somNeuron* winner, somNeuron  *weights, somConfig* config)
{
    short stabilized = 1;
    if(!winner->neighbours)
    {
        getNeighbours1D(winner, weights, config);
    }
    for(int i=0; i<winner->nc;i++)
    {
      stabilized = updateNeuron1D(v, winner, winner->neighbours[i], config) & stabilized;         
    };
    return stabilized;
}

//Update winner neuron and neighbours for 2D map return 1 if at least one neuron was updated 0 otherwise
short updateNeurons2D(dataVector* v, somNeuron* winner, somNeuron  **weights, somConfig* config)
{
    short stabilized = 1;
    if(!winner->neighbours)
    {
        getNeighbours2D(winner, weights, config);
    }
    for(int i=0; i<winner->nc;i++)
    {
      stabilized = updateNeuron2D(v, winner, winner->neighbours[i], config) & stabilized;         
    };
    return stabilized;
}

//Update winner neuron and neighbours for 2D map return 1 if at least one neuron was updated 0 otherwise
short updateNeurons3D(dataVector* v, somNeuron* winner, somNeuron  ***weights, somConfig* config)
{
    short stabilized = 1;
    if(!winner->neighbours)
    {
        getNeighbours3D(winner, weights, config);
    }
    for(int i=0; i<winner->nc;i++)
    {
      stabilized = updateNeuron3D(v, winner, winner->neighbours[i], config) & stabilized;         
    };
    return stabilized;
}

//Return a random value between a boundary
double getRandom(dataBoundary boundary){
    return (((double)rand()/RAND_MAX)*(boundary.max - boundary.min)) + boundary.min;
}
//Initialize SOM 1D weights with random values based on parameters boundaries
void initialize1D(somNeuron *weights, somConfig config, dataBoundary *boundaries){
    for(int i=0;i<config.nw; i++){
        weights[i].v = (double*)malloc(config.p * sizeof(double));
        weights[i].c = i;
        weights[i].neighbours = NULL;
        weights[i].nc= 0;
        for(int j=0;j<config.p;j++){
            weights[i].v[j]= getRandom(boundaries[j]);
        }
    }
}

//Initialize SOM 2D weights with random values based on parameters boundaries
void initialize2D(somNeuron** weights, somConfig config, dataBoundary *boundaries){
    for(int i=0;i<config.map_r; i++)
    {
        weights[i] = (somNeuron*)malloc(config.map_c * sizeof(somNeuron));
        for(int j=0;j<config.map_c;j++)
        {
            weights[i][j].v = (double*)malloc(config.p * sizeof(double));
            weights[i][j].r = i;
            weights[i][j].c = j;
            weights[i][j].neighbours = NULL;
            weights[i][j].nc= 0;
            for(int k=0;k<config.p;k++)
            {
                weights[i][j].v[k]= getRandom(boundaries[k]);
            }
        }
    }
}

//Initialize SOM 3D weights with random values based on parameters boundaries
void initialize3D(somNeuron*** weights, somConfig config, dataBoundary *boundaries){
    for(int i=0;i<config.map_b; i++)
    {
        weights[i] = (somNeuron**)malloc(config.map_r * sizeof(somNeuron*));
        for(int j=0;i<config.map_r;j++)
        {
            weights[i][j] = (somNeuron*)malloc(config.map_c * sizeof(somNeuron));
            for(int k=0;k<config.map_c;k++){
                weights[i][j][k].v = (double*)malloc(config.p * sizeof(double));
                weights[i][j][k].b = i;
                weights[i][j][k].r = j;
                weights[i][j][k].c = k;
                weights[i][j][k].neighbours = NULL;
                weights[i][j][k].nc = 0;
                for(int l=0;l<config.p;l++)
                {
                    weights[i][j][k].v[l]= getRandom(boundaries[l]);
                }
            }
        }
    }
}


//Get min and max for each parameters of the data set
void initializeBoundaries(dataBoundary *boundaries, dataVector *data, somConfig config){
    for(int i=0; i<config.p; i++){
        dataBoundary b = {__DBL_MAX__, __DBL_MIN__};
        boundaries[i]= b;
    }
    for(int i = 0; i<config.n; i++){
        for(int j =0; j<config.p; j++){
            boundaries[j].min = min(data[i].v[j], boundaries[j].min);
            boundaries[j].max = max(data[i].v[j], boundaries[j].max);
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
    //Arbitrary 4/3 ratio (16/9 didn't exist in the eighties ;))
    double ratio = 4.0/3;
    int y = floor(sqrt(config->nw / ratio));
    y-=config->nw%y;
    int x = config->nw/y;
    if(config->nw%(x*y)>0){
        config->nw = x*y;
    }
    config->map_r=y;
    config->map_c=x;
    if(!config->radius){
        config->radius = ceil((sqrt(config->initialPercentCoverage*config->nw) - 1)/2);
    }
}

void set3DMapSize(somConfig *config)
{
    int size = floor(cbrt(config->nw));
    if(config->nw%((int)pow(size,3))>0){
        config->nw = (int)pow(size, 3);
    }
    config->map_r=size;
    config->map_c=size;
    config->map_b=size;
    if(!config->radius){
        config->radius = ceil((cbrt(config->initialPercentCoverage*config->nw)-1)/2);
    }
}

void* getsom1D(dataVector* data, somConfig *config, dataBoundary* boundaries)
{
    somNeuron *weights = (somNeuron*)malloc(config->nw *sizeof(somNeuron));
    initialize1D(weights, *config, boundaries);
    return weights;
}

void* getsom2D(dataVector* data, somConfig *config, dataBoundary* boundaries)
{
    if(!config->map_r|| !config->map_c){
        set2DMapSize(config);
    }
    somNeuron **weights = (somNeuron**)malloc(config->map_r * sizeof(somNeuron*));
    initialize2D(weights, *config, boundaries);
    return weights;
}

void* getsom3D(dataVector* data, somConfig *config, dataBoundary* boundaries)
{
    if(!config->map_b || !config->map_r|| !config->map_c){
        set3DMapSize(config);
    }
    somNeuron ***weights = (somNeuron***)malloc(config->map_b * sizeof(somNeuron**));
    initialize3D(weights, *config, boundaries);
    return weights;
}

somConfig* getsomDefaultConfig(){
    somConfig* config = malloc(sizeof(somConfig));
    config->stabilizationTrigger = 0.001;
    config->isMapClosed = 0;
    config->dimension = twoD;
    config->alpha = 0.99;
    config->sigma = 0.99;
    config->alphaDecreaseRate=0.99;
    config->sigmaDecreaseRate=0.90;
    config->radiusDecreaseRate = 3;
    config->initialPercentCoverage = 0.6;
    config->maxEpisodes = 1000;
}

short learn1D(dataVector* v, void* weights, somConfig* config)
{
    somNeuron* winner = find_winner1D(v, weights, config->nw, config->p);
    return updateNeurons1D(v, winner, (somNeuron*)weights, config);
}

short learn2D(dataVector* v, void* weights, somConfig* config)
{
    somNeuron* winner = find_winner2D(v, weights, config);
    return updateNeurons2D(v, winner, (somNeuron**)weights, config);
}

short learn3D(dataVector* v, void* weights, somConfig* config)
{
   somNeuron* winner = find_winner3D(v, weights, config);
   return updateNeurons3D(v, winner, (somNeuron***)weights, config);
}

void clear_neighbours1D(void* weights, somConfig* config)
{
    somNeuron* som = (somNeuron*)weights;
    for(int i=0; i< config->map_c; i++)
    {
        free(som[i].neighbours);
        som[i].neighbours = NULL;
        som[i].nc = 0;
    }
}

void clear_neighbours2D(void* weights, somConfig* config)
{
    somNeuron** som = (somNeuron**)weights;
    for(int i=0; i< config->map_r; i++)
    {
        for(int j=0; j< config->map_c; j++)
        {
            free(som[i][j].neighbours);
            som[i][j].neighbours = NULL;
            som[i][j].nc = 0;
        }
    }
}

void clear_neighbours3D(void* weights, somConfig* config)
{
    somNeuron*** som = (somNeuron***)weights;
    for(int i=0; i< config->map_b; i++)
    {
        for(int j=0; j< config->map_r; j++)
        {
            for(int k=0; k< config->map_c; k++)
            {
                free(som[i][j][k].neighbours);
                som[i][j][k].neighbours = NULL;
                som[i][j][k].nc = 0;
            }
        }
    }
}

//Get stabilized som neurons that has been train using provided data and config
void* getsom(dataVector* data, somConfig *config)
{
    
    if(!config->nw){
        config->nw = floor(5 *sqrt(config->n *1.0));
        config->nw -= config->nw%12;
    }
   
    dataBoundary boundaries[config->p];
    initializeBoundaries(boundaries, data, *config);
    void* (*initfp)(dataVector*, somConfig*, dataBoundary*);
    short (*learnfp)(dataVector*, void*, somConfig*);
    void (*clearnbfp)(void*, somConfig*);
    switch (config->dimension){
        case oneD:
         initfp = getsom1D;
         learnfp = learn1D;
         clearnbfp = clear_neighbours1D;
         break;
        case threeD:
         initfp = getsom3D;
         learnfp = learn3D;
         clearnbfp = clear_neighbours3D;
         break;
        default:
         initfp = getsom2D;
         learnfp = learn2D;
         clearnbfp = clear_neighbours2D;
    }
    void* weights = initfp(data, config, boundaries);
    somConfig cfg = *config;
    int vectorsToPropose[cfg.n];
    int stabilizedVectors[cfg.n];
    for(int i=0;i<cfg.n;i++){
        vectorsToPropose[i]=i;
        stabilizedVectors[i]=i;
    }
    int episode = 0;
    int tested[config->n];
    for(int i=0;i<config->n;i++){
        tested[i]=0;
    }
    while(cfg.n >0 && episode <cfg.maxEpisodes)
    {
        int n = cfg.n-1;
        for(int i=n;i>=0;i--)
        {
            int ivector = (((double)rand()/RAND_MAX)*(i));
            int proposed = vectorsToPropose[ivector];
            tested[proposed] = 1;
            short hasStabilized = learnfp(&data[proposed], weights, &cfg);
            vectorsToPropose[ivector] = vectorsToPropose[i];
            if(hasStabilized)
            {               
                stabilizedVectors[proposed] = n;
                stabilizedVectors[n] = proposed;
                n--;
            }
            vectorsToPropose[i] = stabilizedVectors[i];
        }
        cfg.n = n +1;
        cfg.alpha*= cfg.alphaDecreaseRate;
        cfg.sigma*= cfg.sigmaDecreaseRate;
        if(episode>0 && episode%cfg.radiusDecreaseRate == 0){  
            if(cfg.radius > 1)
            {
                cfg.radius--;
                clearnbfp(weights, &cfg);
            }         
        }
        episode++;
    }
    printf("Stopped after %d episodes\n", episode);
    return weights;
}

void clear_mem1D(void* weights, void* score, somConfig* config)
{
    somScore* sc = score ?  (somScore*)score : NULL;
    somNeuron* som = (somNeuron*)weights;
    for(int i = 0; i < config->map_c; i++)
    {
        free(som[i].v);
        free(som[i].neighbours);
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
}

void clear_mem(dataVector* data, void* weights, somScoreResult* score, somConfig* config)
{
    for(int i = 0; i<config->n;i++)
    {
        free(data[i].v);
    }
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
    free(config);
}

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

int argMax(int* values, int n)
{
    int result=-1;
    int maxValue=0;
    for(int i=0;i<n;i++)
    {
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
    return count ==0 ? -1: count <2;   
}

void* score1D(dataVector* data, void* weights, somConfig* config, int nclasses)
{
    somScore* score = (somScore*)malloc(sizeof(somScore) * config->map_c);
    somNeuron* som = (somNeuron*)weights;
    for(int i=0;i<config->map_c;i++)
    {
        score[i].scores = malloc(sizeof(int)*nclasses);
        for(int j=0;j<nclasses;j++)
        {
            score[i].scores[j]=0;
        }
    }
    for(int i=0;i<config->n;i++)
    {
        dataVector v = data[i];
        somNeuron* winner = find_winner1D(&v, weights, config->nw, config->p);
        score[winner->c].scores[v.class]++;
    }
    for(int i=0;i<config->map_c;i++)
    {
        score[i].iclass = argMax(score[i].scores, nclasses);
        score[i].hasMultipleResult = hasMultipleResult(score[i].scores, nclasses);
    }

    return score;
}

void* score2D(dataVector* data, void* weights, somConfig* config, int nclasses)
{
    somScore** score = (somScore**)malloc(sizeof(somScore*) * config->map_r);
    somNeuron** som = (somNeuron**)weights;
    for(int i=0;i<config->map_r;i++)
    {
        score[i] = (somScore*)malloc(sizeof(somScore)* config->map_c);
        for(int j=0;j<config->map_c;j++)
        {
            score[i][j].scores = malloc(sizeof(int)*nclasses);
            for(int k=0;k<nclasses;k++)
            {
                score[i][j].scores[k]=0;
            }
        }
    }
    for(int i=0;i<config->n;i++)
    {
        dataVector v = data[i];
        somNeuron* winner = find_winner2D(&v, weights, config);
        score[winner->r][winner->c].scores[v.class]++;
    }
    for(int i=0;i<config->map_r;i++)
    {
        for(int j=0;j<config->map_c;j++)
        {
            score[i][j].iclass = argMax(score[i][j].scores, nclasses);
            score[i][j].hasMultipleResult = hasMultipleResult(score[i][j].scores, nclasses);
        }
        
    }
    return score;
}

void* score3D(dataVector* data, void* weights, somConfig* config, int nclasses)
{
    somScore*** score = (somScore***)malloc(sizeof(somScore**) * config->map_b);
    somNeuron*** som = (somNeuron***)weights;
    for(int i=0;i<config->map_b;i++)
    {
        score[i] = (somScore**)malloc(sizeof(somScore*)* config->map_r);
        for(int j=0;j<config->map_r;j++)
        {
            score[i][j] = malloc(sizeof(somScore)*config->map_c);
            for(int k=0;k<config->map_c;k++)
            {
                score[i][j][k].scores= malloc(sizeof(int)*nclasses);
                for(int l=0;l<nclasses;l++)
                {
                    score[i][j][k].scores[l]=0;
                }
            }
        }
    }
    for(int i=0;i<config->n;i++)
    {
        dataVector v = data[i];
        somNeuron* winner = find_winner3D(&v, weights, config);
        score[winner->b][winner->r][winner->c].scores[v.class]++;
    }
    for(int i=0;i<config->map_b;i++)
    {
        for(int j=0;j<config->map_r;j++)
        {
            for(int k=0;k<config->map_c;k++)
            {
                score[i][j][k].iclass = argMax(score[i][j][k].scores, nclasses);
                score[i][j][k].hasMultipleResult = hasMultipleResult(score[i][j][k].scores, nclasses);
            }
            
        }
        
    }
    return score;
}

somScoreResult* getscore(dataVector* data, void* weights, somConfig* config)
{
    somScoreResult* result = malloc(sizeof(somScoreResult));
    void* (*scorefp)(dataVector*,  void* , somConfig*, int);
    switch (config->dimension)
    {
        case oneD: scorefp = score1D; break;
        case threeD: scorefp = score3D; break;
        default: scorefp = score2D; break;
    }
    result->nclasses = getClassesCount(data, config->n);
    result->scores = scorefp(data, weights, config, result->nclasses);
    return result;
}

void append(FILE *fp, somNeuron *weights, somConfig* config, long stepid, int scores[]){
    int p = config->p;
    int nw = config->nw;
    for(int i=0; i< nw; i++){
        fputs("[", fp);

        for(int j=0;j<p;j++){
            fprintf(fp, "%f", weights[i].v[j]);
            if(j<p-1)
            {
                fputs(",", fp);
            }
        }
        fprintf(fp, "];%d;%ld;%d\n", i, stepid, scores == NULL ? 0 :scores[i]);
    }
}

void writeAppend(long stepid, somNeuron *weights, somConfig* config, int scores[]){
    FILE * fp;
    fp = fopen("som.data", "a");
    if(fp != NULL){
        append(fp, weights, config, stepid, scores);
    }
    fclose(fp);
}

void write(somNeuron* weights, somConfig* config){
    FILE * fp;
    fp = fopen("som.data", "w");
    if(fp != NULL){
        append(fp, weights, config, -1, NULL);
    }
    fclose(fp);
}



