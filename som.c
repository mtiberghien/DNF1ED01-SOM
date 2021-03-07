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
    int end_b = min(config->map_b, n->b + config->radius);
    int rc = end_r - start_r;
    int cc = end_c - start_c;
    int bc = end_b - start_b;
    n->nc = rc * cc * bc;
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

void initNeuron(somNeuron*n, somConfig config, dataBoundary *boundaries, int b, int r, int c)
{
    n->b=b;
    n->r=r;
    n->c=c;
    n->v=(double*)malloc(config.p * sizeof(double));
    n->neighbours = NULL;
    n->nc = 0;
    for(int j=0;j<config.p;j++)
    {
        n->v[j]= getRandom(boundaries[j]);
    }
}

//Initialize SOM 1D weights with random values based on parameters boundaries
void initialize1D(somNeuron *weights, somConfig config, dataBoundary *boundaries){
    for(int i=0;i<config.nw; i++){
        initNeuron(&weights[i], config, boundaries,-1,-1,i);
    }
}

//Initialize SOM 2D weights with random values based on parameters boundaries
void initialize2D(somNeuron** weights, somConfig config, dataBoundary *boundaries){
    for(int i=0;i<config.map_r; i++)
    {
        weights[i] = (somNeuron*)malloc(config.map_c * sizeof(somNeuron));
        for(int j=0;j<config.map_c;j++)
        {
            initNeuron(&weights[i][j], config, boundaries,-1,i,j);
        }
    }
}

//Initialize SOM 3D weights with random values based on parameters boundaries
void initialize3D(somNeuron*** weights, somConfig config, dataBoundary *boundaries){
    for(int i=0;i<config.map_b; i++)
    {
        weights[i] = (somNeuron**)malloc(config.map_r * sizeof(somNeuron*));
        for(int j=0;j<config.map_r;j++)
        {
            weights[i][j] = (somNeuron*)malloc(config.map_c * sizeof(somNeuron));
            for(int k=0;k<config.map_c;k++){
               initNeuron(&weights[i][j][k], config, boundaries,i,j,k);
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
        if(config.normalize){
            data[i].norm = normalizeVector(data[i].v, config.p);
        }
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
    int size = ceil(cbrt(config->nw));
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
    setMap1DSize(config);
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
    config->normalize = 1;
    config->stabilizationTrigger = 0.01;
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
    void (*clearscorefp)(void*, somConfig*);
    switch (config->dimension){
        case oneD:
         initfp = getsom1D;
         learnfp = learn1D;
         clearnbfp = clear_neighbours1D;
         clearscorefp = clear_score1D;
         break;
        case threeD:
         initfp = getsom3D;
         learnfp = learn3D;
         clearnbfp = clear_neighbours3D;
         clearscorefp = clear_score3D;
         break;
        default:
         initfp = getsom2D;
         learnfp = learn2D;
         clearnbfp = clear_neighbours2D;
         clearscorefp = clear_score2D;
    }
    void* weights = initfp(data, config, boundaries);
    displayConfig(config);
    somConfig cfg = *config;
    int vectorsToPropose[cfg.n];
    for(int i=0;i<cfg.n;i++){
        vectorsToPropose[i]=i;
    }
    int episode = 0;
    int again = 1;
    short hasStabilized;
    write(weights, config);
    long stepId = 0;
    while(again)
    {
        hasStabilized = 1;
        for(int i=cfg.n-1;i>=0;i--)
        {
            int ivector = (((double)rand()/RAND_MAX)*(i));
            int proposed = vectorsToPropose[ivector];
            hasStabilized = learnfp(&data[proposed], weights, &cfg) & hasStabilized;
            vectorsToPropose[ivector] = vectorsToPropose[i];
            vectorsToPropose[i] = i;
            somScoreResult* score = getscore(data, weights, &cfg);
            writeAppend(stepId++, weights, &cfg, score);
            clearscorefp(score->scores, &cfg);
            free(score);         
        }
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
        again = !hasStabilized && episode <cfg.maxEpisodes;
    }
    if(hasStabilized)
    {
        printf("Stabilized after %d episodes\n", episode);
    }
    else
    {
        printf("Stopped unstabilized after %d ", episode);
    }
    
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

void resetConfig(somConfig* config)
{
    config->map_b=0;
    config->map_c=0;
    config->map_r=0;
    config->nw=0;
    config->radius =0;
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
        dataVector v = data[i];
        somNeuron* winner = find_winner1D(&v, weights, config->nw, config->p);
        updateScore(&score[winner->c], v.class, i);

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
        dataVector v = data[i];
        somNeuron* winner = find_winner2D(&v, weights, config);
        updateScore(&score[winner->r][winner->c], v.class, i);
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
        dataVector v = data[i];
        somNeuron* winner = find_winner3D(&v, weights, config);
        updateScore(&score[winner->b][winner->r][winner->c],v.class, i);
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

int getTerminalColorCode(int index){
    if(index>0)
    {
        int code = index%7;
        if(code>0)
        {
            switch(code)
            {
                case 1: return 31;
                case 2: return 33;
                case 3: return 34;
                case 4: return 32;
                case 5: return 35;
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
                case 1: return 41;
                case 2: return 43;
                case 3: return 44;
                case 4: return 42;
                case 5: return 45;
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
        printf("%21s: %d×\n", "Map blocks", config->map_b);  
    }
    else if(config->dimension != oneD)
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

void displayScore1D(somScoreResult* scoreResult, somConfig* config)
{
    somScore* score = (somScore*)scoreResult->scores;
    for(int i=0;i<config->map_c;i++)
    {
        displayScoreAtom(&score[i]);
    }
}

void displayScore2D(somScoreResult* scoreResult, somConfig* config)
{
    somScore** score = (somScore**)scoreResult->scores;
    for(int i=0;i<config->map_r;i++)
    {
        for(int j=0;j<config->map_c;j++)
        {
            displayScoreAtom(&score[i][j]);
        }
        printf("\n");
    }
}

void displayScore3D(somScoreResult* scoreResult, somConfig* config)
{
    somScore*** score = (somScore***)scoreResult->scores;
    for(int i=0;i<config->map_b;i++)
    {
        printf("Block %d\n", i);
        for(int j=0;j<config->map_r;j++)
        {
            for(int k=0;k<config->map_c;k++)
            {
                displayScoreAtom(&score[i][j][k]);
            }
            printf("\n");
        }
    }
}

void displayScore(somScoreResult* scoreResult, somConfig* config)
{
    void (*displayScorefp)(somScoreResult*, somConfig*);
    switch(config->dimension)
    {
        case oneD: displayScorefp = displayScore1D;break;
        case threeD: displayScorefp = displayScore3D;break;
        default: displayScorefp = displayScore2D;break;
    }
    printf("SOM Map:\n");
    displayScorefp(scoreResult, config);
    printf("\nActivated nodes:%d\n\n", scoreResult->nActivatedNodes);
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

void writeNeuron(FILE* fp, somNeuron* n, somScore* score, long stepId, int p, mapDimension dimension)
{
    fputs("[", fp);
    int x = n->c;
    int y = dimension>oneD? n->r:-1;
    int z = dimension>twoD? n->b:-1;
    int s = -1;
    int class = -1;
    int class2 = -1;
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
    fprintf(fp, "];%d;%d;%d;%ld;%d;%d;%d", x,y,z , stepId, s, class,  class2);
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
        writeNeuron(fp, &som[i], s? &s[i] : NULL, stepId, config->p, oneD);
    }
}

void writeNeurons2D(FILE* fp, void* weights, void* scores, long stepId, somConfig* config)
{
    somNeuron** som = (somNeuron**)weights;
    somScore** s = (somScore**)scores;
    for(int i=0;i<config->map_r;i++)
    {
        for(int j=0;j<config->map_c;j++)
        {
            writeNeuron(fp, &som[i][j], s? &s[i][j] : NULL, stepId, config->p, twoD);
        }  
    }  
}

void writeNeurons3D(FILE* fp, void* weights, void* scores, long stepId, somConfig* config)
{
    somNeuron*** som = (somNeuron***)weights;
    somScore*** s = (somScore***)scores;
    for(int i=0;i<config->map_b;i++)
    {
        for(int j=0;j<config->map_r;j++)
        {
            for(int k=0;k<config->map_c;k++)
            {
                writeNeuron(fp, &som[i][j][k], s? &s[i][j][k] : NULL, stepId, config->p, threeD);
            }  
        }  
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



