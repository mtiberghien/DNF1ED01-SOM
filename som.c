#include "include/som.h"
#include "include/common.h"
#include <string.h>
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
    config->normalize=1;
    config->dimension = twoD;
    config->alpha = 0.01;
    config->initialPercentCoverage = 0.6;
    config->distribution = usingMeans;
    config->nbFactorRadius1 = 0.7;
}

//Get a different file name according to each dimension (visualization purpose)
char* getsomFileName(somConfig* config)
{
    switch(config->dimension)
    {
        case oneD: return config->normalize ? "som1D_n.data":"som1D.data";
        case threeD: return config->normalize ? "som3D_n.data":"som3D.data";
        default: return config->normalize ? "som2D_n.data":"som2D.data";
    }
}

//Reset blocks, rows, columns, node and and radius variable so they can be calculated automatically
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
//Return a random value between a boundary around the mean
double getRandomUsingMean(dataBoundary boundary)
{
    return (((double)rand()/RAND_MAX)*(boundary.max - boundary.min)) + boundary.mean;
}

double getRandomUsingMinMax(dataBoundary boundary)
{
    return (((double)rand()/RAND_MAX)*(boundary.max - boundary.min)) + boundary.min;
}

// Initialize a neuron value
void initNeuron(somNeuron*n, somConfig* config, dataBoundary *boundaries, int b, int r, int c, double (*getrandomfp)(dataBoundary))
{
    int p = config->p;
    n->b=b;
    n->r=r;
    n->c=c;
    n->v=(double*)malloc(p * sizeof(double));
    n->neighbours = NULL;
    n->nc = 0;
    for(int j=0;j<p;j++)
    {
        n->v[j]= getrandomfp(boundaries[j]);
    }
}

//Initialize SOM 1D weights with random values based on parameters boundaries
void initialize1D(somNeuron *weights, somConfig* config, dataBoundary *boundaries, double (*getrandomfp)(dataBoundary))
{
    for(int i=0;i<config->nw; i++){
        initNeuron(&weights[i], config, boundaries,-1,-1,i, getrandomfp);
    }
}

//Initialize SOM 2D weights with random values based on parameters boundaries
void initialize2D(somNeuron** weights, somConfig* config, dataBoundary *boundaries, double (*getrandomfp)(dataBoundary))
{
    for(int i=0;i<config->map_r; i++)
    {
        weights[i] = (somNeuron*)malloc(config->map_c * sizeof(somNeuron));
        for(int j=0;j<config->map_c;j++)
        {
            initNeuron(&weights[i][j], config, boundaries,-1,i,j, getrandomfp);
        }
    }
}

//Initialize SOM 3D weights with random values based on parameters boundaries
void initialize3D(somNeuron*** weights, somConfig* config, dataBoundary *boundaries, double (*getrandomfp)(dataBoundary))
{
    for(int i=0;i<config->map_b; i++)
    {
        weights[i] = (somNeuron**)malloc(config->map_r * sizeof(somNeuron*));
        for(int j=0;j<config->map_r;j++)
        {
            weights[i][j] = (somNeuron*)malloc(config->map_c * sizeof(somNeuron));
            for(int k=0;k<config->map_c;k++){
               initNeuron(&weights[i][j][k], config, boundaries,i,j,k, getrandomfp);
            }
        }
    }
}
//Configure number of columns and size of the radius according to the number of neurons
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
//Configure number of columns, rows and size of the radius according to the number of neurons
void set2DMapSize(somConfig *config)
{
    if(!config->map_r|| !config->map_c)
    {
        int size = ceil(sqrt(config->nw));
        config->map_r=size;
        config->map_c=size;      
    }
    config->nw = config->map_r*config->map_c;

    if(!config->radius){
        config->radius = ceil((sqrt(config->initialPercentCoverage*config->nw) - 1)/2);
    }
}
//Configure number of columns, blocks, rows and size of the radius according to the number of neurons
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
//Initialize 1D SOM
void* getsom1D(dataVector* data, somConfig *config, dataBoundary* boundaries, double (*getrandomfp)(dataBoundary))
{
    setMap1DSize(config);
    somNeuron *weights = (somNeuron*)malloc(config->nw *sizeof(somNeuron));
    initialize1D(weights, config, boundaries, getrandomfp);
    return weights;
}
//Initialize 2D SOM
void* getsom2D(dataVector* data, somConfig *config, dataBoundary* boundaries, double (*getrandomfp)(dataBoundary))
{
    set2DMapSize(config);
    somNeuron **weights = (somNeuron**)malloc(config->map_r * sizeof(somNeuron*));
    initialize2D(weights, config, boundaries, getrandomfp);
    return weights;
}
//Initialize 3D SOM
void* getsom3D(dataVector* data, somConfig *config, dataBoundary* boundaries, double (*getrandomfp)(dataBoundary))
{
    set3DMapSize(config);
    somNeuron ***weights = (somNeuron***)malloc(config->map_b * sizeof(somNeuron**));
    initialize3D(weights, config, boundaries, getrandomfp);
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
//Get the closest neuron for v from 1D SOM
void findWinner1D(dataVector* v,void* weights, somConfig* config)
{
    somNeuron* som = (somNeuron*)weights;
    somNeuron **results = malloc(sizeof(somNeuron*));
    int count = 1;
    double minValue = __DBL_MAX__;
    int p = config->p;
    for(int i = 0; i<config->map_c; i++)
    {
        somNeuron* n = &som[i];
        double distance = distance_function(v->v, n->v, p);
        if(distance<minValue){
            results[count-1]= n;
            minValue = distance;
        }
        else if(distance == minValue){
            results = realloc(results, (++count)*sizeof(somNeuron*));
            results[count-1]= n;
        }
    }
    int selectedIndex = count > 1 ? rand()%count : 0;
    somNeuron* result = results[selectedIndex];
    free(results);
    v->lastWinner = result;
}

//Get the closest neuron for from 2D SOM
void findWinner2D(dataVector* v,void* weights, somConfig* config)
{
    somNeuron** som = (somNeuron**)weights;
    somNeuron **results = malloc(sizeof(somNeuron*));
    int count = 1;
    double minValue = __DBL_MAX__;
    int p = config->p;
    for(int i = 0; i<config->map_r; i++)
    {
        for(int j=0;j<config->map_c;j++)
        {
            somNeuron* n = &som[i][j];
            double distance = distance_function(v->v, n->v, p);
            if(distance<minValue){
                results[count-1]= n;
                minValue = distance;
            }
            else if(distance == minValue){
                results = realloc(results, (++count)*sizeof(somNeuron*));
                results[count-1]= n;
            }
        }
    }
    int selectedIndex = count > 1 ? rand()%count : 0;
    somNeuron* result = results[selectedIndex];
    free(results);
    v->lastWinner = result;
}

//Get the closest neuron for from 2D SOM
void findWinner3D(dataVector* v,void* weights, somConfig* config)
{
    somNeuron*** som = (somNeuron***)weights;
    somNeuron **results = malloc(sizeof(somNeuron*));
    int count = 1;
    double minValue = __DBL_MAX__;
    int p = config->p;
    for(int i = 0; i<config->map_b; i++)
    {
        for(int j=0;j<config->map_r;j++)
        {
            for(int k=0;k<config->map_c;k++)
            {
                somNeuron* n = &som[i][j][k];
                double distance = distance_function(v->v, n->v, p);
                if(distance<minValue){
                    results[count-1]= n;
                    minValue = distance;
                }
                else if(distance == minValue){
                    results = realloc(results, (++count)*sizeof(somNeuron*));
                    results[count-1]= n;
                }
            }
        }
    }
    int selectedIndex = count > 1 ? rand()%count : 0;
    somNeuron* result = results[selectedIndex];
    free(results);
    v->lastWinner = result;
}

//Get the nearest neuron among neighbours of last winner. If the last winner is stabilized return the last winner
somNeuron* getNearest(dataVector* v, void* weights, somConfig* config)
{
    somNeuron* neuron = v->lastWinner;
    somNeuron **results = malloc(sizeof(somNeuron*));
    int count = 1;
    double minValue = __DBL_MAX__;
    int p = config->p;
    for(int i = 0; i<neuron->nc; i++)
    {
        somNeuron* n = neuron->neighbours[i];
        double distance = distance_function(v->v, n->v, p);
        if(distance<minValue){
            results[count-1]= n;
            minValue = distance;
        }
        else if(distance == minValue){
            results = realloc(results, (++count)*sizeof(int));
            results[count-1]= n;
        }
    }
    int selectedIndex = count > 1 ? rand()%count : 0;
    somNeuron* result = results[selectedIndex];
    free(results);
    return result;
}
//Get the nearest neuron from last winner and neighbours.
//If the new winner is not the last winner recursively find the winner ammong new winner neighbours
void find_winner_fromNeighbours(dataVector* v, void * weights, somConfig* config)
{
    somNeuron* nearest = getNearest(v, weights, config);
    if(nearest)
    {
        while(nearest!=v->lastWinner)
        {
            v->lastWinner = nearest;
            nearest = getNearest(v, weights, config);
        }
    }
}
//Add a neigbour to a node
void addNeighbour(somNeuron* n, somNeuron* nb)
{
    n->neighbours[n->nc]= nb;
    n->nc ++;
    n->neighbours = (somNeuron**)realloc(n->neighbours, (n->nc+1)*sizeof(somNeuron*)); 
}
//Initialize the neighbours of a neuron (including itself) for a 1D SOM
void getNeighbours1D(somNeuron* n, void* weights, somConfig* config)
{
    somNeuron* som = (somNeuron*)weights;
    int start = max(0, n->c - config->radius);
    int end = min(config->map_c - 1, n->c + config->radius);
    n->neighbours = (somNeuron**)malloc(sizeof(somNeuron*));
    for(int i=start;i<=end;i++)
    {
        somNeuron* nb = &som[i];
        addNeighbour(n, nb);
    }
}
//Initialize the neighbours of a neuron (including itself) for a 2D SOM
void getNeighbours2D(somNeuron* n, void* weights, somConfig* config)
{
    somNeuron** som = (somNeuron**)weights;
    int start_r = max(0, n->r - config->radius);
    int end_r = min(config->map_r -1, n->r + config->radius);
    int start_c = max(0, n->c - config->radius);
    int end_c = min(config->map_c -1, n->c + config->radius);
    int rc = (end_r - start_r);
    int cc = (end_c - start_c);
    n->neighbours = (somNeuron**)malloc(sizeof(somNeuron*));
    for(int i=start_r;i<=end_r;i++)
    {
        for(int j=start_c;j<=end_c;j++)
        {
            somNeuron* nb = &som[i][j];
            addNeighbour(n, nb);
        }
    }
}
//Initialize the neighbours of a neuron (including itself) for a 3D SOM
void getNeighbours3D(somNeuron* n, void* weights, somConfig* config)
{
    somNeuron*** som = (somNeuron***)weights;
    int start_r = max(0, n->r - config->radius);
    int end_r = min(config->map_r - 1, n->r + config->radius);
    int start_c = max(0, n->c - config->radius);
    int end_c = min(config->map_c - 1, n->c + config->radius);
    int start_b = max(0, n->b - config->radius);
    int end_b = min(config->map_b - 1, n->b + config->radius);
    int rc = end_r - start_r;
    int cc = end_c - start_c;
    int bc = end_b - start_b;
    n->neighbours = (somNeuron**)malloc(sizeof(somNeuron*));
    for(int i=start_b;i<=end_b;i++)
    {
        for(int j=start_r;j<=end_r;j++)
        {
            for(int k=start_c;k<=end_c;k++)
            {
                somNeuron* nb = &som[i][j][k];
                addNeighbour(n, nb);
            }

        }
    }
}

#pragma endregion

#pragma region Update Section
double gaussian_function(double distance, double sigma)
{
    return exp(-distance/2*(sigma*sigma));
}

//Return a value between 0 and 1 according to the 1D distance between the winner neuron and another neuron and a neighborhood factor using neurons coordinates
double neighborhood_function1d(somNeuron* winner, somNeuron* n, double sigma)
{
    double wp[1]={winner->c};
    double np[1]={n->c};
    return gaussian_function(distance_function(wp,np,1), sigma);
}

//Return a value between 0 and 1 according to the 2D distance between the winner neuron and another neuron and a neighborhood factor using using neurons coordinates
double neighborhood_function2d(somNeuron* winner, somNeuron* n, double sigma)
{
    double wp[2]={winner->c, winner->r};
    double np[2]={n->c, n->r};
    return gaussian_function(distance_function(wp,np,2), sigma);
}

//Return a value between 0 and 1 according to the 3D distance between the winner neuron and another neuron and a neighborhood factor using using neurons coordinates
double neighborhood_function3d(somNeuron* winner, somNeuron* n, double sigma)
{
    double wp[3]={winner->c, winner->r, winner->b};
    double np[3]={n->c, n->r, n->b};
    return gaussian_function(distance_function(wp,np,3), sigma);
}

//Return absolute value
double absd(double v)
{
    return v>0?v:-v;
}

//Update the vector using formula alpha (learning rate)*h(neighborhood function)*(distance between parameters of the vector)
void updateNeuron(dataVector* v, somNeuron* n, double h, somConfig* config)
{
    for(int i=0;i<config->p;i++){
        double delta = config->alpha * h *(v->v[i] - n->v[i]);
        n->v[i] += delta;
    }
}

//Calculate the 1D SOM neighborhood function then update a neuron
void updateNeuron1D(dataVector* v, somNeuron* winner, somNeuron* n, somConfig* config)
{   
    double h = neighborhood_function1d(winner, n, config->sigma);
    updateNeuron(v, n, h, config);
}

//Calculate the 2D SOM neighborhood function then update a neuron
void updateNeuron2D(dataVector* v, somNeuron* winner, somNeuron* n, somConfig* config){
    double h = neighborhood_function2d(winner, n, config->sigma);
    updateNeuron(v, n, h, config);
}

//Calculate the 3D SOM neighborhood function then update a neuron
void updateNeuron3D(dataVector* v, somNeuron* winner, somNeuron* n, somConfig* config){
    double h = neighborhood_function3d(winner, n, config->sigma);
    updateNeuron(v, n, h, config);
}

//Free and update the list of neighbours for a specific neuron using the method passed as a parameter
void update_neigbhoursAtom(somNeuron* n, void* weights, somConfig* config, void (*getnbfp)(somNeuron*,void*,somConfig*))
{
    free(n->neighbours);
    n->neighbours = NULL;
    n->nc = 0;
    getnbfp(n,weights, config);
}

//Update the list of neighbours for each 1D SOM neuron
void update_neighbours1D(void* weights, somConfig* config)
{
    somNeuron* som = (somNeuron*)weights;
    for(int i=0; i< config->map_c; i++)
    {
        update_neigbhoursAtom(&som[i], weights, config, getNeighbours1D);
    }
}

//Update the list of neighbours for each 2D SOM neuron
void update_neighbours2D(void* weights, somConfig* config)
{
    somNeuron** som = (somNeuron**)weights;
    for(int i=0; i< config->map_r; i++)
    {
        for(int j=0; j< config->map_c; j++)
        {
            update_neigbhoursAtom(&som[i][j], weights, config, getNeighbours2D);
        }
    }
}

//Update the list of neighbours for each 3D SOM neuron
void update_neighbours3D(void* weights, somConfig* config)
{
    somNeuron*** som = (somNeuron***)weights;
    for(int i=0; i< config->map_b; i++)
    {
        for(int j=0; j< config->map_r; j++)
        {
            for(int k=0; k< config->map_c; k++)
            {
                update_neigbhoursAtom(&som[i][j][k], weights, config, getNeighbours3D);
            }
        }
    }
}

//Update winner neuron and neighbours for 1D SOM
void updateNeurons1D(dataVector* v, somNeuron  *weights, somConfig* config)
{
    somNeuron* winner = v->lastWinner;
    for(int i=0; i<winner->nc;i++)
    {
      updateNeuron1D(v, winner, winner->neighbours[i], config);      
    };
}

//Update winner neuron and neighbours for 2D SOM
void updateNeurons2D(dataVector* v, somNeuron  **weights, somConfig* config)
{
    somNeuron* winner = v->lastWinner;
    for(int i=0; i<winner->nc;i++)
    {
      updateNeuron2D(v, winner, winner->neighbours[i], config);      
    };
}

//Update winner neuron and neighbours for 3D SOM
void updateNeurons3D(dataVector* v, somNeuron  ***weights, somConfig* config)
{
    somNeuron* winner = v->lastWinner;
    for(int i=0; i<winner->nc;i++)
    {
      updateNeuron3D(v, winner, winner->neighbours[i], config);      
    };
}
#pragma endregion
//Learn for a 1D SOM retrieving the winner and updating neighbours
void learn1D(int vectorIndex, dataVector* v, void* weights, somConfig* config, void (*findwnfp)(dataVector*, void*,somConfig*))
{
    findwnfp(v,weights,config);
    updateNeurons1D(v, (somNeuron*)weights, config);
}

//Learn for a 2D SOM retrieving the winner and updating neighbours
void learn2D(int vectorIndex, dataVector* v, void* weights, somConfig* config,void (*findwnfp)(dataVector*, void*,somConfig*))
{
    findwnfp(v,weights,config);
    updateNeurons2D(v, (somNeuron**)weights, config);
}

//Learn for a 3D SOM retrieving the winner and updating neighbours
void learn3D(int vectorIndex, dataVector* v, void* weights, somConfig* config,void (*findwnfp)(dataVector*, void*,somConfig*))
{
   findwnfp(v,weights,config);
   updateNeurons3D(v, (somNeuron***)weights, config);
}
#pragma endregion

#pragma region Clear Section
//Free score memory for 1D SOM
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

//Free score memory for 2D SOM
void clear_score2D(void* score, somConfig* config)
{
    somScore** sc = (somScore**)score;
    for(int i = 0; i < config->map_r; i++)
    {
        clear_score1D(sc[i], config);
    }
    free(sc);
}

//Free score memory for 3D SOM
void clear_score3D(void* score, somConfig* config)
{
    somScore*** sc = (somScore***)score;
    for(int i = 0; i < config->map_b; i++)
    {
        clear_score2D(sc[i], config);
    }
    free(sc);
}

//Clear score result memory
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
//Clear memory for neurons and optional score in 1D SOM
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

//Clear memory for neurons and optional score in 2D SOM
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

//Clear memory for neurons and optional score in 3D SOM
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

//Clear memory config object
void clear_config(somConfig* config)
{
    free(config);
}

//Clear memory for neurons, optional score result and config object
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
double getDistance1D(int i)
{
    double n1[]={0};
    double n2[]={i};
    return distance_function(n1, n2, 1);
}

double getDistance2D(int i)
{
    double n1[]={0,0};
    double n2[]={i,i};
    return distance_function(n1, n2, 2);
}

double getDistance3D(int i)
{
    double n1[]={0,0,0};
    double n2[]={i,i,i};
    return distance_function(n1, n2, 3);
}
double getInitialSigma(somConfig* config, double(*getdistfp)(int))
{
    double dist = getdistfp(1);
    double sigma = sqrt(-dist/2*log(0.8));
    return sigma;
}

void* init(dataVector* data, somConfig* config, dataBoundary* boundaries, short silent)
{
    if(!config->nw){
        config->nw = floor(5 *sqrt(config->n *1.0));
        config->nw -= config->nw%12;
    }
    if(config->alpha<=0)
    {
        config->alpha = 0.01;
    }
    if(!config->epochs)
    {
        config->epochs = config->n*(1/config->alpha);
    }
    double (*getrandomfp)(dataBoundary);
    switch(config->distribution)
    {
        case usingMeans: getrandomfp = getRandomUsingMean;break;
        default: getrandomfp = getRandomUsingMinMax;break;
    }
    void* (*initfp)(dataVector*, somConfig*, dataBoundary*, double(*)(dataBoundary));
    double (*getdistfp)(int);
    switch (config->dimension){
        case oneD:
            initfp = getsom1D;
            getdistfp = getDistance1D;
         break;
        case threeD:
            initfp = getsom3D;
            getdistfp = getDistance3D;
         break;
        default:
        config->dimension = twoD;
            initfp = getsom2D;
            getdistfp = getDistance2D;
        break;
    }
    if(!silent)
    {
        printf("Initializing SOM: ");
        fflush(stdout);
    }
    void* weights = initfp(data, config, boundaries, getrandomfp);
    config->sigma = getInitialSigma(config, getdistfp);
    if(!silent)
    {
        printf("%s\n", "Done");
    }
    return weights;
}

void fit(dataVector* data, void* weights, somConfig* config, short silent)
{
    void (*learnfp)(int, dataVector*, void*, somConfig*, void (*)(dataVector*,void*,somConfig*));
    void (*updatenbfp)(void*, somConfig*);
    void (*findwnfp)(dataVector*, void*, somConfig*);
#ifdef TRACE_SOM
    char *filename = getsomFileName(config);
    void (*clearscorefp)(void*, somConfig*);
#endif
    switch (config->dimension){
        case oneD:
            learnfp = learn1D;
            updatenbfp = update_neighbours1D;
            findwnfp = findWinner1D;
#ifdef TRACE_SOM
         clearscorefp = clear_score1D;
#endif
         break;
        case threeD:
            learnfp = learn3D;
            updatenbfp = update_neighbours3D;
            findwnfp = findWinner3D;
#ifdef TRACE_SOM
         clearscorefp = clear_score3D;
#endif
         break;
        default:
            config->dimension = twoD;
            learnfp = learn2D;
            updatenbfp = update_neighbours2D;
            findwnfp = findWinner2D;
#ifdef TRACE_SOM
         clearscorefp = clear_score2D;
#endif
        break;
    }
    somConfig cfg = *config;
    double tau = cfg.epochs/log(cfg.sigma);
    int vectorsToPropose[cfg.n];
    if(config->nw == 0)
    {
        if(!silent)
        {
            printf("Aborted, need at least one neuron\n");
        }
        return;
    }
    if(!silent)
    {
        printf("Initializing SOM Neighbours: ");
        fflush(stdout);
    }
    updatenbfp(weights, config);
    if(!silent)
    {
        printf("Done\n");
    }
    if(!silent)
    {
        printf("Initializing SOM Last winners: ");
        fflush(stdout);
    }
    int n = config->n;
    for(int i=0;i<n;i++)
    {
         if(!silent)
        {
            printf("%6.2f%%\033[7D", (double)i*100/n);
            fflush(stdout);
        }
        findwnfp(&data[i], weights, config);
        vectorsToPropose[i]=i;
    }
    if(!silent)
    {
        printf("%7s\033[7D%s\n", "","Done");
    }
    if(!silent)
    {
        printf("Calculating  %dD SOM for %d entries and %d parameters :", config->dimension, config->n, config->p);
        fflush(stdout);
    }
    long epoch = 0;
    int currentRadius = cfg.radius;
    double tau2 = cfg.epochs/log(currentRadius);
    findwnfp = find_winner_fromNeighbours;
#ifdef TRACE_SOM
    long time = 0;
    writeSomHisto(filename, weights, config, NULL);
#endif
    while(epoch<cfg.epochs)
    {
        for(int i=cfg.n-1;i>=0;i--)
        {
            if(!silent)
            {
                printf("%6.2f%%\033[7D", (double)epoch*100/cfg.epochs);
                fflush(stdout);
            }
            cfg.alpha = config->alpha*exp(-(double)epoch/cfg.epochs);
            int ivector = ((double)rand()/RAND_MAX)*i;
            int proposed = vectorsToPropose[ivector];
            learnfp(proposed,  &data[proposed], weights, &cfg, findwnfp);
            vectorsToPropose[ivector] = i;
#ifdef TRACE_SOM
                if(time++%TRACE_SOM == 0)
                {
                    somScoreResult* result = getscore(data, weights, config, 1, 1);
                    writeSomHistoAppend(filename, epoch, weights, config, result);
                    clearscorefp(result->scores, config);
                    free(result);
                }
#endif  
            epoch++;
            if(epoch == cfg.epochs)
            {
                break;
            }

        }
        currentRadius = (int)(config->radius*exp(-epoch/tau2));
        if(currentRadius<cfg.radius)
        {
            cfg.radius = currentRadius;
            updatenbfp(weights, &cfg);
        }
    }
#ifdef TRACE_SOM
    somScoreResult* result = getscore(data, weights, config,1,1);
    writeSomHistoAppend(filename, epoch, weights, config, result);
    clearscorefp(result->scores, config);
    free(result);
#endif
    if(!silent)
    {
        {
            printf("Done after %ld epochs\n", epoch);
        }
    }  
}

//Get stabilized som neurons that has been train using provided data and config
//The method initialize first the fonctions to use according to the configured dimension
//The it evaluates the size of SOM map and initialize it
//The learning loop is evaluating randomly each entry from the dataset during one episode
//After each episode the method evaluates if som neurons have stabilized
//When all neurons have stabilized return the neurons map
void* getTrainedSom(dataVector* data, somConfig *config, dataBoundary* boundaries, short silent)
{
    void* weights = init(data,config, boundaries, silent);
    fit(data, weights, config, silent);
    return weights;
}

#pragma region Scoring Section
//Return the number of classes (categories stored as an integer) of a dataset. Used for scoring purpose
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
//Get the index of element having the maximum value
//Skip index is used to retrieve the second best
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
//Evaluates the status of a neuron -1 means that the neuron was never activated, 0 means that identifies a unique class, 1 means several classes
short getScoreStatus(int* values, int n)
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
//Initializes a somScore object
void initScore(somScore* s, int nClasses)
{
    s->scores = malloc(sizeof(int)*nClasses);
    s->entries = malloc(sizeof(int));
    s->totalEntries = 0;
    s->maxClasstotalEntries = 0;
    s->secondClasstotalEntries = 0;
    for(int i=0;i<nClasses;i++)
    {
        s->scores[i]=0;
    }
}
//Update somScore statistics increasing the number of vectors for a specific class.
//Each vector index is also stored (for visualization purpose)
void updateScore(somScore* s, int class, int entry)
{
    s->scores[class]++;
    s->entries[s->totalEntries]=entry;
    s->totalEntries++;
    s->entries = realloc(s->entries, (s->totalEntries+1)*sizeof(int));
}
//Retrieve the status, the best matching class and the second best
void updateScoreStats(somScore* s, int nClasses, somScoreResult* scoreResult)
{
    s->maxClass = s->secondClass = -1;
    s->status = getScoreStatus(s->scores, nClasses);
    if(s->status >=0)
    {
        scoreResult->nActivatedNodes++;
        s->maxClass = argMax(s->scores, nClasses, -1);
        s->maxClasstotalEntries = s->scores[s->maxClass];
        if(s->status >0)
        {
            s->secondClass = argMax(s->scores, nClasses, s->maxClass);
            s->secondClasstotalEntries = s->scores[s->secondClass];
        }
        
    }
}
//Retrieve scores for 1D SOM neurons
void score1D(dataVector* data, void* weights, somConfig* config, somScoreResult* scoreResult, void (*findwnfp)(dataVector*,void*,somConfig*), short silent)
{
    somScore* score = (somScore*)malloc(sizeof(somScore) * config->map_c);
    somNeuron* som = (somNeuron*)weights;
    scoreResult->nActivatedNodes=0;
    int nClasses = scoreResult->nClasses;
    if(!silent)
    {
        printf("Initializing 1D Scores");
        fflush(stdout);
    }
    for(int i=0;i<config->map_c;i++)
    {
        initScore(&score[i], nClasses);
    }
    if(!silent)
    {
        printf("\033[2K\rFinding winners: ");
        fflush(stdout);
    }
    int n = config->n;
    for(int i=0;i<n;i++)
    {
        if(!silent)
        {
            printf("%6.2f%%\033[7D", (double)i*100/n);
            fflush(stdout);
        }
        dataVector* v = &data[i];
        findwnfp(v,weights,config);
        somNeuron* winner = v->lastWinner;
        updateScore(&score[winner->c],v->class, i);

    }
    if(!silent)
    {
        printf("\033[2K\rUpdating stats: ");
        fflush(stdout);

    }
    for(int i=0;i<config->map_c;i++)
    {
        updateScoreStats(&score[i], nClasses, scoreResult);
    }
    if(!silent)
    {
        printf("\033[2K\r");
    }
    scoreResult->scores = score;
}
//Retrieve scores for 2D SOM neurons
void score2D(dataVector* data, void* weights, somConfig* config, somScoreResult* scoreResult, void (*findwnfp)(dataVector*,void*,somConfig*), short silent)
{
    somScore** score = (somScore**)malloc(sizeof(somScore*) * config->map_r);
    somNeuron** som = (somNeuron**)weights;
    scoreResult->nActivatedNodes=0;
    int nClasses = scoreResult->nClasses;
    if(!silent)
    {
        printf("Initializing 2D Scores: ");
        fflush(stdout);
    }
    for(int i=0;i<config->map_r;i++)
    {
        score[i] = (somScore*)malloc(sizeof(somScore)* config->map_c);
        for(int j=0;j<config->map_c;j++)
        {
            initScore(&score[i][j], nClasses);
        }
    }
    if(!silent)
    {
        printf("\033[2K\rFinding winners: ");
        fflush(stdout);
    }
    int n = config->n;
    for(int i=0;i<n;i++)
    {
        if(!silent)
        {
            printf("%6.2f%%\033[7D", (double)i*100/n);
            fflush(stdout);
        }
        dataVector* v = &data[i];
        findwnfp(v,weights,config);
        somNeuron* winner = v->lastWinner;
        updateScore(&score[winner->r][winner->c],v->class, i);
    }
    if(!silent)
    {
        printf("\033[2K\rUpdating stats: ");
        fflush(stdout);

    }
    for(int i=0;i<config->map_r;i++)
    {
        for(int j=0;j<config->map_c;j++)
        {
            updateScoreStats(&score[i][j], nClasses, scoreResult);
        }
        
    }
    if(!silent)
    {
        printf("\033[2K\r");
    }
    scoreResult->scores = score;
}
//Retrieve scores for 3D SOM neurons
void score3D(dataVector* data, void* weights, somConfig* config, somScoreResult* scoreResult, void (*findwnfp)(dataVector*,void*,somConfig*), short silent)
{
    somScore*** score = (somScore***)malloc(sizeof(somScore**) * config->map_b);
    somNeuron*** som = (somNeuron***)weights;
    scoreResult->nActivatedNodes=0;
    int nClasses = scoreResult->nClasses;
    if(!silent)
    {
        printf("Initializing 3D Scores: ");
        fflush(stdout);
    }
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
    if(!silent)
    {
        printf("\033[2K\rFinding winners: ");
        fflush(stdout);
    }
    int n = config->n;
    for(int i=0;i<n;i++)
    {
        if(!silent)
        {
            printf("%6.2f%%\033[7D", (double)i*100/n);
            fflush(stdout);
        }
        dataVector* v = &data[i];
        findwnfp(v,weights,config);
        somNeuron* winner = v->lastWinner;
        updateScore(&score[winner->b][winner->r][winner->c],v->class, i);
    }
    if(!silent)
    {
        printf("\033[2K\rUpdating stats: ");
        fflush(stdout);

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
    if(!silent)
    {
        printf("\033[2K\r");
    }
    scoreResult->scores = score;
}
//Retrieve the scoreResult object for specific weights and config
somScoreResult* getscore(dataVector* data, void* weights, somConfig* config, short useFromNeighbours, short silent)
{
    if(!silent)
    {
        printf("Scoring %d neurons using %d data and %d parameters:\n", config->nw, config->n, config->p);
        fflush(stdout);
    }
    somScoreResult* result = malloc(sizeof(somScoreResult));
    void (*scorefp)(dataVector*,  void* , somConfig*, somScoreResult*, void (*)(dataVector*,void*,somConfig*), short);
    void (*findwnfp)(dataVector*,void*,somConfig*);
    switch (config->dimension)
    {
        case oneD: scorefp = score1D;
            findwnfp = findWinner1D;
         break;
        case threeD: scorefp = score3D;
            findwnfp = findWinner3D;
         break;
        default: scorefp = score2D;
            findwnfp = findWinner2D;
         break;
    }
    if(useFromNeighbours)
    {
        findwnfp = find_winner_fromNeighbours;
    }
    result->nClasses = getClassesCount(data, config->n);
    scorefp(data, weights, config, result, findwnfp, silent);
    if(!silent)
    {
        printf("\033[A\033[KScoring %d neurons using %d data and %d parameters: Done\n", config->nw, config->n, config->p);
    }
    return result;
}
#pragma endregion

#pragma region  Display Section
int getTerminalColorCode(int index){
    if(index>=0)
    {
        int code = index%6;
        switch(code)
        {
            //red
            case 0: return 31;
            //yellow
            case 1: return 33;
            //blue
            case 2: return 34;
            //green
            case 3: return 32;
            //magenta
            case 4: return 35;
            //cyan
            case 5: return 36;
        }
    }
    //default
    return 39;
}

int getTerminalBGColorCode(int index){
    if(index>=0)
    {
        int code = index%6;
        switch(code)
        {
            //red
            case 0: return 41;
            //yellow
            case 1: return 43;
            //blue
            case 2: return 44;
            //green
            case 3: return 42;
            //magenta
            case 4: return 45;
            //cyan
            case 5: return 46;
        }
    }
    //default
    return 49;
}

void displayConfig(somConfig* config)
{
    printf("SOM Settings:\n%21s: %d\n%21s: %d\n%21s: %d\n%21s: %d\n%21s: %d\n%21s: %.2f\n%21s: %.2f\n%s: %.f%%\n", "Entries", config->n,
                        "Parameters", config->p, "Dimension", config->dimension, "Neurons", config->nw , "Radius", config->radius,
                        "Learning rate", config->alpha, "Neighbours factor r1", config->nbFactorRadius1,
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
    int c = s.maxClass;
    if(s.status<1)
    {
        printf("\033[0;%dm", getTerminalColorCode(c));
    }
    else
    {
        printf("\033[0;%d;%dm", getTerminalColorCode(c), getTerminalBGColorCode(s.secondClass));
    }
    if(c>=0)
    {
        printf("%d", c);
    }
    else
    {
        printf(" ");
    }
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
//Write neuron as csv entry using neurons coordinates and score the step is stored vor visualization purpose
void writeNeuron(FILE* fp, somNeuron* n, somScore* score, long stepId, int p)
{
    fputs("[", fp);
    int x = n->c;
    int y = n->r;
    int z = n->b;
    int s = -1;
    int class = -1;
    int classtotalEntries = -1;
    int class2 = -1;
    int class2totalEntries = -1;
    //TODO add stabilization handling (low cost)
    int stabilized = 0;
    if(score)
    {
        s = score->totalEntries;
        classtotalEntries = score->maxClasstotalEntries;
        class = score->maxClass;
        class2 = score->secondClass;
        class2totalEntries = score->secondClasstotalEntries;
    }
    for(int i=0;i<p;i++){
        fprintf(fp, "%f", n->v[i]);
        if(i<p-1)
        {
            fputs(",", fp);
        }
    }
    fprintf(fp, "];%d;%d;%d;%ld;%d;%d;%d;%d;%d", x,y,z , stepId, s, class,  class2, classtotalEntries, class2totalEntries);
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

//Serializes 1D SOM neurons
void writeNeurons1D(FILE* fp, void* weights, void* scores, long stepId, somConfig* config)
{
    somNeuron* som = (somNeuron*)weights;
    somScore* s = (somScore*)scores;
    for(int i=0;i<config->map_c;i++)
    {
        writeNeuron(fp, &som[i], s? &s[i] : NULL, stepId, config->p);
    }
}
//Serializes 2D SOM neurons
void writeNeurons2D(FILE* fp, void* weights, void* scores, long stepId, somConfig* config)
{
    somNeuron** som = (somNeuron**)weights;
    somScore** s = (somScore**)scores;
    for(int i=0;i<config->map_r;i++)
    {
        writeNeurons1D(fp, som[i], s? s[i] : NULL, stepId, config);
    }  
}
//Serializes 3D SOM neurons
void writeNeurons3D(FILE* fp, void* weights, void* scores, long stepId, somConfig* config)
{
    somNeuron*** som = (somNeuron***)weights;
    somScore*** s = (somScore***)scores;
    for(int i=0;i<config->map_b;i++)
    {
        writeNeurons2D(fp, som[i], s? s[i] : NULL, stepId, config);
    }      
}

//Serializes neurons for a specific stepid and a specific socre result
void writeNeurons(FILE *fp, void *weights, somConfig* config, long stepid, somScoreResult* scoreResult)
{
    void (*writefp)(FILE*,void*, void*, long, somConfig*);
    switch(config->dimension){
        case oneD: writefp = writeNeurons1D;break;
        case threeD: writefp = writeNeurons3D;break;
        default: writefp = writeNeurons2D;break;
    }
    writefp(fp, weights, scoreResult ? scoreResult->scores : NULL, stepid,config);
}

void saveNeuron(FILE* fp, somNeuron* n, int p)
{
    fprintf(fp, "%d;%d;%d;", n->b,n->r,n->c);
    for(int i=0;i<p;i++){
        fprintf(fp, "%f", n->v[i]);
        if(i<p-1)
        {
            fputs(";", fp);
        }
    }
    fputs("\n", fp);
}

//Serializes 1D SOM neurons
void saveNeurons1D(FILE* fp, void* weights, somConfig* config)
{
    somNeuron* som = (somNeuron*)weights;
    for(int i=0;i<config->map_c;i++)
    {
        saveNeuron(fp, &som[i], config->p);
    }
}
//Serializes 2D SOM neurons
void saveNeurons2D(FILE* fp, void* weights, somConfig* config)
{
    somNeuron** som = (somNeuron**)weights;
    for(int i=0;i<config->map_r;i++)
    {
        saveNeurons1D(fp, som[i], config);
    }  
}
//Serializes 3D SOM neurons
void saveNeurons3D(FILE* fp, void* weights, somConfig* config)
{
    somNeuron*** som = (somNeuron***)weights;
    for(int i=0;i<config->map_b;i++)
    {
        saveNeurons2D(fp, som[i], config);
    }      
}

//Serializes neurons for a specific stepid and a specific socre result
void saveNeurons(FILE *fp, void *weights, somConfig* config)
{
    void (*writefp)(FILE*,void*, somConfig*);
    switch(config->dimension){
        case oneD: writefp = saveNeurons1D;break;
        case threeD: writefp = saveNeurons3D;break;
        default: writefp = saveNeurons2D;break;
    }
    writefp(fp, weights,config);
}

//Append a specific step screenshot of SOM neurons
void writeSomHistoAppend(char* filename, long stepid, void *weights, somConfig* config, somScoreResult* scoreResult)
{
    FILE * fp;
    fp = fopen(filename, "a");
    if(fp != NULL)
    {
        writeNeurons(fp, weights, config, stepid, scoreResult);
    }
    fclose(fp);
}

//Write a screenshot of SOM neurons
void writeSomHisto(char* filename, void* weights, somConfig* config, somScoreResult* scoreResult){
    FILE * fp;
    fp = fopen(filename, "w");
    if(fp != NULL)
    {
        writeNeurons(fp, weights, config, scoreResult ? 0 : -1, scoreResult);
    }
    fclose(fp);
}

void writeConfig(FILE *fp, somConfig* config)
{
    fprintf(fp, "%d;%d;%d;%d;%d;%d;%d;%d;%f;%f;%f;%f;%d;%ld\n", config->n, config->p, config->nw, config->dimension, config->map_b, config->map_r,config->map_c,
                                    config->radius, config->alpha, config->sigma, config->initialPercentCoverage,
                                    config->nbFactorRadius1, config->normalize, config->epochs);
}

void readConfig(char* line, somConfig* config)
{
    char delim[]=";";
    char *ptr = strtok(line, delim);
    int column=0;
    while(ptr != NULL && column < 14)
    {
        switch(column)
        {
            case 0: config->n = atoi(ptr); break;
            case 1: config->p = atoi(ptr);break;
            case 2: config->nw = atoi(ptr);break;
            case 3: config->dimension = (mapDimension)atoi(ptr);break;
            case 4: config->map_b = atoi(ptr);break;
            case 5: config->map_r = atoi(ptr);break;
            case 6: config->map_c = atoi(ptr);break;
            case 7: config->radius = atoi(ptr);break;
            case 8: config->alpha = atof(ptr);break;
            case 9: config->sigma = atof(ptr);break;
            case 10: config->initialPercentCoverage = atof(ptr);break;
            case 11: config->nbFactorRadius1 = atof(ptr);break;
            case 12: config->normalize = (short)atoi(ptr);break;
            case 13: config->epochs = atol(ptr);break;
        }
        ptr = strtok(NULL, delim);
        column++;
    }
    if(ptr)
    {
        free(ptr);
    }
}

void readSom(FILE* fp, somNeuron* n, int p)
{
    size_t len = 0;
    ssize_t read;
    char* line = NULL;
    read = getline(&line, &len, fp);
    char delim[]=";";
    char *ptr = strtok(line, delim);
    int column=0;
    n->neighbours = NULL;
    n->nc = 0;
    while(ptr != NULL && column < 3)
    {
        switch(column)
        {
            case 0: n->b = atoi(ptr);break;
            case 1: n->r = atoi(ptr);break;
            case 2: n->c = atoi(ptr);break;
        }
        ptr = strtok(NULL, delim);
        column++;
    }
    column =0;
    n->v = malloc(p*sizeof(double));
    while(ptr!= NULL && column<p)
    {
        n->v[column] = atof(ptr);
        ptr = strtok(NULL, delim);
        column++;
    }
    if(ptr)
    {
        free(ptr);
    }
    if(line)
    {
        free(line);
    }
}

void* readSom1D(FILE* fp, somConfig* config)
{
    int p= config->p;
    somNeuron* weights = (somNeuron*)malloc(config->map_c*sizeof(somNeuron));
    for(int i=0;i<config->map_c;i++)
    {
        readSom(fp, &weights[i], p);
    }
    return weights;
}

void* readSom2D(FILE* fp, somConfig* config)
{
    somNeuron** weights = (somNeuron**)malloc(config->map_r*sizeof(somNeuron*));
    for(int i=0;i<config->map_r;i++)
    {
        weights[i]= readSom1D(fp, config);
    }
    return weights;
}

void* readSom3D(FILE* fp, somConfig* config)
{
    somNeuron*** weights = (somNeuron***)malloc(config->map_b*sizeof(somNeuron**));
    for(int i=0;i<config->map_b;i++)
    {
        weights[i]= readSom2D(fp, config);
    }
    return weights;
}

void saveSom(void* weights, somConfig* config, char* filename)
{
    FILE * fp;
    fp = fopen(filename, "w");
    if(fp != NULL)
    {
        writeConfig(fp, config);
        saveNeurons(fp, weights, config);
    }
    fclose(fp);
}

void* loadSom(char* filename, somConfig* config)
{
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    fp = fopen(filename, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    int ln = 0;
    read = getline(&line, &len, fp);
    if(read>1)
    {
        readConfig(line, config);
    }
    void* (*readfp)(FILE*, somConfig*);
    switch(config->dimension){
        case oneD: readfp = readSom1D;break;
        case threeD: readfp = readSom3D;break;
        default: readfp = readSom2D;break;
    }
    void* weights = readfp(fp, config);

    fclose(fp);
    if(line)
    {
        free(line);
    }
    return weights;
}


#pragma endregion



