#include "include/som.h"
#include "include/common.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// Return min between 2 doubles
double min(double x, double y)
{
    return y < x ? y : x;
}
// Return max between 2 doubles
double max(double x, double y){
    return y > x ? y : x;
}

void setLocationFromIndex(int index, somConfig config, neuronLocation* location){
    location->y = index/config.map_c;
    location->x = index%config.map_c;
}

int getIndexFromLocation(neuronLocation location, somConfig config){
    return (config.map_c * location.y) + location.x;
}

//Calculate the euclidian distance between 2 vectors using p dimensions
double distance_function(double *v, double *w, int p)
{
    double sum=0;
    for(int i=0;i<p;i++){
        sum+= pow((v[i] - w[i]),2);
    }
    return sqrt(sum);
}
//Return the index of the neuron closest to a entry vector using the distance_function
int fi_function(dataVector v, somNeuron *weights, int nw, int p)
{
    int *results = malloc(sizeof(int));
    int count = 1;
    double minValue = __DBL_MAX__;
    for(int i = 0; i<nw; i++){
        double distance = distance_function(v.v, weights[i].w, p);
        if(distance<minValue){
            results[count-1]= i;
            minValue = distance;
        }
        else if(distance == minValue){
            results = realloc(results, ++count*sizeof(int));
            results[count-1]=i;
        }
    }
    int selectedIndex = count > 1 ? (rand()*1.0/RAND_MAX)*(count) : 0;
    int result = results[selectedIndex];
    free(results);
    return result;
}


//Return a value between 0 and 1 according to the distance between the winner neuron and another neuron and a neighborhood factor using p parameters
double neighborhood_function(neuronLocation winner, neuronLocation r, int p, double sigma)
{
    double wp[2]={winner.x, winner.y};
    double rp[2]={r.x, r.y};
    return exp(-(distance_function(wp, rp, 2)/pow(sigma, 2)));
}
//Update the weight vector of a neuron adding for each parameter the difference between entry vector parameter and current vector parameter
//The difference is multiplied by a learning factor (epsilon) and a the result (value between 0 and 1) of neighborhood function
//If no parameter has been updated significantly return 0, 1 otherwise
short updateNeuron(dataVector v, neuronLocation lwinner, neuronLocation lr, somNeuron* weights, somConfig config){
    short flagChange = 0;
    double triggerChange = 0.01;
    double h = neighborhood_function(lwinner, lr, config.p, config.sigma);
    somNeuron r = weights[getIndexFromLocation(lr, config)];
    for(int i=0;i<config.p;i++){
        double delta = config.alpha * h *(v.v[i] - r.w[i]);
        r.w[i] += delta;
        flagChange = delta > triggerChange || delta < -triggerChange ? 1: flagChange;
    }
    return flagChange;
}
//When a winner neuron has been determined for and entry vector, all the neurons are updated the function will return 0 if no neuron has been modified significantly
//1 otherwise
short updateNeurons(dataVector v, int iWinner, somNeuron  *weights, somConfig config)
{
    short flagChange = 0;
    neuronLocation winnerLocation;
    setLocationFromIndex(iWinner, config, &winnerLocation);
    int startx = max(0, winnerLocation.x - config.radius);
    int endx = min(config.map_c - 1, winnerLocation.x + config.radius);
    int starty = max(0, winnerLocation.y - config.radius);
    int endy = min(config.map_r - 1, winnerLocation.y + config.radius);
    for(int i=starty;i<=endy; i++){
        for(int j=startx;j<=endx;j++){
            neuronLocation r = {j,i};
            if(updateNeuron(v, winnerLocation, r, weights, config)){
                flagChange = 1;
            }
        };
    }
    return flagChange;
}

//Return a random value between a boundary
double getRandom(dataBoundary boundary){
    return (((double)rand()/RAND_MAX)*(boundary.max - boundary.min)) + boundary.min;
}
//Initialize SOM weights with random values based on parameters boundaries
void initialize(somNeuron *weights, somConfig config, dataBoundary *boundaries){
    for(int i=0;i<config.nw; i++){
        weights[i].w = (double*)malloc(config.p * sizeof(double));
        for(int j=0;j<config.p;j++){
            weights[i].w[j]= getRandom(boundaries[j]);
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
//Find the winner neuron for an input vector v and upated weights using config parameters
short learn(dataVector v, somNeuron * weights, somConfig config)
{
    int iWinner = fi_function(v, weights, config.nw, config.p);
    return updateNeurons(v, iWinner, weights, config);
}
//Get the neuron weight index that activates with an input vector
int predict(dataVector v, somNeuron * weights, somConfig config)
{
    return fi_function(v, weights, config.nw, config.p);
}

int setMapSize(somConfig *config){
    double ratio = 4.0/3;
    int y = floor(sqrt(config->nw / ratio));
    y-=config->nw%y;
    int x = config->nw/y;
    if(config->nw%(x*y)>0){
        y+=1;
    }
    config->map_r=y;
    config->map_c=x;
    config->radius = (y-1)/2;
}
//Get the weights neurons initialized using an input dataset and a som settings
somNeuron *getsom(dataVector* data, somConfig *config)
{
    if(config->nw <=0){
        config->nw = floor(5 *sqrt(config->n *1.0));
        config->nw -= config->nw%12;
    }
    setMapSize(config);
    dataBoundary boundaries[config->p];
    initializeBoundaries(boundaries, data, *config);
    somNeuron *weights = (somNeuron*)malloc(config->nw *sizeof(somNeuron));
    initialize(weights, *config, boundaries);
    return weights;
}

void append(FILE *fp, somNeuron *weights, somConfig config, long stepid, int scores[]){
    for(int i=0; i<config.nw; i++){
        fputs("[", fp);
        for(int j=0;j<config.p;j++){
            fprintf(fp, "%f", weights[i].w[j]);
            if(j<config.p-1){
                fputs(",", fp);
            }
        }
        fprintf(fp, "];%d;%ld;%d\n", i, stepid, scores == NULL ? 0 :scores[i]);
    }
}

void writeAppend(long stepid, somNeuron *weights, somConfig config, int scores[]){
    FILE * fp;
    fp = fopen("som.data", "a");
    if(fp != NULL){
        append(fp, weights, config, stepid, scores);
    }
    fclose(fp);
}

void write(somNeuron* weights, somConfig config){
    FILE * fp;
    fp = fopen("som.data", "w");
    if(fp != NULL){
        append(fp, weights, config, -1, NULL);
    }
    fclose(fp);
}



