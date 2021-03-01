#include "include/som.h"
#include <math.h>
#include <stdlib.h>
// Return min between 2 doubles
double min(double x, double y)
{
    return y < x ? y : x;
}
// Return max between 2 doubles
double max(double x, double y){
    return y > x ? y : x;
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
int fi_function(dataVector v, somNeuron *weights, int n, int p)
{
    int result = -1;
    double minValue = __DBL_MAX__;
    for(int i = 0; i<n; i++){
        double distance = distance_function(v.v, weights[i].w, p);
        minValue = min(distance, minValue);
        if(distance == minValue){
            result = i;
        }
    }
}
//Return a value between 0 and 1 according to the distance between the winner neuron and another neuron and a neighborhood factor using p parameters
double neighborhood_function(somNeuron winner, somNeuron r, int p, double sigma)
{
    return exp(-(distance_function(winner.w, r.w, p)/pow(sigma, 2)));
}
//Update the weight vector of a neuron adding for each parameter the difference between entry vector parameter and current vector parameter
//The difference is multiplied by a learning factor (epsilon) and a the result (value between 0 and 1) of neighborhood function
//If no parameter has been updated significantly return 0, 1 otherwise
short updateNeuron(dataVector v, somNeuron winner, somNeuron r, int p, double epsilon, double sigma){
    short flagChange = 0;
    double h = neighborhood_function(winner, r, p, sigma);
    for(int i=0;i<p;i++){
        double delta = epsilon * h *(v.v[i] - r.w[i]);
        r.w[i] += delta;
        flagChange = delta > 0.001 ? 1: flagChange;
    }
    return flagChange;
}
//When a winner neuron has been determined for and entry vector, all the neurons are updated the function will return 0 if no neuron has been modified significantly
//1 otherwise
short updateNeurons(dataVector v, somNeuron winner, somNeuron  *weights, somConfig config)
{
    short flagChange = 0;
    for(int i=0;i<config.n;i++){
        if(updateNeuron(v, winner, weights[i], config.p, config.epsilon, config.sigma)){
            flagChange = 1;
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
    for(int i=0;i<config.n; i++){
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
    somNeuron winner = weights[fi_function(v, weights, config.n, config.p)];
    return updateNeurons(v, winner, weights, config);
}
//Get the neuron weight index that activates with an input vector
int predict(dataVector v, somNeuron * weights, int n, int p)
{
    return fi_function(v, weights, n, p);
}
//Get the weights neurons initialized using an input dataset and a som settings
somNeuron * getsom(dataVector* data, somConfig config)
{
    dataBoundary boundaries[config.p];
    initializeBoundaries(boundaries, data, config);
    somNeuron *weights = (somNeuron*)malloc(config.n * sizeof(somNeuron));
    initialize(weights, config, boundaries);
    return weights;
}


