#include "common.h"

//The SOM neuron
typedef struct somNeuron{
    double *w;
} somNeuron;

//Learn from an entry vector. The winner neuron is estimated the neurons are updated accordingly
short learn(double *v, somNeuron * weights, somConfig config);
//Predict from an entry vector. Return the winner neuron index
int predict(double *v, somNeuron * weights, int n, int p);
//Get Initialized weights vectors for a specific dataset and a specific config;
somNeuron * getsom(double ** data, somConfig config);