#include "common.h"

//The SOM neuron
typedef struct somNeuron{
    double *w;
    double norm;
} somNeuron;

//Learn from an entry vector. The winner neuron is estimated the neurons are updated accordingly
short learn(dataVector v, somNeuron * weights, somConfig config);
//Predict from an entry vector. Return the winner neuron index
int predict(dataVector v, somNeuron * weights, somConfig config);
//Get Initialized weights vectors for a specific dataset and a specific config;
somNeuron *getsom(dataVector *data, somConfig *config);
//Append the current state of SOM neurons (one line by neuron) using provided step id
void writeAppend(long stepid, somNeuron* weights, somConfig config, int scores[]);
void write(somNeuron* weights, somConfig config);
