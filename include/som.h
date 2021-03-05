#include "common.h"

//The SOM neuron
typedef struct somNeuron{
    //stores the vector
    double* v;
    struct somNeuron** neighbours;
    //neighbours count
    int nc;
    int b;
    int r;
    int c;
} somNeuron;

typedef struct somScore
{
    int iclass;
    short hasMultipleResult;
    int* scores;
}somScore;

typedef struct somScoreResult
{
    int nclasses;
    void* scores;
}somScoreResult;

//Get default settings for SOM
somConfig* getsomDefaultConfig();
//Get Initialized weights vectors for a specific dataset and a specific config;
void* getsom(dataVector* data, somConfig *config);
//Append the current state of SOM neurons (one line by neuron) using provided step id
void writeAppend(long stepid, somNeuron* weights, somConfig* config, int scores[]);
void write(somNeuron* weights, somConfig* config);
somScoreResult* getscore(dataVector* data, void* weights, somConfig *config);
void clear_mem(dataVector* data, void* weights,  somScoreResult* score,somConfig *config);
