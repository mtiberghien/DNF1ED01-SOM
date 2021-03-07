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
    double norm;
} somNeuron;

typedef struct somScore
{
    int totalEntries;
    int maxClass;
    int secondClass;
    //-1 means no class related, 0 one class related, 1 many class related
    short status;
    int* scores;
}somScore;

typedef struct somScoreResult
{
    int nClasses;
    int nActivatedNodes;
    void* scores;
}somScoreResult;

//Get default settings for SOM
somConfig* getsomDefaultConfig();
//Get Initialized weights vectors for a specific dataset and a specific config;
void* getsom(dataVector* data, somConfig *config);
//Append the current state of SOM neurons (one line by neuron) using provided step id and score
void writeAppend(long stepid, somNeuron *weights, somConfig* config, somScoreResult* scoreResult);
void write(somNeuron* weights, somConfig* config);
somScoreResult* getscore(dataVector* data, void* weights, somConfig *config);
//Clear som objects (weighs and score)
void clear_mem(void* weights,  somScoreResult* score,somConfig *config);
//Clear somConfig object
void clear_config(somConfig* config);
void displayConfig(somConfig* config);
void displayScore(somScoreResult* scoreResult, somConfig* config);
void resetConfig(somConfig* config);
