#include "common.h"

//The SOM neuron
typedef struct somNeuron{
    //stores the vector
    double* v;
    struct somNeuron** neighbours;
    //neighbours count
    int nc;
    //block index (used in 3D map only)
    int b;
    //row index (used in 2D and 3D map)
    int r;
    //column index (used in all dimensions)
    int c;
} somNeuron;

//scoring statistics for one neuron
typedef struct somScore
{
    //number of entries that activate a neuron
    int totalEntries;
    //winner class (class with most entries)
    int maxClass;
    //second class with most entries (used to detect ambiguous classification)
    int secondClass;
    //-1 means no class related, 0 one and one class only related, 1 more thant one class related
    short status;
    //number of entries for each classes
    int* scores;
    //index of each entry that activates the node
    int* entries;
}somScore;

//score result for all neurons
typedef struct somScoreResult
{
    //total number of classes (read from data)
    int nClasses;
    //number of nodes that activate at least on entry
    int nActivatedNodes;
    //score map (according to the dimension can be 1D, 2D or 2D map)
    void* scores;
}somScoreResult;

//Get default settings for SOM
somConfig* getsomDefaultConfig();
//Get Initialized weights vectors for a specific dataset and a specific config;
void* getsom(dataVector* data, somConfig *config);
//Append the current state of SOM neurons (one line by neuron) using provided step id and score
void writeAppend(long stepid, somNeuron *weights, somConfig* config, somScoreResult* scoreResult);
//Write the neurons as csv
void write(somNeuron* weights, somConfig* config);
somScoreResult* getscore(dataVector* data, void* weights, somConfig *config);
//Clear som objects (weighs and score)
void clear_mem(void* weights,  somScoreResult* score,somConfig *config);
//Clear somConfig object
void clear_config(somConfig* config);
//Display the config to the terminal output
void displayConfig(somConfig* config);
//Display the score map to the terminal output
void displayScore(somScoreResult* scoreResult, somConfig* config);
//reset the number of neurons, blocks, rows, columns and radius so it can be configured automatically
void resetConfig(somConfig* config);
