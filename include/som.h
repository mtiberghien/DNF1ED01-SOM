#include "common.h"


//scoring statistics for one neuron
typedef struct somScore
{
    //number of entries that activate a neuron
    int totalEntries;
    //winner class (class with most entries)
    int maxClass;
    int maxClasstotalEntries;
    //second class with most entries (used to detect ambiguous classification)
    int secondClass;
    //-1 means no class related, 0 one and one class only related, 1 more thant one class related
    int secondClasstotalEntries;
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
    //global mean of confidence for class prediction weighted by the percentage of activated nodes
    double confidence;
}somScoreResult;

typedef struct somPrediction
{
    int class;
    double confidence;
}somPrediction;

//Get default settings for SOM
somConfig* getsomDefaultConfig();
//Get Initialized and trained weights vectors for a specific dataset specific config and data boundaries. With silent set to 1, no output on the Terminal
void* getTrainedSom(dataVector* data, somConfig *config, dataBoundary* boundaries, short silent);
//Append the current state of SOM neurons (one line by neuron) using provided step id and score
void writeSomHistoAppend(char* filename, long stepid, void *weights, somConfig* config, somScoreResult* scoreResult);
//Write the neurons as csv
void writeSomHisto(char* filename, void* weights, somConfig* config, somScoreResult* scoreResult);
somScoreResult* getscore(dataVector* data, void* weights, somConfig *config, short useFromNeighbours, short silent);
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
void saveSom(void* weights, somConfig* config, char* filename);
void* loadSom(char* filename, somConfig* config);
//Train SOM weights with provided data and config
void fit(dataVector* data, void* weights, somConfig* config, short silent);
//Predict test data using trained weights
somPrediction* predict(dataVector* data, void* weights, somConfig* config, short silent);
