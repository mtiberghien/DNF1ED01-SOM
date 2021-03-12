#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "include/irisdata.h"
#include "include/parkinsondata.h"
#include "include/mnistdata.h"
#include "include/som.h"
#include "include/common.h"

int main()
{
    void* weights;
    somConfig *config = getsomDefaultConfig();
    dataVector *data_raw = getMNISTData(config);
    dataVector data[500];

    for(int i =0;i<500;i++)
    {
        int index = rand()%config->n;
        data[i]=data_raw[index];
    }
    config->n=500;
    dataBoundary boundaries[config->p];
    calculateBoundaries(data, boundaries, config);
 
    for(int i=twoD;i<threeD;i++)
    {
        config->dimension = i;
        config->stabilizationTrigger = 0.001;
        config->alpha = 0.1;
        config->alphaDecreaseRate=0.99;
        config->sigma = 1;
        config->sigmaDecreaseRate=0.9;
        config->radiusDecreaseRate = 5;
        config->initialPercentCoverage = 0.65;
        config->maxEpisodes = 1000;
        config->useNeighboursMethod = 1;
        config->useNeighboursTrigger = 20;
         weights = getsom(data, config,boundaries, 0);
        somScoreResult* result = getscore(data, weights, config);
        displayConfig(config);
        displayScore(result, config);
        write(weights, config, result);
        clear_mem(weights, result, config);
        resetConfig(config);
    }
    
    clear_data(data_raw, config);
    clear_config(config);    
}
