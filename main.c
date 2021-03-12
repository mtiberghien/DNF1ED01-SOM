#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "include/irisdata.h"
#include "include/parkinsondata.h"
#include "include/som.h"
#include "include/common.h"

int main()
{
    void* weights;
    somConfig *config = getsomDefaultConfig();
    dataVector *data = getIrisData(config);
    dataBoundary boundaries[config->p];
    calculateBoundaries(data, boundaries, config);
    int maxClasses = 0;
    int activatedNodes = 0;

    for(int i=oneD;i<=threeD;i++)
    {
        config->dimension = i;
         weights = getsom(data, config,boundaries, 0);
        somScoreResult* result = getscore(data, weights, config);
        displayConfig(config);
        displayScore(result, config);
        
        clear_mem(weights, result, config);
        resetConfig(config);
    }
    
    clear_data(data, config);
    clear_config(config);    
}
