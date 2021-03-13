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
    dataVector *data_raw = getMNISTData(config, 10000);
    dataVector data[500];
    int proposed[config->n];
    for (int i = 0; i < config->n; i++)
    {
        proposed[i]=i;
    }
    int n = config->n;
    for (int i = 0; i < 500; i++)
    {
        int index = rand()%n;
        data[i]= data_raw[proposed[index]];
        proposed[index]=--n;
    }
    config->n=500;                                                      
    
    dataBoundary boundaries[config->p];
    calculateBoundaries(data, boundaries, config);
    int maxClasses = 0;
    int activatedNodes = 0;

    for(int i=twoD;i<=twoD;i++)
    {
        config->dimension = i;
        config->useNeighboursTriggerRate = 0.1;
        config->map_r=40;
        config->map_c=40;
         weights = getsom(data, config,boundaries, 0);
        somScoreResult* result = getscore(data, weights, config);
        displayConfig(config);
        displayScore(result, config);
        write(weights, config, result);    
        clear_mem(weights, result, config);
        resetConfig(config);
    }
    config->n=1000;
    clear_data(data_raw, config);
    clear_config(config);    
}
