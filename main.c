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
    config->normalize=0;
    dataVector *data = getIrisData(config);
    /* dataVector data[500];
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
    config->n=500;        */                                               
    for (size_t i = oneD; i <= threeD; i++)
    {
        dataBoundary boundaries[config->p];
        calculateBoundaries(data, boundaries, config);
        int activatedNodes = 0;
        config->alpha = 0.1;
        config->dimension=i;
        weights = getsom(data, config,boundaries, 0);
        somScoreResult* result = getscore(data, weights, config);
        displayConfig(config);
        displayScore(result, config);  
        
        clear_mem(weights, result, config);
        resetConfig(config); 
    }
        
    //config->n=10000;
    clear_data(data, config);
    clear_config(config);    
}
