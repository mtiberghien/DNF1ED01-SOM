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
    config->normalize=1;
    int raw_limit = 10000;
    int sample_limit = 1000;
    printf("Reading data: ");
    fflush(stdout);
    dataVector *data_raw = getMNISTData(config, raw_limit);
    printf("Done (%d lines read)\n", config->n);
    printf("Creating Sample: ");
    fflush(stdout);
    dataVector data[sample_limit];
    int proposed[config->n];
    for (int i = 0; i < config->n; i++)
    {
        proposed[i]=i;
    }
    int n = config->n;
    for (int i = 0; i < sample_limit; i++)
    {
        int index = rand()%n;
        data[i]= data_raw[proposed[index]];
        proposed[index]=--n;
    }
    config->n=sample_limit;
    printf("Created Sample with %d lines\n", config->n);
    printf("Calculating boundaries: ");
    fflush(stdout);
    dataBoundary boundaries[config->p];
    calculateBoundaries(data, boundaries, config); 
    printf("Done\n");                                                    
    for (int i = twoD; i <=twoD;i++)
    {
        config->dimension = i;
        config->alpha = 0.05;
        config->map_r = 40;
        config->map_c = 40;
        config->epochs = sample_limit * 20;
        weights = getsom(data, config,boundaries, 0);
        somScoreResult* result = getscore(data, weights, config);
        displayConfig(config);
        displayScore(result, config);  
        writeSom(weights, config, result);
        clear_mem(weights, result, config);
        resetConfig(config); 
    }
        
    config->n=raw_limit;
    clear_data(data_raw, config);
    clear_config(config);    

}
