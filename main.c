#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "include/irisdata.h"
#include "include/parkinsondata.h"
#include "include/mnistdata.h"
#include "include/som.h"
#include "include/common.h"
#include <time.h>

void printTaskDuration(time_t begin, time_t end)
{
    long dif = (long)difftime(end,begin);
    int hours = dif/3600;
    int minutes = (dif-(3600*hours))/60;
    int seconds = (dif-(3600*hours)-(60*minutes));

    printf("Task accomplished in %d hour(s), %d minute(s) %d seconde(s).\n", hours,  minutes, seconds);
}

void testIris()
{
    somConfig *config = getsomDefaultConfig();
    config->alpha=0.05;
    config->epochs = 2000;
    void* weights;
    dataVector* data = getIrisData(config);
    dataBoundary boundaries[config->p];
    calculateBoundaries(data, boundaries, config);                                                   
    for (int i = oneD; i <=threeD;i++)
    {
        config->dimension = i;
        weights = getTrainedSom(data, config,boundaries, 0);
        somScoreResult* result = getscore(data, weights, config, 1, 1);
        displayConfig(config);
        displayScore(result, config);
        clear_mem(weights, result, config);
        resetConfig(config); 
    }
        
    clear_data(data, config);
    clear_config(config);    
}

void displayMNISTNeuron(somScore** scores, somNeuron** weights, somConfig* config, int class)
{
    int maxi =-1;
    int maxj =-1;
    int maxEntries=0;
    for(int i=0;i<config->map_r;i++)
    {
        for(int j=0;j<config->map_c;j++)
        {
            somScore s = scores[i][j];
            if(s.maxClass == class)
            {
                if(s.maxClasstotalEntries > maxEntries)
                {
                    maxi=i;
                    maxj=j;
                    maxEntries = s.maxClasstotalEntries;
                }
            }
        }
    }
    if(maxi>=0 && maxj >=0)
    {
        int size = sqrt(config->p);
        printf("Class %d\n", class);
        for(int i=0;i<size;i++)
        {
            int base = i*size;
            for(int j=0;j<size;j++)
            {
                printf("%3d ", (int)(255*weights[maxi][maxj].v[base+j]));
            }
            printf("\n");
        }
        
    }
}

void testMNIST_train()
{
    char* filename = "trained_som.txt";
    somConfig *config = getsomDefaultConfig();
    void* weights;
    config->normalize=1;
    int raw_limit = -1;
    int sample_limit = 500;
    time_t begin = time(NULL);
    printf("Reading data: ");
    fflush(stdout);
    dataVector *data_raw = getMNISTData(config, raw_limit,0);
    raw_limit = config->n;
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
    printf("Created Sample with %d lines among %d\n", sample_limit, config->n);
    config->n=sample_limit;
    printf("Calculating boundaries: ");
    fflush(stdout);
    dataBoundary boundaries[config->p];
    calculateBoundaries(data, boundaries, config); 
    printf("Done\n");                                                    
    config->alpha=0.05;
    config->map_r=40;
    config->map_c=40;
    weights = getTrainedSom(data, config,boundaries, 0);
    somScoreResult* result = getscore(data, weights, config, 1, 1);   
    displayConfig(config);
    displayScore(result, config);
    for(int i=0;i<10;i++)
    {
        displayMNISTNeuron((somScore**)result->scores, (somNeuron**)weights, config, i);
    } 
    saveSom(weights, config, filename);
    writeSomHisto("som_mnist.data", weights, config, result);
    clear_mem(weights, result, config);
        
    config->n=raw_limit;
    clear_data(data_raw, config);
    clear_config(config);
    time_t end = time(NULL);
    printTaskDuration(begin,end);
}

void testMNIST_test()
{
    time_t begin = time(NULL);
    char* filename = "trained_som.txt";
    somConfig *config = getsomDefaultConfig();
    void* weights = loadSom(filename, config);
    displayConfig(config);
    config->normalize=1;
    int raw_limit = -1;
    int sample_limit = 500;
    printf("Reading data: ");
    fflush(stdout);
    dataVector *data_raw = getMNISTData(config, raw_limit,1);
    printf("Done (%d lines read)\n", config->n);
    raw_limit = config->n;
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
    printf("Created Sample with %d lines among %d\n", sample_limit, config->n);
    config->n=sample_limit;
    somScoreResult* result = getscore(data, weights, config,0, 0);
    
    int success = 0;
    somScore** scores = (somScore**)result->scores;
    for(int i=0;i<sample_limit; i++)
    {
        somNeuron winner = *data[i].lastWinner;
        if(data[i].class == scores[winner.r][winner.c].maxClass)
        {
            success++;
        }
    }

    printf("Tested %d data with a success rate of %.2f%%\n", sample_limit, 100*(double)success/sample_limit);

    clear_mem(weights, result, config);
        
    config->n=raw_limit;
    clear_data(data_raw, config);
    clear_config(config);
    time_t end = time(NULL);
    printTaskDuration(begin, end);   
}

void testParkinson()
{
    somConfig *config = getsomDefaultConfig();
    void* weights;
    dataVector* data = getParkinsonsData(config);
    dataBoundary boundaries[config->p];
    calculateBoundaries(data, boundaries, config);
    config->nw=100;
    config->alpha = 0.01;                                                
    weights = getTrainedSom(data, config,boundaries, 0);
    somScoreResult* result = getscore(data, weights, config, 1, 1);
    displayConfig(config);
    displayScore(result, config);
    clear_mem(weights, result, config); 
    clear_data(data, config);
    clear_config(config); 
}

int main()
{
    testMNIST_test();

}
