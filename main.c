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
    dataVector *data = getParkinsonsData(config);
    dataBoundary boundaries[config->p];
    calculateBoundaries(data, boundaries, config);
    int maxClasses = 0;
    int activatedNodes = 0;
     for(double alpha=0.1;alpha<=1;alpha+=0.1)
    {
        for(double alphaRate=0.1;alphaRate<=1;alphaRate+=0.1)
        {
            for(double sigma=0.1;sigma<=1;sigma+=0.1)
            {
                for(double sigmaRate=1;sigmaRate<=1;sigmaRate+=0.1)
                {
                    config->alpha = alpha;
                    config->alphaDecreaseRate = alphaRate;
                    config->sigma = sigma;
                    config->sigmaDecreaseRate = sigmaRate;

                    weights = getsom(data, config,boundaries, 1);
                    somScoreResult* result = getscore(data, weights, config);
                    if(result->nActivatedNodes >= maxClasses)
                    {
                        maxClasses = result->nActivatedNodes;
                        printf("alpha: %.2f, alphaRate: %.2f, sigma: %.2f, sigmaRate: %.2f\n", alpha, alphaRate, sigma, sigmaRate);
                        displayScore(result, config);
                    }
                    
                    
                    clear_mem(weights, result, config);
                    resetConfig(config);
                }
            }
        }
    } 
    /* for(int i=oneD;i<=threeD;i++)
    {
        config->dimension = i;
         weights = getsom(data, config,boundaries, 0);
        somScoreResult* result = getscore(data, weights, config);
        displayConfig(config);
        displayScore(result, config);
        
        clear_mem(weights, result, config);
        resetConfig(config);
    } */
    
    clear_data(data, config);
    clear_config(config);    
}
