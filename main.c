#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "include/irisdata.h"
#include "include/som.h"

int main()
{
    somNeuron** weights;
    somConfig *config = getsomDefaultConfig();
    dataVector *data = getIrisData(config);
    config->normalize = 0;
    config->stabilizationTrigger = 0.4;
    config->alpha = 0.99;
    config->sigma = 0.90;
    for(int i=oneD;i<=threeD;i++)
    {
        config->dimension=i;
        weights = (somNeuron**)getsom(data, config);

        somScoreResult* result = getscore(data, weights, config);
        int activatedNodes = 0;
        displayScore(result, config);
        
        resetConfig(config);
        clear_mem(weights, result, config);
    }
            
    clear_data(data, config);
    clear_config(config);

    
}
