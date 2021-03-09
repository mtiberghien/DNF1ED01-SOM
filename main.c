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
    for(int i=oneD;i<=threeD;i++)
    {
        config->dimension=i;
        weights = (somNeuron**)getsom(data, config);

        somScoreResult* result = getscore(data, weights, config);
        int activatedNodes = 0;
        displayConfig(config);
        displayScore(result, config);
        
        resetConfig(config);
        clear_mem(weights, result, config);
    }
            
    clear_data(data, config);
    clear_config(config);

    
}
