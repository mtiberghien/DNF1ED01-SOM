#include <stdio.h>
#include <stdlib.h>
#include "include/irisdata.h"
#include "include/som.h"


void showDataLine(double data[150][5], int lineIndex){
    printf("%.2f, %.2f, %.2f, %.2f, %.0f\n", data[lineIndex][0], data[lineIndex][1], data[lineIndex][2], data[lineIndex][3], data[lineIndex][4]);
}

void clear_mem(double ** data, somNeuron * weights, int n){
    for(int i=0;i<n;i++){
        free(weights[i].w);
        free(data[i]);
    }
    free(weights);
    free(data);
}


int main()
{
    somConfig config;
    double **data = (double**)getIrisData(&config);
    somNeuron *weights = getsom(data, config);

    for(int i=0;i<config.p;i++){
        printf("%d: %f ", i, weights[3].w[i]);
    }
    printf("\n");

    clear_mem(data, weights, config.n);
}
