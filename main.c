#include <stdio.h>
#include <stdlib.h>
#include "include/irisdata.h"
#include "include/som.h"


void showDataLine(double data[150][5], int lineIndex){
    printf("%.2f, %.2f, %.2f, %.2f, %.0f\n", data[lineIndex][0], data[lineIndex][1], data[lineIndex][2], data[lineIndex][3], data[lineIndex][4]);
}

void clear_mem(dataVector *data, somNeuron *weights, somConfig config){
    for(int i=0;i<config.n;i++){
        free(data[i].v);
    }
    for(int i=0;i<config.nw;i++){
        free(weights[i].w);
    }
    free(weights);
    free(data);
}

char* getIrisLabel(int index){
        switch(index){
            case 0: return "    Iris-setosa:";
            case 1: return "Iris-versicolor:";
            case 2: return " Iris-virginica:";
        }
        return "          Other:";
}

int getmaxClassValidationIndex(int classIndex, int validations[][150]){
    int result = -1;
    int maxValue = 0;
    for(int i=0;i<3;i++){
       int val = validations[i][classIndex];
       if(val>maxValue){
           result = i;
           maxValue = val;
       }
    }
    return result;
}


int main()
{
    somConfig config;
    config.epsilon = 0.05;
    config.sigma = 0.3;
    dataVector *data = getIrisData(&config);
    somNeuron *weights = getsom(data, &config);
    int episodes = 10;
    for(int i=0;i<episodes;i++){
        printf("Learning episode: %d\n", i+1);
        for(int j=0;j<config.n;j++){
        int ivector = (((double)rand()/RAND_MAX)*(config.nw));
        learn(data[ivector], weights, config);
        }
    }

    double predictions[config.n];
    int scores[config.nw];
     for(int i=0;i<config.nw;i++){
           scores[i]=0;
        }
    for(int i=0;i<config.n;i++){
           int iWinner = predict(data[i], weights, config);
           scores[iWinner]+=1;
           predictions[i] = iWinner;
        }
    int sum=0;
    int classes =0;
    for(int i=0;i<config.nw;i++){
            if(scores[i]){
                classes++;
                printf("%d: %d entries\n", i, scores[i]);
            }

        }

    printf("classes number:%d\n", classes);
    printf("\n");
    int validations[3][config.nw];
    for(int i=0;i<3;i++){
        for(int j=0;j<config.nw;j++){
            validations[i][j]=0;
        }
    }
    for(int i=0;i<3;i++){
        printf("%s",getIrisLabel(i));
        for(int j=0;j<config.nw;j++){
            for(int k=0;k<config.n;k++){
                if(data[k].v[4]== i && predictions[k]==j){
                    validations[i][j]+=1;
                }
            }
            if(validations[i][j]){
                printf("%3d:%2d ", j, validations[i][j]);
            }
        }
        printf("\n");
    }
    for(int i=0;i<3;i++){
        int maxClassSum = 0;
        printf("%s", getIrisLabel(i));
        for(int j=0;j<config.nw;j++){
            if(getmaxClassValidationIndex(j, validations) == i){
                    maxClassSum+=validations[i][j];
                }
        }
        printf("percentage classified (combining winning subgroups):%.2f%%\n", maxClassSum*100/50.0);
    }
                    


    clear_mem(data, weights, config);
}
