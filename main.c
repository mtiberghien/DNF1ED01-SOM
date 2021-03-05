#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "include/irisdata.h"
#include "include/som.h"


char* getIrisLabel(int index){
        switch(index){
            case 0: return "    Iris-setosa:";
            case 1: return "Iris-versicolor:";
            case 2: return " Iris-virginica:";
        }
        return "          Other:";
}

char* getTerminalColorCode(int index){
    switch(index){
        //red
        case 1: return ";31m";
        //yellow
        case 2: return ";33m";
        //blue
        case 3: return ";34m";
    }
    //default
    return "m";
}

int main()
{
    
    somNeuron** weights;
    somConfig *config = getsomDefaultConfig();
    dataVector *data = getIrisData(config);

            
    weights = (somNeuron**)getsom(data, config);

    somScoreResult* result = getscore(data, weights, config);
    somScore** score = result->scores;
    int activatedNodes = 0;
    printf("SOM Settings:\n%19s: %d\n%19s: %d\n%19s: %d\n%19s: %d\n%19s: %d\n%19s: %.2f\n%19s: %.2f\n\n", "Entries", config->n,
                    "Neurons", config->nw ,"Map rows", config->map_r, "Map columns", config->map_c, "Radius", config->radius,
                    "Learning rate", config->alpha, "Neighborhood factor", config->sigma); 
    for(int i=0;i<config->map_r;i++)
    {
        for(int j=0;j<config->map_c;j++)
        {
            int c = score[i][j].iclass + 1;
            if(score[i][j].hasMultipleResult >=0)
            {
                activatedNodes++;
            }
            printf("\033[0%s", getTerminalColorCode(c));
            printf("%d ", c);
            printf("\033[0m");
        }
        printf("\n");
    }
    printf("Activated nodes:%d\n", activatedNodes);
    

    clear_mem(data, weights, result, config);

    
    
                //write(weights, config);
                /* */
                
                
                /* for(int i=0;i<config.n;i++)
                {
                    predictions[i]=0;
                }

                for(int i=0;i<config.nw;i++)
                {
                    scores[i]=0;
                }

                for(int i=0;i<config.n;i++)
                {
                    somNeuron winner = *predict(&data[i], weights, &config);
                    scores[winner.c]+=1;
                    predictions[i] = winner.c;
                }

                int sum=0;
                int classes =0;
                for(int i=0;i<config.nw;i++)
                {
                    if(scores[i]){
                        classes++;
                        //printf("%d: %d entries\n", i, scores[i]);
                    }

                }

                if(classes >= maxClasses)
                {
                    maxClasses = classes;
                    printf("alpha:%.2f, sigma:%.2f, classes:%d\n", alpha, sigma, classes);
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
                if(data[k].class == i && predictions[k]==j){
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
            int result = -1;
            int maxValue = 0;
            for(int i=0;i<3;i++){
            int val = validations[i][j];
            if(val>maxValue){
                result = i;
                maxValue = val;
            }
            }
            if(result == i){
                    maxClassSum+=validations[i][j];
                }
        }
        printf("percentage classified (combining winning subgroups):%.2f%%\n", maxClassSum*100/50.0);
    }
    int map[config.map_r][config.map_c];
    for(int i=0;i<config.map_r;i++){
        for(int j=0;j<config.map_c;j++){
            map[i][j]=0;
        }
    }
    for(int i=0;i<config.n;i++){
        int real_class = data[i].class;
        int neuron_index = predictions[i];
        map[neuron_index/config.map_c][neuron_index%config.map_c] = real_class+1;
    }
    printf("\nNeuron Map:\n");
    for(int i=0;i<config.map_r;i++){
        for(int j=0;j<config.map_c;j++){
            printf("\033[0%s", getTerminalColorCode(map[i][j]));
            printf("%d ", map[i][j]);
            printf("\033[0m");
        }
        printf("\n");
    }
                } */    

    

    /* FILE * fp;
    fp = fopen("predictions.data", "w");
    if(fp != NULL){
       for(int i=0;i<config.map_r;i++){
        for(int j=0;j<config.map_c;j++){
            fprintf(fp, "%d;%d\n", (i*config.map_c + j), map[i][j]);
        }
    } 
    }
    fclose(fp); */



    
}
