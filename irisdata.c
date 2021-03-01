#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "include/common.h"

//Get flower class as double for later analysis
double getIrisClass(char* word){
    if(!strcmp(word,"Iris-setosa")){
        return 0;
    }
    else if(!strcmp(word, "Iris-versicolor")){
        return 1;
    }
    else if(!strcmp(word, "Iris-virginica")){
        return 2;
    }
    return -1;
}

//Read iris.data file ans set values in a double array with 5 columns: Sepal.Length, Sepal.Width, Petal.Length, Petal.Width, Class
void * getIrisData(somConfig *config){
   FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    double ** data = (double**)malloc(sizeof(double*));
    fp = fopen("iris.data", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    int ln = 0;
    while ((read = getline(&line, &len, fp)) != -1) {
        if(read>1){
            data = (double**) realloc(data, (ln+1) * sizeof(double*));
            char delim[]=",";
            int column = 0;
            char *ptr = strtok(line, delim);
            data[ln] = (double*)malloc(5 * sizeof (double));
            while(ptr != NULL && column < 5)
            {
                if(column < 4){
                    data[ln][column] = strtod(ptr, NULL);
                }
                else if(column == 4){
                    ptr = strtok(ptr, "\n");
                    data[ln][column] = getIrisClass(ptr);
                }
                ptr = strtok(NULL, delim);
                column++;
            }
            ln++;
            if(ptr){
                free(ptr);
            }
        }
    }

    fclose(fp);
    if (line)
        free(line);
    config->n=ln;
    config->p=5;   
    return data;
}

