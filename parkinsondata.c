#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "include/common.h"

dataVector* getParkinsonsData(somConfig *config)
{
    FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;
    dataVector * data = (dataVector*)malloc(sizeof(dataVector));
    fp = fopen("parkinsons.data", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);
    int ln = 0;
    config->p=22;
    //skip headers
    getline(&line, &len, fp); 
    while ((read = getline(&line, &len, fp)) != -1) {
        if(read>1){
            data = (dataVector*) realloc(data, (ln+1) * sizeof(dataVector));
            char delim[]=",";
            int column = 0;
            int i=0;
            char *ptr = strtok(line, delim);
            data[ln].v = (double*)malloc(config->p * sizeof (double));
            while(ptr != NULL && column < config->p+2)
            {
                //First column is irrelevant (the name of patient and number of record) for classification
                if(column != 0)
                {
                    //class index for status is 17
                    if(column != 17){
                        data[ln].v[i] = strtod(ptr, NULL);
                        i++;
                    }
                    else
                    {
                        data[ln].class = atoi(ptr);
                    }
                }
                ptr = strtok(NULL, delim);
                column++;
            }
            if(config->normalize)
            {    
                data[ln].norm = normalizeVector(data[ln].v, config->p);
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

    return data;
}