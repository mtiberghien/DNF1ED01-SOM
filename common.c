#include <math.h>
#include "include/common.h"
#include <stdlib.h>

// Return min between 2 doubles
double min(double x, double y)
{
    return y < x ? y : x;
}
// Return max between 2 doubles
double max(double x, double y){
    return y > x ? y : x;
}
//Calculate the norm (the square root of the sum of square o vector elements) and divide the vector by the norm
//Returns the norm
double normalizeVector(double* v, int p){
    double norm=0;
    for(int i = 0;i<p;i++){
        norm+=pow(v[i],2);
    }

    norm = sqrt(norm);
    if(norm!=0)
    {
        for(int i= 0;i<p;i++){
            v[i]/=norm;
        }
    }
    return norm;
}
//Mutliply each element (p is the numbe of elements) of the vector by the norm
void denormalizeVector(double* v, int p, double norm){
    for(int i=0;i<p;i++){
        v[i] *= norm;
    }
}
//Free data dynamically allocated memory
void clear_data(dataVector* data, somConfig* config)
{
    if(data)
    {
        for(int i = 0; i<config->n;i++)
        {
            free(data[i].v);
        }
    }
}