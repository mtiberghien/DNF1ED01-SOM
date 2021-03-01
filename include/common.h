#ifndef COMMON_H
#define COMMON_H

typedef struct dataVector{
    double *v;
} dataVector;

//SOM settings
typedef struct somConfig{
    //Number of entries
    int n;
    //Number of parameters
    int p;
    //Learning factor
    double epsilon;
    //Neighborhood factor
    double sigma;
} somConfig;

typedef struct dataBoundary{
    double min;
    double max;
} dataBoundary;
#endif

//get the min value for 2 doubles
double min(double x, double y);
double max(double x, double y);