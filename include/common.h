#ifndef COMMON_H
#define COMMON_H

typedef struct dataVector{
    double *v;
    double norm;
    int class;
} dataVector;

typedef struct neuronLocation{
    int x;
    int y;
} neuronLocation;

//SOM settings
typedef struct somConfig{
    //Number of entries
    int n;
    //Number of parameters
    int p;
    //Learning factor
    double alpha;
    //Neighborhood factor
    double sigma;
    //Number of weights
    int nw;
    int map_r;
    int map_c;
    int radius;
} somConfig;

typedef struct dataBoundary{
    double min;
    double max;
} dataBoundary;
#endif

//get the min value for 2 doubles
double min(double x, double y);
double max(double x, double y);
double normalizeVector(double* v, int p);
void denormalizeVector(double* v, int p, double norm);