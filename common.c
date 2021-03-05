#include <math.h>

// Return min between 2 doubles
double min(double x, double y)
{
    return y < x ? y : x;
}
// Return max between 2 doubles
double max(double x, double y){
    return y > x ? y : x;
}

double normalizeVector(double* v, int p){
    double norm=0;
    for(int i = 0;i<p;i++){
        norm+=pow(v[i],2);
    }

    norm = sqrt(norm);
    for(int i= 0;i<p;i++){
        v[i]/=norm;
    }
    return norm;
}

void denormalizeVector(double* v, int p, double norm){
    for(int i=0;i<p;i++){
        v[i] *= norm;
    }
}