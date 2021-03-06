#ifndef COMMON_H
#define COMMON_H

// Defines a data vector
typedef struct dataVector{
    //stores the vector
    double *v;
    //stores the norm
    double norm;
    //stores optionnaly the known label for further validation
    int class;
} dataVector;

typedef struct neuronLocation{
    int x;
    int y;
} neuronLocation;
// Defines the dimension of map projection
typedef enum mapDimension{
    defaultD = 0,
    oneD = 1,
    twoD = 2,
    threeD = 3
} mapDimension;

//SOM settings
typedef struct somConfig{
    //Number of entries -> Should be initialized when reading data
    int n;
    //Number of parameters -> Should be initialized when reading data
    int p;
    //Learning factor (0 to 1)-> 0.99 by default
    double alpha;
    //Each episode (all data processed once) alpha will be multiplied by this value (should be between 0 and 1) -> default 0.99
    double alphaDecreaseRate;
    //Neighborhood factor (0 to 1)-> 0.99 by default
    double sigma;
    //Each episode (all data processed once) sigma will be multiplied by this value (should be between 0 and 1) -> default 0.99
    double sigmaDecreaseRate;
    //Number of weights -> if not provided will be calculated as 5*sqrt(n)
    int nw;
    //Map dimension (1, 2, or 3) -> 2 by default
    mapDimension dimension;
    //Number of blocks (3D map only) -> calculated if not provided
    int map_b;
    //Number of rows (2D and 3D map) -> calculated if not provided
    int map_r;
    //Number of columns (equals number of nodes for 1D map) -> calculated if not provided
    int map_c;
    //Neighborhood radius (the algorithm will look for ((2*radius)+1)^dimensions neighboors (winner included)
    // -> calculated if not provided
    int radius;
    //Each n episodes (each data processed once) of learning the radius will decrease -> 3 by default (every 3 episode radius will decrease)
    int radiusDecreaseRate;
    //A neuron will be considered as stable if the update absolute delta fore each parameter is smaller than the value -> 0.001 by default
    double stabilizationTrigger;
    //The neighborhood initial percentage coverage (0 to 1) -> 0.6 by default
    double initialPercentCoverage;
    // 1 if the map shouldn't have borders and consider that last and first element of a dimension are neighbours, 0 otherwhise -> default = 0
    short isMapClosed;
    // The neurons should stabilize automatically (speed related to data volume and changeTrigger). This parameter force the end of learning after n episodes.
    // value <0 means no limit -> 1000 by default
    int maxEpisodes;
} somConfig;

// Define a value area
typedef struct dataBoundary{
    double min;
    double max;
} dataBoundary;
#endif

//get the min value for 2 doubles
double min(double x, double y);
//get the min value for 2 doubles
double max(double x, double y);
//Normalize the vector using p parameters -> return the norm
double normalizeVector(double* v, int p);
//Denormalize the vector using p parameters and the norm
void denormalizeVector(double* v, int p, double norm);