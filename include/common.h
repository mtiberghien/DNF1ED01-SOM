#ifndef COMMON_H
#define COMMON_H

//The SOM neuron
typedef struct somNeuron{
    //stores the vector
    double* v;
    struct somNeuron** neighbours;
    //neighbours count
    int nc;
    //block index (used in 3D map only)
    int b;
    //row index (used in 2D and 3D map)
    int r;
    //column index (used in all dimensions)
    int c;
    //known class (from score) for trained neurons (used for validation)
    int class;
    //percentage of confidence for the known class
    double confidence;
} somNeuron;

// Defines a data vector
typedef struct dataVector{
    //stores the vector
    double *v;
    //stores the norm
    double norm;
    //stores optionnaly the known label for further validation
    int class;
    //Stores the last winner for the data entry
    somNeuron* lastWinner;
} dataVector;

// Defines the dimension of map projection
typedef enum mapDimension{
    defaultD = 0,
    oneD = 1,
    twoD = 2,
    threeD = 3
} mapDimension;

// Defines the neuron distribution strategy
typedef enum initialDistribution{
    //Will define random values around the mean of each paramter
    usingMeans = 0,
    //Will define random values in the each parameter domain
    usingMinMax = 1
} initialDistribution;

//SOM settings
typedef struct somConfig{

    //Number of entries -> Should be initialized when reading data
    int n;
    //Number of parameters -> Should be initialized when reading data
    int p;
    //Learning factor (0 to 1)-> 0.99 by default
    double alpha;
    //Neighborhood factor (0 to 1)-> 0.99 by default
    double sigma;
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
    //The neighborhood initial percentage coverage (0 to 1) -> 0.6 by default
    double initialPercentCoverage;
    // 1 if input data should be normalized, 0 otherwise -> 1 by default
    short normalize;
    //Number of epochs (one epoch = learn with one input vector) -> 1 by default
    long epochs;
    //The initializaztion strategy for weigts -> means by default
    initialDistribution distribution;
} somConfig;

// Define a value area
typedef struct dataBoundary{
    double min;
    double max;
    double mean;
} dataBoundary;
#endif

//get the min value for 2 doubles
double min(double x, double y);
//get the min value for 2 doubles
double max(double x, double y);
//Get the norm (square root of sum of square of p elements)
double getNorm(double*v, int p);
//Normalize the vector using p parameters -> return the norm
double normalizeVector(double* v, int p);
//Denormalize the vector using p parameters and the norm
void denormalizeVector(double* v, int p, double norm);
//Free data memory
void clear_data(dataVector* data, somConfig* config);
//Calculate the boundaries for a dataset calculating the mean for each parameter and boudaries around the mean
//These boundaries will be then used by getSom method to set random initial position for the neurons
void calculateBoundaries(dataVector* data, dataBoundary* boundaries, somConfig* config);