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
    //1 if the neuron is considered as stabilized, 0 otherwise
    short isStabilized;
    //Stores the sum of updates made to a neuron during one episode
    double* updates;
    //Stores the indexes of entries that activated the neuron during one episode
    int* entries;
    //Stores the number of entries that activated the neuron during one episode
    int ec;
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
    // The neurons should stabilize automatically (speed related to data volume and changeTrigger). This parameter force the end of learning after n episodes.
    // value <0 means no limit -> 1000 by default
    int maxEpisodes;
    // 1 if input data should be normalized, 0 otherwise -> 1 by default
    short normalize;
    // 1 if want to use neighbour method (faster but might split some classes if to early) to find the winner
    short useNeighboursMethod;
    // Number of episodes to wait after starting to use neighbours method
    int useNeighboursTrigger;
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