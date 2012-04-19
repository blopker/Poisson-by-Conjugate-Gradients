// Linear algebra functions for Poisson Solver

#include "linalg.h"

double ddot(double* vecv, double* vecw, int size, int p) 
{
    int i;
    double partial = 0, a;
    int nrank;
    
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Compute partials from already distributed vectors
    for(i = 0; i < size/p; i++)
        partial += *(vecv + i) * *(vecw + i);
   
    MPI_Allreduce(&partial, &a, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
    return a;
}



void daxpy(double* vecv, double* vecw, int size, double a, double b, int p) 
{
    int i;
    double partial = 0;
    int nrank;
    
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Compute partials from already distributed vectors
    for(i = 0; i < size/p; i++)
        *(vecv + i) = (a * *(vecv + i)) + (b * *(vecw + i));
    
    return;
}