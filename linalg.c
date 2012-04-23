// Linear algebra functions for Poisson Solver

#include "linalg.h"

double ddot(double* vecv, double* vecw, int slice) 
{
    int i;
    double partial = 0, a;
    
    // Compute partials from already distributed vectors
    for(i = 0; i < slice; i++)
        partial += *(vecv + i) * *(vecw + i);
   
    MPI_Allreduce(&partial, &a, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
    return a;
}



void daxpy(double* vecv, double* vecw, int slice, double a, double b) 
{
    int i;
    
    // Compute partials from already distributed vectors
    for(i = 0; i < slice; i++)
        *(vecv + i) = (a * *(vecv + i)) + (b * *(vecw + i));
    
    return;
}

void matvec(double* vec, int slice, double* out){
    
}