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

void matvec(double* vec, int slice, int k, double* out){
    int rank, p, i;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    double buffer[k];

    if (rank == 0)
    {
        for (i = 0; i < k; ++i)
        {
            buffer[i] = 0;
        }
    } else {
        for (i = 0; i < k; ++i)
        {
            buffer[i] = vec[i];
        }
    }

    double last[k]; // first k elements from rank + 1 cpu, 0 array if rank = p-1
    if (p == 1)
    {
        for (i = 0; i < k; ++i)
        {
            last[i] = buffer[i];
        }
    } else {
        MPI_Send(buffer, k, MPI_DOUBLE, (rank-1)%p, 0, MPI_COMM_WORLD);
        MPI_Recv(&last, k, MPI_DOUBLE, (rank+1)%p, 0, MPI_COMM_WORLD, &status);  
    }  
      
    if (rank == p-1)
    {
        for (i = 0; i < k; ++i)
        {
            buffer[i] = 0;
        }
    } else {
        for (i = 0; i < k; ++i)
        {
            buffer[i] = vec[slice-k+i];
        }
    }

    double first[k]; // last k elements from rank - 1 cpu, 0 array if rank = 0
    if (p == 1)
    {
        for (i = 0; i < k; ++i)
        {
            first[i] = buffer[i];
        }
    } else {
        MPI_Send(buffer, k, MPI_DOUBLE, (rank+1)%p, 0, MPI_COMM_WORLD);
        MPI_Recv(&first, k, MPI_DOUBLE, (rank-1)%p, 0, MPI_COMM_WORLD, &status);   
    }    

    double co[5];
    for (i = 0; i < slice; ++i)
    {
        co[0] = ((i-k) < 0)? first[i] : vec[i-k];
        co[1] = ((i%k-1) < 0)? 0 : vec[i-1];
        co[2] = 4*vec[i];
        co[3] = ((i%k+1) >= k)? 0 : vec[i+1];
        co[4] = ((i+k) > slice)? last[i] : vec[i+k];
        out[i] = co[2] - co[0] - co[1] - co[3] - co[4];
    }
}
