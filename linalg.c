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
    	if (rank % 2) {
            MPI_Send(buffer, k, MPI_DOUBLE, (rank-1+p)%p, 0, MPI_COMM_WORLD);
            MPI_Recv(&last, k, MPI_DOUBLE, (rank+1+p)%p, 0, MPI_COMM_WORLD, &status);
    	} else {
            MPI_Recv(&last, k, MPI_DOUBLE, (rank+1+p)%p, 0, MPI_COMM_WORLD, &status);
            MPI_Send(buffer, k, MPI_DOUBLE, (rank-1+p)%p, 0, MPI_COMM_WORLD);
    	}
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
    	if (rank % 2) {
            MPI_Send(buffer, k, MPI_DOUBLE, (rank+1+p)%p, 0, MPI_COMM_WORLD);
            MPI_Recv(&first, k, MPI_DOUBLE, (rank-1+p)%p, 0, MPI_COMM_WORLD, &status);
    	} else {
            MPI_Recv(&first, k, MPI_DOUBLE, (rank-1+p)%p, 0, MPI_COMM_WORLD, &status);
            MPI_Send(buffer, k, MPI_DOUBLE, (rank+1+p)%p, 0, MPI_COMM_WORLD);
    	}

    }    

    int col;
    for (i = 0; i < slice; ++i)
    {
    	col = i%k;
		// Multiply point by 4
		out[i] = 4*vec[i];
		// Deduct point above
		out[i] -= (i-k < 0)? first[col] : vec[i-k];
		// Deduct point below
		out[i] -= (i+k > slice)? last[col] : vec[i+k];
		// Deduct point to left
		out[i] -= (col == 0)? 0 : vec[i-1];
		// Deduct point to right
		out[i] -= (col == k-1)? 0 : vec[i+1];
    }
}
