// Linear algebra tester
// compile: mpicc -o linalgtest linalgtest.c linalg.c -lm
// run: mpirun -n 2 linalgtest

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#include "linalg.h"


int main( int argc, char* argv[] ) 
{
    int p, rank, slice, size = 8;
    double r;
    double* v;
    double* w;
    double a, b;
    
    MPI_Init( &argc, &argv );
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    slice = size/p;
    
    v = (double *)malloc( (slice) * sizeof(double) );
    w = (double *)malloc( (slice) * sizeof(double) );
    
    
    // Initialize v & w
    int i;
    for (i = 0; i < slice; i++)
    {
    v[i] = 1.0;
    w[i] = 2.0;
    }
    
    // Initialize alpha & beta
    a = 3, b = 2;
    
    r = ddot(v,w,slice);
    daxpy(v,w,slice,a,b);
    
    // 
    if(rank==0)
        {
        printf( "ddot result:\nr = %3.2f\n\n", r);
        printf("daxpy ( v = %3.2f*v + %3.2f*w ) result:\n", a, b);
        }
    
    for (i = 0; i < slice; i++)
		printf("%3.2f\n", v[i]);
    
    free(v);
	free(w);
    
    MPI_Finalize();
    
	return 0;
}