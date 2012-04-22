/*
 * main.c
 *
 *  Created on: Apr 18, 2012
 *      Author: ninj0x
 */


/* UCSB CS240A, Spring Quarter 2011
 * Main and supporting functions for the Conjugate Gradient Solver on a 5-point stencil
 *
 * NAME(s):
 * PERM(s):
 * April 4th, 2011
 */
#include "mpi.h"
#include "hw2harness.h"
#include <stdio.h>
#include <stdlib.h>

// Separated linalg functions
#include "linalg.h"

void cgsolve(double* b, int k, int slice, double* x);
double* load_vec( char* filename, int* k );
void save_vec( int k, double* x );


int main( int argc, char* argv[] ) {
	int writeOutX = 0;
	int n, k;
	int iterations = 1000;
	double norm;
	double* b;
	double* x;
	double time;
	double t1, t2;
	int p;
	int rank;
	int slice; // n/p, size of each CPU's b vector.
	MPI_Init( &argc, &argv );
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Read command line args.
	// 1st case runs model problem, 2nd Case allows you to specify your own b vector
	if ( argc == 4 && !strcmp( argv[1], "-i" ) ) {
		double* vec = load_vec( argv[2], &k );
		n = k*k;
		slice = n/p;
		b = (double *)malloc( (slice) * sizeof(double) );

		// each processor slices vec to build its own part of the b vector!
		int i;
		for (i = 0; i < slice; ++i)
		{
			b[i] = vec[(rank*slice) + i];
		}

		free(vec);

	} else if (argc == 3) {
		printf("Running model problem\n");
		k = atoi( argv[1] );
		n = k*k;
		slice = n/p;
		b = (double *)malloc( (slice) * sizeof(double) );

		// each processor calls cs240_getB to build its own part of the b vector!
		int i;
		for (i = 1; i <= slice; ++i)
		{
			b[i] = cs240_getB((rank*slice) + i, n);
		}
	} else {
        if(rank==0)
        {
		printf( "\nCGSOLVE Usage: \n\t"
			"Model Problem:\tmpirun -np [number_procs] cgsolve [k] [output_1=y_0=n]\n\t"
			"Custom Input:\tmpirun -np [number_procs] cgsolve -i [input_filename] [output_1=y_0=n]\n\n");
        }
		exit(0);

	}

	int i;
	for (i = 0; i <= 100; ++i)
	{
		printf("%f\n", b[i]);
	}
	printf("debug3\n");
	writeOutX = atoi( argv[argc-1] ); // Write X to file if true, do not write if unspecified.

	// Start Timer
	t1 = MPI_Wtime();

	// Initialize x vector
	x = (double *)malloc( (slice) * sizeof(double) );
	for (i = 0; i < slice; ++i)
	{
		x[i] = 0;
	}

	// CG Solve here!	
	cgsolve(b, k, slice, x);

	// End Timer
	t2 = MPI_Wtime();

	if ( writeOutX ) {
		save_vec( k, x );
	}

	// Output
	printf( "Problem size (k): %d\n",k);
	printf( "Norm of the residual after %d iterations: %lf\n",iterations,norm);
	printf( "Elapsed time during CGSOLVE: %lf\n", t1-t2);

	// Deallocate
	free(b);
	free(x);

	MPI_Finalize();

	return 0;
}


/*
 * Supporting Functions
 *
 */

void cgsolve(double* b, int k, int slice, double* x){
	// r = b; % r = b - A*x starts equal to b
	// d = r; % first search direction is r
	double r[slice], rnew[slice], d[slice], ad[slice];
	int i;
	for (i = 0; i < slice; ++i)
	{
		r[i] = b[i];
		d[i] = r[i];
	}
	double alpha, beta;
	double error;
	// while (still iterating)
	do
	{
		matvec(d, slice, ad); // A*d
		alpha = ddot(r, r, slice)/ddot(d, ad, slice); // alpha = r'*r / (d'*A*d);
		daxpy(x, d, slice, 1, alpha); // x = x + alpha * d; % step to next guess

		for (i = 0; i < slice; ++i) // rnew = r - alpha * A*d; % update residual r
		{
			rnew[i] = r[i];
		}
		daxpy(rnew, ad, slice, 1, -alpha); 

		error = ddot(rnew, rnew, slice); // beta = rnew'*rnew / r'*r;
		beta = error/ddot(r, r, slice);

		for (i = 0; i < slice; ++i) // r = rnew;
		{
			r[i] = rnew[i];
		}

		daxpy(d, r, slice, beta, 1); // d = r + beta * d; % compute new search direction

	} while (error > .01);

}

// Load Function
// NOTE: does not distribute data across processors
double* load_vec( char* filename, int* k ) {
	FILE* iFile = fopen(filename, "r");
	int nScan;
	int nTotal = 0;
	int n;

	if ( iFile == NULL ) {
		printf("Error reading file.\n");
		exit(0);
	}

	nScan = fscanf( iFile, "k=%d\n", k );
	if ( nScan != 1 ) {
		printf("Error reading dimensions.\n");
		exit(0);
	}

	n = (*k)*(*k);
	double* vec = (double *)malloc( n * sizeof(double) );

	do {
		nScan = fscanf( iFile, "%lf", &vec[nTotal++] );
	} while ( nScan >= 0 );

	if ( nTotal != n+1 ) {
		printf("Incorrect number of values scanned n=%d, nTotal=%d.\n",n,nTotal);
		exit(0);
	}

	return vec;
}

// Save a vector to a file.
void save_vec( int k, double* x ) {
	FILE* oFile;
	int i;
	oFile = fopen("xApprox.txt","w");

	fprintf( oFile, "k=%d\n", k );

	for (i = 0; i < k*k; i++) {
    	fprintf( oFile, "%lf\n", x[i]);
 	}

	fclose( oFile );
}
