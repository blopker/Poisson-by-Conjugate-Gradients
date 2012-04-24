/* UCSB CS240A, Spring Quarter 2011
 * Main and supporting functions for the Conjugate Gradient Solver on a 5-point stencil
 *
 * NAME(s): Drew Waranis, Karl Bo Lopker
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
void save_vec( int k, int slice, int p, int rank, double* x);


int main( int argc, char* argv[] ) {
	int writeOutX = 0;
	int n, k, i;
	int iterations = 1000;
	double norm;

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
//		b = (double *)malloc( (slice) * sizeof(double) );

		// each processor slices vec to build its own part of the b vector!
		for (i = 0; i < slice; ++i)
		{
//			b[i] = vec[(rank*slice) + i];
		}
//		x = (double *)malloc( (slice) * sizeof(double) );
		free(vec);

	} else if (argc == 3) {
		printf("Running model problem\n");


	} else {
        if(rank==0)
        {
		printf( "\nCGSOLVE Usage: \n\t"
			"Model Problem:\tmpirun -np [number_procs] cgsolve [k] [output_1=y_0=n]\n\t"
			"Custom Input:\tmpirun -np [number_procs] cgsolve -i [input_filename] [output_1=y_0=n]\n\n");
        }
		exit(0);

	}
	k = atoi( argv[1] );
	n = k*k;
	slice = n/p;
	double* b;
	double* x;

	b = (double *)malloc( (slice) * sizeof(double) );
	// each processor calls cs240_getB to build its own part of the b vector!
	x = (double *)malloc( (slice) * sizeof(double) );
	for (i = 1; i <= slice; ++i)
	{
		b[i-1] = cs240_getB((rank*slice) + i, n);
	}

	writeOutX = atoi( argv[argc-1] ); // Write X to file if true, do not write if unspecified.

	// Start Timer
	t1 = MPI_Wtime();

	// Initialize x vector
	for (i = 0; i < slice; ++i)
	{
		x[i] = 0;
	}

	printf( "\nCGSOLVE starting...\n");
	// CG Solve here!	
	cgsolve(b, k, slice, x);
	printf( "\nCGSOLVE done!\n");
	// End Timer
	t2 = MPI_Wtime();

	for (i = slice-20; i < slice; ++i)
	{
		printf( "x: %f\n", x[i]);
	}

	if ( writeOutX ) {
		save_vec( k, slice, p, rank, x );
	}

	// Output
	printf( "Problem size (k): %d\n",k);
	printf( "Norm of the residual after %d iterations: %lf\n",iterations,norm);
	printf( "Elapsed time during CGSOLVE: %lf\n", t2-t1);

	// Deallocate
	printf( "Deallocating...\n");
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
	double* r, *rnew, *d, *ad;
	r = (double *)malloc( (slice) * sizeof(double) );
	rnew = (double *)malloc( (slice) * sizeof(double) );
	d = (double *)malloc( (slice) * sizeof(double) );
	ad = (double *)malloc( (slice) * sizeof(double) );
	int i;
	for (i = 0; i < slice; ++i)
	{
		r[i] = b[i]; // r = b; % r = b - A*x starts equal to b
		d[i] = r[i]; // d = r; % first search direction is r
	}
	double alpha, beta;
	double error;

	// while (still iterating)
	int j = 0;
	do
	{
		matvec(d, slice, k, ad); // Compute A*d once and store it for later
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
		j++;
	} while (j < 3);
	free(r);
	free(rnew);
	free(d);
	free(ad);

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
void save_vec( int k, int slice, int p, int rank, double* x ) {
	int i;
	for (i = 0; i < p; ++i) {
		if (rank == i) {
			FILE* oFile;
			int i;
			oFile = fopen("xApprox.txt","w");

			fprintf( oFile, "k=%d\n", k );

			for (i = 0; i < slice; i++) {
				fprintf( oFile, "%lf\n", x[i]);
				printf("x: %lf\n", x[i]);
			}
			fclose( oFile );
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}
