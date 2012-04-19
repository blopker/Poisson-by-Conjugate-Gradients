#ifndef LINALG_H
#define LINANLG_H

#include "mpi.h"

//Function declarations
double ddot(double* vecv, double* vecw, int size, int p);
void daxpy(double* vecv, double* vecw, int size, double a, double b, int p);

#endif