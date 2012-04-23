#ifndef LINALG_H
#define LINANLG_H

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

//Function declarations
double ddot(double* vecv, double* vecw, int slice);
void daxpy(double* vecv, double* vecw, int slice, double a, double b);
void matvec(double* vec, int slice, int k, double* out);
#endif