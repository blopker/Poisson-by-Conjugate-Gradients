cs240a-hw2
==========

Solving Poisson&#39;s Equation by Conjugate Gradients
by Karl Lopker & Drew Waranis

To build:
make

Usage:
Model Problem: mpirun -np [number_procs] cgsolve [k] [output_1=y_0=n]
Custom Input: mpirun -np [number_procs] cgsolve -i [input_filename] [output_1=y_0=n]