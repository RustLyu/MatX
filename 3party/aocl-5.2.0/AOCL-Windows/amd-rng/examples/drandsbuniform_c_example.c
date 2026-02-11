
/* dranduniform Sobol as BRNG Example Program Text */
/*
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
 */

/* Sobol Generator provides multidimensional sequences */
#include <rng.h>
#include <stdio.h>

#define SEQLENGTH     3
#define DIMENSION     4
#define MSTATE        60
#define MDIMENSION    1

rng_int_t dimension[MDIMENSION], state[MSTATE];
double    x[SEQLENGTH*DIMENSION];

int main(void)
{
  /* Length of sequence to be generated in each dimension */
  const int sl =  SEQLENGTH;

  /* Number of dimensions*/
  const int dim = DIMENSION;
  double a, b;
  rng_int_t n, genid, subid, ldimension, lstate, info;


  printf("\n");
  printf("--RNG-example: sobol uniform random sequences using dranduniform--\n");
  printf("------------------------------------------------------------------\n");
  printf("\n");

  /* Initialize the number of variates required */
  n = sl*dim;

  /* Use SFMT19937 Generator as the base generator */
  genid = 7;
  subid = 0;

  lstate  = MSTATE;
  ldimension   = 1;
  dimension[0] = dim;

  /* Initialize the base generator */
  drandinitialize(genid, subid, dimension, &ldimension, state, &lstate, &info);

  /* Generate a sequence from a uniform U(0,1) distribution */
  a = 0.0;
  b = 1.0;

  dranduniform(n, a, b, state, x, &info);

  /* Output sequence is expected to be in following form
   * x[0], x[1], x[2] ... x[dim-1] first member of the sequence
   * x[dim], x[dim+1], x[dim+2] ... x[2*dim-1] second member of the sequence
   * the sequence continues like this till given length of sequence is generated
   */

  /* Print the sequence */
  printf("Numbers from a Sobol uniform U(0,1) distribution:\n");

  for (int i=0; i<sl; i++) {
    for (int j=0; j<dim; j++) {
      printf("%10.4f", x[(i*dim)+ j] );
    }
    printf("\n");
  }

  printf("\n");

  return 0;
}
