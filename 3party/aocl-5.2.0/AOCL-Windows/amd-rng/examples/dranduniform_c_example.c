
/* dranduniform Example Program Text */
/*
 * Copyright (C) 2005-2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <rng.h>
#include <stdio.h>


#define MN     12
#define MSTATE 20
#define MSEED  10

rng_int_t seed[MSEED], state[MSTATE];
double    x[MN];


int main(void)
{
  const int mn = MN;
  double a, b;
  rng_int_t n, genid, subid, lseed, lstate, info;


  printf("\n");
  printf("--RNG-example: uniform random numbers using dranduniform--\n");
  printf("----------------------------------------------------------\n");
  printf("\n");

  /* Initialize the number of variates required */
  n = mn;

  /* Use the basic generator as the base generator */
  genid = 1;
  subid = 1;

  /* Populate the seed array, basic generator needs one seed, and a
     STATE array of length 16 */
  lstate  = 16;
  lseed   = 1;
  seed[0] = 122421;

  /* Initialize the base generator */
  drandinitialize(genid, subid, seed, &lseed, state, &lstate, &info);

  /* Generate a sequence from a uniform U(0,1) distribution */
  a = 0.0;
  b = 1.0;

  dranduniform(n, a, b, state, x, &info);

  /* Print the sequence */
  printf("Numbers from a uniform U(0,1) distribution:\n");

  for (int i=0; i<n; i++) {
    printf("%10.4f", x[i]);

    if ((i+1)%4 == 0)
      printf("\n");
  }

  printf("\n");

  return 0;
}
