
/* drandgaussian SFMT as BRNG Example Program Text */
/*
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
 */

#include <rng.h>
#include <stdio.h>


#define MN     12
#define MSTATE 637
#define MSEED  624

rng_int_t seed[MSEED], state[MSTATE];
double    x[MN];

int main(void)
{
  const int mn = MN;
  double mu, var;
  rng_int_t n, genid, subid, lseed, lstate, info;


  printf("\n");
  printf("--RNG-example: sfmt gaussian random numbers using drandgaussian--\n");
  printf("-----------------------------------------------------------------\n");
  printf("\n");

  /* Initialize the number of variates required */
  n = mn;

  /* Use SFMT Generator as the base generator */
  genid = 6;
  subid = 1;

  lstate  = MSTATE;
  lseed   = 1;
  seed[0] = 122421;

  /* Initialize the base generator */
  drandinitialize(genid, subid, seed, &lseed, state, &lstate, &info);

  /* Generate gaussian variate using given mean, variance */
  mu  = 0.0;
  var = 1.0;

  drandgaussian(n, mu, var, state, x, &info);

  /* Print the sequence */
  printf("Numbers from a SFMT gaussian distribution:\n");

  for (int i=0; i<n; i++) {
    printf("%10.4f", x[i]);

    if ((i+1)%4 == 0)
      printf("\n");
  }

  printf("\n");

  return 0;
}
