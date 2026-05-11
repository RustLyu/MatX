//====  Copyright (c) 2017 Advanced Micro Devices, Inc.  All rights reserved.
//
//               Developed by: Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimers.
//
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
//
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//===----------------------------------------------------------------------===//

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "secrng.h"

// Number of random numbers to generate
#define N 100

// Number of trials
#define NT 1000

int main(void)
{
   int ret = 0;
   int i;
   uint16_t rng16;
   uint32_t rng32;
   uint64_t rng64;
   uint32_t* rng32_arr = NULL;
   uint64_t* rng64_arr = NULL;
   unsigned char* rng_chararr = NULL;

   ret = is_RDRAND_supported();

   if (ret == SECRNG_SUPPORTED)
   {
        printf("\nRDRAND supported!\n\n");
        
        //Get 16-bit random number
        ret = get_rdrand16u(&rng16, NT);
        
        if (ret == SECRNG_SUCCESS)
                printf("RDRAND rng 16-bit value %u\n\n", rng16);
        else
                printf("Failure in retrieving random value using RDRAND!\n");

        //Get 32-bit random number
        ret = get_rdrand32u(&rng32, NT);
        
        if (ret == SECRNG_SUCCESS)
                printf("RDRAND rng 32-bit value %u\n\n", rng32);
        else
                printf("Failure in retrieving random value using RDRAND!\n");

        //Get 64-bit random number
        ret = get_rdrand64u(&rng64, NT);
        
        if (ret == SECRNG_SUCCESS)
                printf("RDRAND rng 64-bit value %lu\n\n", rng64);
        else
                printf("Failure in retrieving random value using RDRAND!\n");

        //Get a range of 32-bit random values
        rng32_arr = (uint32_t*)  malloc(sizeof(uint32_t) * N);

        ret = get_rdrand32u_arr(rng32_arr, N, NT);

        if (ret == SECRNG_SUCCESS)
        {
                printf("RDRAND for %u 32-bit random values succeeded!\n", N);
                printf("First 10 values in the range are: \n");
                for (i = 0; i < (N > 10? 10 : N); i++)
                        printf("%u\n", rng32_arr[i]);

                printf("\n"); 
        }
        else
                printf("Failure in retrieving array of random values using RDRAND!\n"); 

        //Get a range of 64-bit random values
        rng64_arr = (uint64_t*)  malloc(sizeof(uint64_t) * N);

        ret = get_rdrand64u_arr(rng64_arr, N, NT);

        if (ret == SECRNG_SUCCESS)
        {
                printf("RDRAND for %u 64-bit random values succeeded!\n", N);
                printf("First 10 values in the range are: \n");
                for (i = 0; i < (N > 10? 10 : N); i++)
                        printf("%lu\n", rng64_arr[i]);

                printf("\n");
        }
        else
                printf("Failure in retrieving array of random values using RDRAND!\n");      

        //Get random bytes of given size
        rng_chararr = (unsigned char*)  malloc(sizeof(unsigned char) * N);

        ret = get_rdrand_bytes_arr(rng_chararr, N, NT);       
        if (ret == SECRNG_SUCCESS)
        {
                printf("RDRAND for %u random bytes succeeded!\n", N);
                printf("First 10 values in the range are: \n");
                for (i = 0; i < (N > 10? 10 : N); i++)
                        printf("%u\n", rng_chararr[i]);

                printf("\n");
        }
        else
                printf("Failure in retrieving array of random values using RDRAND!\n");      
 
   }
   else
   {
        printf("No support for RDRAND!\n");
   }
        
   printf("***********************************************************************\n");
   
   ret = is_RDSEED_supported();
   if (ret == SECRNG_SUPPORTED)
   {
        printf("RDSEED supported!\n\n");

        //Get 16-bit random seed value          
        ret = get_rdseed16u(&rng16, NT);
        
        if (ret == SECRNG_SUCCESS)
                printf("RDSEED rng 16-bit value %u\n\n", rng16);
        else
                printf("Failure in retrieving random value using RDSEED!\n");

        //Get 32-bit random seed value          
        ret = get_rdseed32u(&rng32, NT);
        
        if (ret == SECRNG_SUCCESS)
                printf("RDSEED rng 32-bit value %u\n\n", rng32);
        else
                printf("Failure in retrieving random value using RDSEED!\n");

        //Get 64-bit random seed value          
        ret = get_rdseed64u(&rng64, NT);
        
        if (ret == SECRNG_SUCCESS)
                printf("RDSEED rng 64-bit value %lu\n\n", rng64);
        else
                printf("Failure in retrieving random value using RDSEED!\n");

        //Get a range of 32-bit random values
        if (!rng32_arr)
                rng32_arr = (uint32_t *)  malloc(sizeof(uint32_t) * N);

        ret = get_rdseed32u_arr(rng32_arr, N, NT);

        if (ret == SECRNG_SUCCESS)
        {
                printf("RDSEED for %u 32-bit random values succeeded!\n", N);
                printf("First 10 values in the range are: \n");
                for (i = 0; i < (N > 10? 10 : N); i++)
                        printf("%u\n", rng32_arr[i]);

                printf("\n");
        }
        else
                printf("Failure in retrieving array of random values using RDSEED!\n");

        //Get a range of 64-bit random values
        if (!rng64_arr)
                rng64_arr = (uint64_t *)  malloc(sizeof(uint64_t) * N);

        ret = get_rdseed64u_arr(rng64_arr, N, NT);

        if (ret == SECRNG_SUCCESS)
        {
                printf("RDSEED for %u 64-bit random values succeeded!\n", N);
                printf("First 10 values in the range are: \n");
                for (i = 0; i < (N > 10? 10 : N); i++)
                        printf("%lu\n", rng64_arr[i]);

                printf("\n");
        }
        else
                printf("Failure in retrieving array of random values using RDSEED!\n");

        //Get random bytes of a given size
        if (!rng_chararr)
                rng_chararr = (unsigned char*)  malloc(sizeof(unsigned char) * N);

        ret = get_rdseed_bytes_arr(rng_chararr, N, NT);       
        if (ret == SECRNG_SUCCESS)
        {
                printf("RDSEED for %u random bytes succeeded!\n", N);
                printf("First 10 values in the range are: \n");
                for (i = 0; i < (N > 10? 10 : N); i++)
                        printf("%u\n", rng_chararr[i]);

                printf("\n");
        }
        else
                printf("Failure in retrieving array of random values using RDSEED!\n");      
   }
   else
   {
        printf("No support for RDSEED!\n");
   }
  
   if (rng32_arr)
        free(rng32_arr);
   if (rng64_arr)
        free(rng64_arr);
   if (rng_chararr)
        free(rng_chararr);

   if (ret > 0)
	return 0;

   return ret;
}

