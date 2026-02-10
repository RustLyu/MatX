/* Internal helpers for complex f64 used by dense_c64 and sparse_c64. */
#pragma once

#include "matx/matx.h"

static inline matx_complex_f64 matx_c64_add(matx_complex_f64 a,
                                            matx_complex_f64 b) {
  matx_complex_f64 r;
  r.real = a.real + b.real;
  r.imag = a.imag + b.imag;
  return r;
}

static inline matx_complex_f64 matx_c64_mul(matx_complex_f64 a,
                                            matx_complex_f64 b) {
  matx_complex_f64 r;
  r.real = a.real * b.real - a.imag * b.imag;
  r.imag = a.real * b.imag + a.imag * b.real;
  return r;
}
