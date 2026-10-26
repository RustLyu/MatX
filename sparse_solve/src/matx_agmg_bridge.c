// =============================================================================
// matx_agmg_bridge.c — AGMG ↔ LP64 BLAS Bridge DLL
// =============================================================================
//
// AGMG (Fortran) requires LP64 (32-bit int) Fortran BLAS/LAPACK symbols:
//   dgemv_, dnrm2_, daxpy_, ddot_, dcopy_, dscal_,
//   dpotrf_, dgetrf_, dgetrs_,
//   zgemv_, dznrm2_, zcopy_, zdotc_, zaxpy_
//
// MatX uses ILP64 (64-bit int) OpenBLAS which exports the SAME symbol names
// but with a DIFFERENT integer ABI.  Linking both directly into the same
// executable would cause either incorrect results or runtime hangs.
//
// SOLUTION: this file is compiled as a SHARED library (DLL on Windows).
//   - It statically links libagmg.a + LP64 libopenblas.dll.a.
//   - Fortran BLAS symbols are resolved internally, hidden from the main exe.
//   - dagmg_ / zagmg_ are exported (from libagmg.a) — the main exe calls
//     AGMG through these and never sees the LP64 BLAS symbols.
//   - MatX's own CBLAS calls go through ILP64 libopenblas_64.dll (separate
//     DLL with its own import table → no ABI conflict).
//
// Windows DLL isolation: each DLL has its own import address table.  The
// bridge DLL imports dgemv_ etc. from LP64 libopenblas.dll.  The main exe
// imports cblas_dgemv etc. from ILP64 libopenblas_64.dll.  Different DLLs,
// different IATs, no conflict even if symbol names overlap.
//
// =============================================================================

/* dagmg_ — double-precision real AGMG solver (from libagmg.a) */
extern void dagmg_(int *n, double *a, int *ja, int *ia,
                   double *f, double *x,
                   int *ijob, int *iprint, int *nrest,
                   int *iter, double *tol);

/* zagmg_ — double-precision complex AGMG solver (from libagmg.a) */
extern void zagmg_(int *n, double *a, int *ja, int *ia,
                   double *f, double *x,
                   int *ijob, int *iprint, int *nrest,
                   int *iter, double *tol);

/* Dummy reference to ensure the linker pulls dagmg_ / zagmg_ from libagmg.a
   and resolves their BLAS dependencies inside this DLL. */
__attribute__((used)) static void (*const agmg_refs[])(void) = {
    (void (*)(void))&dagmg_,
    (void (*)(void))&zagmg_,
};