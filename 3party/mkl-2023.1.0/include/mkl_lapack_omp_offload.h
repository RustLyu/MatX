/*******************************************************************************
* Copyright 2019-2022 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
!  Content:
!      Intel(R) oneAPI Math Kernel Library (oneMKL) C interface for
!      OpenMP offload for LAPACK
!******************************************************************************/

#ifndef _MKL_LAPACK_OMP_OFFLOAD_H_
#define _MKL_LAPACK_OMP_OFFLOAD_H_

#include "mkl_types.h"
#include "mkl_lapack_omp_variant.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cgebrd,a,d,e,tauq,taup,work,info)
void cgebrd(const MKL_INT* m, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda, float* d, float* e,
            MKL_Complex8* tauq, MKL_Complex8* taup, MKL_Complex8* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(dgebrd,a,d,e,tauq,taup,work,info)
void dgebrd(const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, double* d, double* e, double* tauq,
            double* taup, double* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(sgebrd,a,d,e,tauq,taup,work,info)
void sgebrd(const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, float* d, float* e, float* tauq,
            float* taup, float* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zgebrd,a,d,e,tauq,taup,work,info)
void zgebrd(const MKL_INT* m, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda, double* d, double* e,
            MKL_Complex16* tauq, MKL_Complex16* taup, MKL_Complex16* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cgesvd,a,s,u,vt,work,rwork,info)
void cgesvd(const char* jobu, const char* jobvt, const MKL_INT* m, const MKL_INT* n, MKL_Complex8* a,
            const MKL_INT* lda, float* s, MKL_Complex8* u, const MKL_INT* ldu, MKL_Complex8* vt, const MKL_INT* ldvt,
            MKL_Complex8* work, const MKL_INT* lwork, float* rwork, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zgesvd,a,s,u,vt,work,rwork,info)
void zgesvd(const char* jobu, const char* jobvt, const MKL_INT* m, const MKL_INT* n, MKL_Complex16* a,
            const MKL_INT* lda, double* s, MKL_Complex16* u, const MKL_INT* ldu, MKL_Complex16* vt, const MKL_INT* ldvt,
            MKL_Complex16* work, const MKL_INT* lwork, double* rwork, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(dgesvd,a,s,u,vt,work,info)
void dgesvd(const char* jobu, const char* jobvt, const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda,
            double* s, double* u, const MKL_INT* ldu, double* vt, const MKL_INT* ldvt, double* work,
            const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(sgesvd,a,s,u,vt,work,info)
void sgesvd(const char* jobu, const char* jobvt, const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda,
            float* s, float* u, const MKL_INT* ldu, float* vt, const MKL_INT* ldvt, float* work, const MKL_INT* lwork,
            MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cgetrf_batch_strided,a,ipiv,info)
void cgetrf_batch_strided(const MKL_INT* m, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda,
                          const MKL_INT* stride_a, MKL_INT* ipiv, const MKL_INT* stride_ipiv, const MKL_INT* batch_size,
                          MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(dgetrf_batch_strided,a,ipiv,info)
void dgetrf_batch_strided(const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, const MKL_INT* stride_a,
                          MKL_INT* ipiv, const MKL_INT* stride_ipiv, const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(sgetrf_batch_strided,a,ipiv,info)
void sgetrf_batch_strided(const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, const MKL_INT* stride_a,
                          MKL_INT* ipiv, const MKL_INT* stride_ipiv, const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zgetrf_batch_strided,a,ipiv,info)
void zgetrf_batch_strided(const MKL_INT* m, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda,
                          const MKL_INT* stride_a, MKL_INT* ipiv, const MKL_INT* stride_ipiv, const MKL_INT* batch_size,
                          MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cgetrf,a,ipiv,info)
void cgetrf(const MKL_INT* m, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda, MKL_INT* ipiv,
            MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(dgetrf,a,ipiv,info)
void dgetrf(const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, MKL_INT* ipiv, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(sgetrf,a,ipiv,info)
void sgetrf(const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, MKL_INT* ipiv, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zgetrf,a,ipiv,info)
void zgetrf(const MKL_INT* m, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda, MKL_INT* ipiv,
            MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cgetrfnp_batch_strided,a,info)
void cgetrfnp_batch_strided(const MKL_INT* m, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda,
                            const MKL_INT* stride_a, const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(dgetrfnp_batch_strided,a,info)
void dgetrfnp_batch_strided(const MKL_INT* m, const MKL_INT* n, double* a, const MKL_INT* lda, const MKL_INT* stride_a,
                            const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(sgetrfnp_batch_strided,a,info)
void sgetrfnp_batch_strided(const MKL_INT* m, const MKL_INT* n, float* a, const MKL_INT* lda, const MKL_INT* stride_a,
                            const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zgetrfnp_batch_strided,a,info)
void zgetrfnp_batch_strided(const MKL_INT* m, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda,
                            const MKL_INT* stride_a, const MKL_INT* batch_size, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cgetri,a,ipiv,work,info)
void cgetri(const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda, const MKL_INT* ipiv, MKL_Complex8* work,
            const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(dgetri,a,ipiv,work,info)
void dgetri(const MKL_INT* n, double* a, const MKL_INT* lda, const MKL_INT* ipiv, double* work, const MKL_INT* lwork,
            MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(sgetri,a,ipiv,work,info)
void sgetri(const MKL_INT* n, float* a, const MKL_INT* lda, const MKL_INT* ipiv, float* work, const MKL_INT* lwork,
            MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zgetri,a,ipiv,work,info)
void zgetri(const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda, const MKL_INT* ipiv, MKL_Complex16* work,
            const MKL_INT* lwork, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cgetrs,a,ipiv,b,info)
void cgetrs(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const MKL_Complex8* a, const MKL_INT* lda,
            const MKL_INT* ipiv, MKL_Complex8* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(dgetrs,a,ipiv,b,info)
void dgetrs(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const double* a, const MKL_INT* lda,
            const MKL_INT* ipiv, double* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(sgetrs,a,ipiv,b,info)
void sgetrs(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const float* a, const MKL_INT* lda,
            const MKL_INT* ipiv, float* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zgetrs,a,ipiv,b,info)
void zgetrs(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const MKL_Complex16* a, const MKL_INT* lda,
            const MKL_INT* ipiv, MKL_Complex16* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cgetrsnp_batch_strided,a,b,info)
void cgetrsnp_batch_strided(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const MKL_Complex8* a,
                            const MKL_INT* lda, const MKL_INT* stride_a, MKL_Complex8* b, const MKL_INT* ldb,
                            const MKL_INT* stride_b, const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(dgetrsnp_batch_strided,a,b,info)
void dgetrsnp_batch_strided(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const double* a,
                            const MKL_INT* lda, const MKL_INT* stride_a, double* b, const MKL_INT* ldb,
                            const MKL_INT* stride_b, const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(sgetrsnp_batch_strided,a,b,info)
void sgetrsnp_batch_strided(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const float* a,
                            const MKL_INT* lda, const MKL_INT* stride_a, float* b, const MKL_INT* ldb,
                            const MKL_INT* stride_b, const MKL_INT* batch_size, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zgetrsnp_batch_strided,a,b,info)
void zgetrsnp_batch_strided(const char* trans, const MKL_INT* n, const MKL_INT* nrhs, const MKL_Complex16* a,
                            const MKL_INT* lda, const MKL_INT* stride_a, MKL_Complex16* b, const MKL_INT* ldb,
                            const MKL_INT* stride_b, const MKL_INT* batch_size, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cheev,a,w,work,rwork,info)
void cheev(const char* jobz, const char* uplo, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda, float* w,
           MKL_Complex8* work, const MKL_INT* lwork, float* rwork, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zheev,a,w,work,rwork,info)
void zheev(const char* jobz, const char* uplo, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda, double* w,
           MKL_Complex16* work, const MKL_INT* lwork, double* rwork, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cheevd,a,w,work,rwork,iwork,info)
void cheevd(const char* jobz, const char* uplo, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda, float* w,
            MKL_Complex8* work, const MKL_INT* lwork, float* rwork, const MKL_INT* lrwork, MKL_INT* iwork,
            const MKL_INT* liwork, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zheevd,a,w,work,rwork,iwork,info)
void zheevd(const char* jobz, const char* uplo, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda, double* w,
            MKL_Complex16* work, const MKL_INT* lwork, double* rwork, const MKL_INT* lrwork, MKL_INT* iwork,
            const MKL_INT* liwork, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cheevx,a,m,w,z,work,rwork,iwork,ifail,info)
void cheevx(const char* jobz, const char* range, const char* uplo, const MKL_INT* n, MKL_Complex8* a,
            const MKL_INT* lda, const float* vl, const float* vu, const MKL_INT* il, const MKL_INT* iu,
            const float* abstol, MKL_INT* m, float* w, MKL_Complex8* z, const MKL_INT* ldz, MKL_Complex8* work,
            const MKL_INT* lwork, float* rwork, MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zheevx,a,m,w,z,work,rwork,iwork,ifail,info)
void zheevx(const char* jobz, const char* range, const char* uplo, const MKL_INT* n, MKL_Complex16* a,
            const MKL_INT* lda, const double* vl, const double* vu, const MKL_INT* il, const MKL_INT* iu,
            const double* abstol, MKL_INT* m, double* w, MKL_Complex16* z, const MKL_INT* ldz, MKL_Complex16* work,
            const MKL_INT* lwork, double* rwork, MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(chegvd,a,b,w,work,rwork,iwork,info)
void chegvd(const MKL_INT* itype, const char* jobz, const char* uplo, const MKL_INT* n, MKL_Complex8* a,
            const MKL_INT* lda, MKL_Complex8* b, const MKL_INT* ldb, float* w, MKL_Complex8* work, const MKL_INT* lwork,
            float* rwork, const MKL_INT* lrwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zhegvd,a,b,w,work,rwork,iwork,info)
void zhegvd(const MKL_INT* itype, const char* jobz, const char* uplo, const MKL_INT* n, MKL_Complex16* a,
            const MKL_INT* lda, MKL_Complex16* b, const MKL_INT* ldb, double* w, MKL_Complex16* work,
            const MKL_INT* lwork, double* rwork, const MKL_INT* lrwork, MKL_INT* iwork, const MKL_INT* liwork,
            MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(chegvx,a,b,m,w,z,work,rwork,iwork,ifail,info)
void chegvx(const MKL_INT* itype, const char* jobz, const char* range, const char* uplo, const MKL_INT* n,
            MKL_Complex8* a, const MKL_INT* lda, MKL_Complex8* b, const MKL_INT* ldb, const float* vl, const float* vu,
            const MKL_INT* il, const MKL_INT* iu, const float* abstol, MKL_INT* m, float* w, MKL_Complex8* z,
            const MKL_INT* ldz, MKL_Complex8* work, const MKL_INT* lwork, float* rwork, MKL_INT* iwork, MKL_INT* ifail,
            MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zhegvx,a,b,m,w,z,work,rwork,iwork,ifail,info)
void zhegvx(const MKL_INT* itype, const char* jobz, const char* range, const char* uplo, const MKL_INT* n,
            MKL_Complex16* a, const MKL_INT* lda, MKL_Complex16* b, const MKL_INT* ldb, const double* vl,
            const double* vu, const MKL_INT* il, const MKL_INT* iu, const double* abstol, MKL_INT* m, double* w,
            MKL_Complex16* z, const MKL_INT* ldz, MKL_Complex16* work, const MKL_INT* lwork, double* rwork,
            MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(chetrd,a,d,e,tau,work,info)
void chetrd(const char* uplo, const MKL_INT* n, MKL_Complex8* a, const MKL_INT* lda, float* d, float* e,
            MKL_Complex8* tau, MKL_Complex8* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zhetrd,a,d,e,tau,work,info)
void zhetrd(const char* uplo, const MKL_INT* n, MKL_Complex16* a, const MKL_INT* lda, double* d, double* e,
            MKL_Complex16* tau, MKL_Complex16* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(dorgqr,a,tau,work,info)
void dorgqr(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, double* a, const MKL_INT* lda, const double* tau,
            double* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(sorgqr,a,tau,work,info)
void sorgqr(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, float* a, const MKL_INT* lda, const float* tau,
            float* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(dormqr,a,tau,c,work,info)
void dormqr(const char* side, const char* trans, const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const double* a,
            const MKL_INT* lda, const double* tau, double* c, const MKL_INT* ldc, double* work, const MKL_INT* lwork,
            MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(sormqr,a,tau,c,work,info)
void sormqr(const char* side, const char* trans, const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, const float* a,
            const MKL_INT* lda, const float* tau, float* c, const MKL_INT* ldc, float* work, const MKL_INT* lwork,
            MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(csteqr,d,e,z,work,info)
void csteqr(const char* compz, const MKL_INT* n, float* d, float* e, MKL_Complex8* z, const MKL_INT* ldz, float* work,
            MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(dsteqr,d,e,z,work,info)
void dsteqr(const char* compz, const MKL_INT* n, double* d, double* e, double* z, const MKL_INT* ldz, double* work,
            MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(ssteqr,d,e,z,work,info)
void ssteqr(const char* compz, const MKL_INT* n, float* d, float* e, float* z, const MKL_INT* ldz, float* work,
            MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zsteqr,d,e,z,work,info)
void zsteqr(const char* compz, const MKL_INT* n, double* d, double* e, MKL_Complex16* z, const MKL_INT* ldz,
            double* work, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(dsyev,a,w,work,info)
void dsyev(const char* jobz, const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, double* w, double* work,
           const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(ssyev,a,w,work,info)
void ssyev(const char* jobz, const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda, float* w, float* work,
           const MKL_INT* lwork, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(dsyevd,a,w,work,iwork,info)
void dsyevd(const char* jobz, const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, double* w,
            double* work, const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(ssyevd,a,w,work,iwork,info)
void ssyevd(const char* jobz, const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda, float* w, float* work,
            const MKL_INT* lwork, MKL_INT* iwork, const MKL_INT* liwork, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(dsyevx,a,m,w,z,work,iwork,ifail,info)
void dsyevx(const char* jobz, const char* range, const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda,
            const double* vl, const double* vu, const MKL_INT* il, const MKL_INT* iu, const double* abstol, MKL_INT* m,
            double* w, double* z, const MKL_INT* ldz, double* work, const MKL_INT* lwork, MKL_INT* iwork,
            MKL_INT* ifail, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(ssyevx,a,m,w,z,work,iwork,ifail,info)
void ssyevx(const char* jobz, const char* range, const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda,
            const float* vl, const float* vu, const MKL_INT* il, const MKL_INT* iu, const float* abstol, MKL_INT* m,
            float* w, float* z, const MKL_INT* ldz, float* work, const MKL_INT* lwork, MKL_INT* iwork, MKL_INT* ifail,
            MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(dsygvd,a,b,w,work,iwork,info)
void dsygvd(const MKL_INT* itype, const char* jobz, const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda,
            double* b, const MKL_INT* ldb, double* w, double* work, const MKL_INT* lwork, MKL_INT* iwork,
            const MKL_INT* liwork, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(ssygvd,a,b,w,work,iwork,info)
void ssygvd(const MKL_INT* itype, const char* jobz, const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda,
            float* b, const MKL_INT* ldb, float* w, float* work, const MKL_INT* lwork, MKL_INT* iwork,
            const MKL_INT* liwork, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(dsygvx,a,b,m,w,z,work,iwork,ifail,info)
void dsygvx(const MKL_INT* itype, const char* jobz, const char* range, const char* uplo, const MKL_INT* n, double* a,
            const MKL_INT* lda, double* b, const MKL_INT* ldb, const double* vl, const double* vu, const MKL_INT* il,
            const MKL_INT* iu, const double* abstol, MKL_INT* m, double* w, double* z, const MKL_INT* ldz, double* work,
            const MKL_INT* lwork, MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(ssygvx,a,b,m,w,z,work,iwork,ifail,info)
void ssygvx(const MKL_INT* itype, const char* jobz, const char* range, const char* uplo, const MKL_INT* n, float* a,
            const MKL_INT* lda, float* b, const MKL_INT* ldb, const float* vl, const float* vu, const MKL_INT* il,
            const MKL_INT* iu, const float* abstol, MKL_INT* m, float* w, float* z, const MKL_INT* ldz, float* work,
            const MKL_INT* lwork, MKL_INT* iwork, MKL_INT* ifail, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(dsytrd,a,d,e,tau,work,info)
void dsytrd(const char* uplo, const MKL_INT* n, double* a, const MKL_INT* lda, double* d, double* e, double* tau,
            double* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(ssytrd,a,d,e,tau,work,info)
void ssytrd(const char* uplo, const MKL_INT* n, float* a, const MKL_INT* lda, float* d, float* e, float* tau,
            float* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(ctrtrs,a,b,info)
void ctrtrs(const char* uplo, const char* trans, const char* diag, const MKL_INT* n, const MKL_INT* nrhs,
            const MKL_Complex8* a, const MKL_INT* lda, MKL_Complex8* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(dtrtrs,a,b,info)
void dtrtrs(const char* uplo, const char* trans, const char* diag, const MKL_INT* n, const MKL_INT* nrhs,
            const double* a, const MKL_INT* lda, double* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(strtrs,a,b,info)
void strtrs(const char* uplo, const char* trans, const char* diag, const MKL_INT* n, const MKL_INT* nrhs,
            const float* a, const MKL_INT* lda, float* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(ztrtrs,a,b,info)
void ztrtrs(const char* uplo, const char* trans, const char* diag, const MKL_INT* n, const MKL_INT* nrhs,
            const MKL_Complex16* a, const MKL_INT* lda, MKL_Complex16* b, const MKL_INT* ldb, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cungqr,a,tau,work,info)
void cungqr(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, MKL_Complex8* a, const MKL_INT* lda,
            const MKL_Complex8* tau, MKL_Complex8* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zungqr,a,tau,work,info)
void zungqr(const MKL_INT* m, const MKL_INT* n, const MKL_INT* k, MKL_Complex16* a, const MKL_INT* lda,
            const MKL_Complex16* tau, MKL_Complex16* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cunmqr,a,tau,c,work,info)
void cunmqr(const char* side, const char* trans, const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
            const MKL_Complex8* a, const MKL_INT* lda, const MKL_Complex8* tau, MKL_Complex8* c, const MKL_INT* ldc,
            MKL_Complex8* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zunmqr,a,tau,c,work,info)
void zunmqr(const char* side, const char* trans, const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
            const MKL_Complex16* a, const MKL_INT* lda, const MKL_Complex16* tau, MKL_Complex16* c, const MKL_INT* ldc,
            MKL_Complex16* work, const MKL_INT* lwork, MKL_INT* info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cgeqrf,a,tau,work,info)
void cgeqrf(const MKL_INT *m, const MKL_INT *n, MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *tau, MKL_Complex8 *work, const MKL_INT *lwork, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(dgeqrf,a,tau,work,info)
void dgeqrf(const MKL_INT *m, const MKL_INT *n, double *a, const MKL_INT *lda, double *tau, double *work, const MKL_INT *lwork, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(sgeqrf,a,tau,work,info)
void sgeqrf(const MKL_INT *m, const MKL_INT *n, float *a, const MKL_INT *lda, float *tau, float *work, const MKL_INT *lwork, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zgeqrf,a,tau,work,info)
void zgeqrf(const MKL_INT *m, const MKL_INT *n, MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *tau, MKL_Complex16 *work, const MKL_INT *lwork, MKL_INT *info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cgetri_oop_batch_strided,a,ipiv,ainv,info)
void cgetri_oop_batch_strided(const MKL_INT *n, const MKL_Complex8 *a, const MKL_INT *lda, const MKL_INT *stride_a, const MKL_INT *ipiv, const MKL_INT *stride_ipiv, MKL_Complex8 *ainv, const MKL_INT *ldainv, const MKL_INT *stride_ainv, const MKL_INT *batch_size, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(dgetri_oop_batch_strided,a,ipiv,ainv,info)
void dgetri_oop_batch_strided(const MKL_INT *n, const double *a, const MKL_INT *lda, const MKL_INT *stride_a, const MKL_INT *ipiv, const MKL_INT *stride_ipiv, double *ainv, const MKL_INT *ldainv, const MKL_INT *stride_ainv, const MKL_INT *batch_size, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(sgetri_oop_batch_strided,a,ipiv,ainv,info)
void sgetri_oop_batch_strided(const MKL_INT *n, const float *a, const MKL_INT *lda, const MKL_INT *stride_a, const MKL_INT *ipiv, const MKL_INT *stride_ipiv, float *ainv, const MKL_INT *ldainv, const MKL_INT *stride_ainv, const MKL_INT *batch_size, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zgetri_oop_batch_strided,a,ipiv,ainv,info)
void zgetri_oop_batch_strided(const MKL_INT *n, const MKL_Complex16 *a, const MKL_INT *lda, const MKL_INT *stride_a, const MKL_INT *ipiv, const MKL_INT *stride_ipiv, MKL_Complex16 *ainv, const MKL_INT *ldainv, const MKL_INT *stride_ainv, const MKL_INT *batch_size, MKL_INT *info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cgetri_oop_batch,a,ipiv,ainv,info)
void cgetri_oop_batch(const MKL_INT *n, const MKL_Complex8 **a, const MKL_INT *lda, const MKL_INT **ipiv, MKL_Complex8 **ainv, const MKL_INT *ldainv, const MKL_INT *group_count, const MKL_INT *group_size, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(dgetri_oop_batch,a,ipiv,ainv,info)
void dgetri_oop_batch(const MKL_INT *n, const double **a, const MKL_INT *lda, const MKL_INT **ipiv, double **ainv, const MKL_INT *ldainv, const MKL_INT *group_count, const MKL_INT *group_size, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(sgetri_oop_batch,a,ipiv,ainv,info)
void sgetri_oop_batch(const MKL_INT *n, const float **a, const MKL_INT *lda, const MKL_INT **ipiv, float **ainv, const MKL_INT *ldainv, const MKL_INT *group_count, const MKL_INT *group_size, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zgetri_oop_batch,a,ipiv,ainv,info)
void zgetri_oop_batch(const MKL_INT *n, const MKL_Complex16 **a, const MKL_INT *lda, const MKL_INT **ipiv, MKL_Complex16 **ainv, const MKL_INT *ldainv, const MKL_INT *group_count, const MKL_INT *group_size, MKL_INT *info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cgetrs_batch_strided,a,ipiv,b,info)
void cgetrs_batch_strided(const char *trans, const MKL_INT *n, const MKL_INT *nrhs, const MKL_Complex8 *a, const MKL_INT *lda, const MKL_INT *stride_a, const MKL_INT *ipiv, const MKL_INT *stride_ipiv, MKL_Complex8 *b, const MKL_INT *ldb, const MKL_INT *stride_b, const MKL_INT *batch_size, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(dgetrs_batch_strided,a,ipiv,b,info)
void dgetrs_batch_strided(const char *trans, const MKL_INT *n, const MKL_INT *nrhs, const double *a, const MKL_INT *lda, const MKL_INT *stride_a, const MKL_INT *ipiv, const MKL_INT *stride_ipiv, double *b, const MKL_INT *ldb, const MKL_INT *stride_b, const MKL_INT *batch_size, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(sgetrs_batch_strided,a,ipiv,b,info)
void sgetrs_batch_strided(const char *trans, const MKL_INT *n, const MKL_INT *nrhs, const float *a, const MKL_INT *lda, const MKL_INT *stride_a, const MKL_INT *ipiv, const MKL_INT *stride_ipiv, float *b, const MKL_INT *ldb, const MKL_INT *stride_b, const MKL_INT *batch_size, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zgetrs_batch_strided,a,ipiv,b,info)
void zgetrs_batch_strided(const char *trans, const MKL_INT *n, const MKL_INT *nrhs, const MKL_Complex16 *a, const MKL_INT *lda, const MKL_INT *stride_a, const MKL_INT *ipiv, const MKL_INT *stride_ipiv, MKL_Complex16 *b, const MKL_INT *ldb, const MKL_INT *stride_b, const MKL_INT *batch_size, MKL_INT *info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cpotrf,a,info)
void cpotrf(const char *uplo, const MKL_INT *n, MKL_Complex8 *a, const MKL_INT *lda, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(dpotrf,a,info)
void dpotrf(const char *uplo, const MKL_INT *n, double *a, const MKL_INT *lda, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(spotrf,a,info)
void spotrf(const char *uplo, const MKL_INT *n, float *a, const MKL_INT *lda, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zpotrf,a,info)
void zpotrf(const char *uplo, const MKL_INT *n, MKL_Complex16 *a, const MKL_INT *lda, MKL_INT *info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cpotri,a,info)
void cpotri(const char *uplo, const MKL_INT *n, MKL_Complex8 *a, const MKL_INT *lda, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(dpotri,a,info)
void dpotri(const char *uplo, const MKL_INT *n, double *a, const MKL_INT *lda, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(spotri,a,info)
void spotri(const char *uplo, const MKL_INT *n, float *a, const MKL_INT *lda, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zpotri,a,info)
void zpotri(const char *uplo, const MKL_INT *n, MKL_Complex16 *a, const MKL_INT *lda, MKL_INT *info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cpotrs,a,b,info)
void cpotrs(const char *uplo, const MKL_INT *n, const MKL_INT *nrhs, const MKL_Complex8 *a, const MKL_INT *lda, MKL_Complex8 *b, const MKL_INT *ldb, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(dpotrs,a,b,info)
void dpotrs(const char *uplo, const MKL_INT *n, const MKL_INT *nrhs, const double *a, const MKL_INT *lda, double *b, const MKL_INT *ldb, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(spotrs,a,b,info)
void spotrs(const char *uplo, const MKL_INT *n, const MKL_INT *nrhs, const float *a, const MKL_INT *lda, float *b, const MKL_INT *ldb, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zpotrs,a,b,info)
void zpotrs(const char *uplo, const MKL_INT *n, const MKL_INT *nrhs, const MKL_Complex16 *a, const MKL_INT *lda, MKL_Complex16 *b, const MKL_INT *ldb, MKL_INT *info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cgetrf_batch,a,ipiv,info)
void cgetrf_batch(const MKL_INT *m, const MKL_INT *n, MKL_Complex8 **a, const MKL_INT *lda, MKL_INT **ipiv, const MKL_INT *group_count, const MKL_INT *group_size, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(dgetrf_batch,a,ipiv,info)
void dgetrf_batch(const MKL_INT *m, const MKL_INT *n, double **a, const MKL_INT *lda, MKL_INT **ipiv, const MKL_INT *group_count, const MKL_INT *group_size, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(sgetrf_batch,a,ipiv,info)
void sgetrf_batch(const MKL_INT *m, const MKL_INT *n, float **a, const MKL_INT *lda, MKL_INT **ipiv, const MKL_INT *group_count, const MKL_INT *group_size, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zgetrf_batch,a,ipiv,info)
void zgetrf_batch(const MKL_INT *m, const MKL_INT *n, MKL_Complex16 **a, const MKL_INT *lda, MKL_INT **ipiv, const MKL_INT *group_count, const MKL_INT *group_size, MKL_INT *info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(cgetrfnp_batch,a,info)
void cgetrfnp_batch(const MKL_INT *m, const MKL_INT *n, MKL_Complex8 **a, const MKL_INT *lda, const MKL_INT *group_count, const MKL_INT *group_size, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(dgetrfnp_batch,a,info)
void dgetrfnp_batch(const MKL_INT *m, const MKL_INT *n, double **a, const MKL_INT *lda, const MKL_INT *group_count, const MKL_INT *group_size, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(sgetrfnp_batch,a,info)
void sgetrfnp_batch(const MKL_INT *m, const MKL_INT *n, float **a, const MKL_INT *lda, const MKL_INT *group_count, const MKL_INT *group_size, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zgetrfnp_batch,a,info)
void zgetrfnp_batch(const MKL_INT *m, const MKL_INT *n, MKL_Complex16 **a, const MKL_INT *lda, const MKL_INT *group_count, const MKL_INT *group_size, MKL_INT *info) NOTHROW;

#pragma omp MKL_LAPACK_DECLARE_VARIANT(dgels_batch_strided,a,b,info)
void dgels_batch_strided(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *nrhs, double *a, const MKL_INT *lda, const MKL_INT *stride_a, double *b, const MKL_INT *ldb, const MKL_INT *stride_b, const MKL_INT *batch_size, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(sgels_batch_strided,a,b,info)
void sgels_batch_strided(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *nrhs, float *a, const MKL_INT *lda, const MKL_INT *stride_a, float *b, const MKL_INT *ldb, const MKL_INT *stride_b, const MKL_INT *batch_size, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(zgels_batch_strided,a,b,info)
void zgels_batch_strided(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *nrhs, MKL_Complex16 *a, const MKL_INT *lda, const MKL_INT *stride_a, MKL_Complex16 *b, const MKL_INT *ldb, const MKL_INT *stride_b, const MKL_INT *batch_size, MKL_INT *info) NOTHROW;
#pragma omp MKL_LAPACK_DECLARE_VARIANT(cgels_batch_strided,a,b,info)
void cgels_batch_strided(const char *trans, const MKL_INT *m, const MKL_INT *n, const MKL_INT *nrhs, MKL_Complex8 *a, const MKL_INT *lda, const MKL_INT *stride_a, MKL_Complex8 *b, const MKL_INT *ldb, const MKL_INT *stride_b, const MKL_INT *batch_size, MKL_INT *info) NOTHROW;

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* _MKL_LAPACK_OMP_VARIANT_H_ */
