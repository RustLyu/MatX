!===============================================================================
! Copyright 2020-2022 Intel Corporation.
!
! This software and the related documents are Intel copyrighted  materials,  and
! your use of  them is  governed by the  express license  under which  they were
! provided to you (License).  Unless the License provides otherwise, you may not
! use, modify, copy, publish, distribute,  disclose or transmit this software or
! the related documents without Intel's prior written permission.
!
! This software and the related documents  are provided as  is,  with no express
! or implied  warranties,  other  than those  that are  expressly stated  in the
! License.
!===============================================================================

!  Content:
!      Intel(R) oneAPI Math Kernel Library (oneMKL) FORTRAN interface for
!      OpenMP offload for LAPACK
!*******************************************************************************

module onemkl_lapack_omp_offload_lp64

   include "mkl_lapack_omp_variant_lp64.f90"

   interface

      subroutine cgebrd(m, n, a, lda, d, e, tauq, taup, work, lwork,   &
                        info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      complex*8, intent(inout) :: a(lda,*)
      real, intent(out) :: d(*)
      real, intent(out) :: e(*)
      complex*8, intent(out) :: tauq(*)
      complex*8, intent(out) :: taup(*)
      complex*8, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_cgebrd_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_cgebrd_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,d,e,tauq,taup,work)
      end subroutine cgebrd

      subroutine dgebrd(m, n, a, lda, d, e, tauq, taup, work, lwork,   &
                        info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      double precision, intent(inout) :: a(lda,*)
      double precision, intent(out) :: d(*)
      double precision, intent(out) :: e(*)
      double precision, intent(out) :: tauq(*)
      double precision, intent(out) :: taup(*)
      double precision, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_dgebrd_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dgebrd_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,d,e,tauq,taup,work)
      end subroutine dgebrd

      subroutine sgebrd(m, n, a, lda, d, e, tauq, taup, work, lwork,   &
                        info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      real, intent(inout) :: a(lda,*)
      real, intent(out) :: d(*)
      real, intent(out) :: e(*)
      real, intent(out) :: tauq(*)
      real, intent(out) :: taup(*)
      real, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_sgebrd_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_sgebrd_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,d,e,tauq,taup,work)
      end subroutine sgebrd

      subroutine zgebrd(m, n, a, lda, d, e, tauq, taup, work, lwork,   &
                        info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      complex*16, intent(inout) :: a(lda,*)
      double precision, intent(out) :: d(*)
      double precision, intent(out) :: e(*)
      complex*16, intent(out) :: tauq(*)
      complex*16, intent(out) :: taup(*)
      complex*16, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_zgebrd_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_zgebrd_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,d,e,tauq,taup,work)
      end subroutine zgebrd

      subroutine cgeqrf(m, n, a, lda, tau, work, lwork, info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      complex*8, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      complex*8, intent(out) :: tau(*)
      complex*8, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
!$omp declare variant (mkl_lapack_cgeqrf_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_cgeqrf_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,tau,work,info)
      end subroutine cgeqrf

      subroutine dgeqrf(m, n, a, lda, tau, work, lwork, info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      double precision, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      double precision, intent(out) :: tau(*)
      double precision, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
!$omp declare variant (mkl_lapack_dgeqrf_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dgeqrf_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,tau,work,info)
      end subroutine dgeqrf

      subroutine sgeqrf(m, n, a, lda, tau, work, lwork, info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      real, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      real, intent(out) :: tau(*)
      real, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
!$omp declare variant (mkl_lapack_sgeqrf_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_sgeqrf_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,tau,work,info)
      end subroutine sgeqrf

      subroutine zgeqrf(m, n, a, lda, tau, work, lwork, info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      complex*16, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      complex*16, intent(out) :: tau(*)
      complex*16, intent(out) :: work(*)
      integer, intent(in) :: lwork
      integer, intent(out) :: info
!$omp declare variant (mkl_lapack_zgeqrf_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_zgeqrf_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,tau,work,info)
      end subroutine zgeqrf

      subroutine cgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,&
                        work, lwork, rwork, info) bind(c)
      character*1, intent(in) :: jobu
      character*1, intent(in) :: jobvt
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: ldu
      integer, intent(in) :: ldvt
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      complex*8, intent(inout) :: a(lda,*)
      real, intent(out) :: s(*)
      complex*8, intent(out) :: u(ldu,*)
      complex*8, intent(out) :: vt(ldvt,*)
      complex*8, intent(out) :: work(*)
      real, intent(out) :: rwork(*)
!$omp declare variant (mkl_lapack_cgesvd_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_cgesvd_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,s,u,vt,work,rwork)
      end subroutine cgesvd

      subroutine dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,&
                        work, lwork, info) bind(c)
      character*1, intent(in) :: jobu
      character*1, intent(in) :: jobvt
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: ldu
      integer, intent(in) :: ldvt
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      double precision, intent(inout) :: a(lda,*)
      double precision, intent(out) :: s(*)
      double precision, intent(out) :: u(ldu,*)
      double precision, intent(out) :: vt(ldvt,*)
      double precision, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_dgesvd_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dgesvd_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,s,u,vt,work)
      end subroutine dgesvd

      subroutine sgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,&
                        work, lwork, info) bind(c)
      character*1, intent(in) :: jobu
      character*1, intent(in) :: jobvt
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: ldu
      integer, intent(in) :: ldvt
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      real, intent(inout) :: a(lda,*)
      real, intent(out) :: s(*)
      real, intent(out) :: u(ldu,*)
      real, intent(out) :: vt(ldvt,*)
      real, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_sgesvd_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_sgesvd_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,s,u,vt,work)
      end subroutine sgesvd

      subroutine zgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt,&
                        work, lwork, rwork, info) bind(c)
      character*1, intent(in) :: jobu
      character*1, intent(in) :: jobvt
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: ldu
      integer, intent(in) :: ldvt
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      complex*16, intent(inout) :: a(lda,*)
      double precision, intent(out) :: s(*)
      complex*16, intent(out) :: u(ldu,*)
      complex*16, intent(out) :: vt(ldvt,*)
      complex*16, intent(out) :: work(*)
      double precision, intent(out) :: rwork(*)
!$omp declare variant (mkl_lapack_zgesvd_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_zgesvd_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,s,u,vt,work,rwork)
      end subroutine zgesvd

      subroutine cgetrf(m, n, a, lda, ipiv, info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(out) :: info
      complex*8, intent(inout) :: a(lda,*)
      integer, intent(out) :: ipiv(*)
!$omp declare variant (mkl_lapack_cgetrf_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_cgetrf_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,ipiv)
      end subroutine cgetrf

      subroutine dgetrf(m, n, a, lda, ipiv, info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(out) :: info
      double precision, intent(inout) :: a(lda,*)
      integer, intent(out) :: ipiv(*)
!$omp declare variant (mkl_lapack_dgetrf_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dgetrf_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,ipiv)
      end subroutine dgetrf

      subroutine sgetrf(m, n, a, lda, ipiv, info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(out) :: info
      real, intent(inout) :: a(lda,*)
      integer, intent(out) :: ipiv(*)
!$omp declare variant (mkl_lapack_sgetrf_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_sgetrf_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,ipiv)
      end subroutine sgetrf

      subroutine zgetrf(m, n, a, lda, ipiv, info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(out) :: info
      complex*16, intent(inout) :: a(lda,*)
      integer, intent(out) :: ipiv(*)
!$omp declare variant (mkl_lapack_zgetrf_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_zgetrf_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,ipiv)
      end subroutine zgetrf

      subroutine cgetrf_batch_strided(m, n, a, lda, stride_a, ipiv,    &
                                      stride_ipiv, batch_size,         &
                                      info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: stride_ipiv
      integer, intent(in) :: batch_size
      complex*8, intent(inout) :: a(stride_a,*)
      integer, intent(out) :: ipiv(stride_ipiv,*)
      integer, intent(out) :: info(*)
!$omp declare variant (mkl_lapack_cgetrf_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_cgetrf_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,ipiv,info)
      end subroutine cgetrf_batch_strided

      subroutine dgetrf_batch_strided(m, n, a, lda, stride_a, ipiv,    &
                                      stride_ipiv, batch_size,         &
                                      info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: stride_ipiv
      integer, intent(in) :: batch_size
      double precision, intent(inout) :: a(stride_a,*)
      integer, intent(out) :: ipiv(stride_ipiv,*)
      integer, intent(out) :: info(*)
!$omp declare variant (mkl_lapack_dgetrf_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dgetrf_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,ipiv,info)
      end subroutine dgetrf_batch_strided

      subroutine sgetrf_batch_strided(m, n, a, lda, stride_a, ipiv,    &
                                      stride_ipiv, batch_size,         &
                                      info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: stride_ipiv
      integer, intent(in) :: batch_size
      real, intent(inout) :: a(stride_a,*)
      integer, intent(out) :: ipiv(stride_ipiv,*)
      integer, intent(out) :: info(*)
!$omp declare variant (mkl_lapack_sgetrf_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_sgetrf_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,ipiv,info)
      end subroutine sgetrf_batch_strided

      subroutine zgetrf_batch_strided(m, n, a, lda, stride_a, ipiv,    &
                                      stride_ipiv, batch_size,         &
                                      info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: stride_ipiv
      integer, intent(in) :: batch_size
      complex*16, intent(inout) :: a(stride_a,*)
      integer, intent(out) :: ipiv(stride_ipiv,*)
      integer, intent(out) :: info(*)
!$omp declare variant (mkl_lapack_zgetrf_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_zgetrf_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,ipiv,info)
      end subroutine zgetrf_batch_strided

      subroutine cgetri(n, a, lda, ipiv, work, lwork, info) bind(c)
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      complex*8, intent(inout) :: a(lda,*)
      integer, intent(in) :: ipiv(*)
      complex*8, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_cgetri_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_cgetri_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,ipiv,work)
      end subroutine cgetri

      subroutine dgetri(n, a, lda, ipiv, work, lwork, info) bind(c)
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      double precision, intent(inout) :: a(lda,*)
      integer, intent(in) :: ipiv(*)
      double precision, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_dgetri_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dgetri_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,ipiv,work)
      end subroutine dgetri

      subroutine sgetri(n, a, lda, ipiv, work, lwork, info) bind(c)
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      real, intent(inout) :: a(lda,*)
      integer, intent(in) :: ipiv(*)
      real, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_sgetri_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_sgetri_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,ipiv,work)
      end subroutine sgetri

      subroutine zgetri(n, a, lda, ipiv, work, lwork, info) bind(c)
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      complex*16, intent(inout) :: a(lda,*)
      integer, intent(in) :: ipiv(*)
      complex*16, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_zgetri_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_zgetri_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,ipiv,work)
      end subroutine zgetri

      subroutine cgetri_oop_batch_strided(n, a, lda, stride_a, ipiv,   &
                                          stride_ipiv, ainv, ldainv,   &
                                          stride_ainv, batch_size,     &
                                          info) bind(c)
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: stride_ipiv
      integer, intent(in) :: ldainv
      integer, intent(in) :: stride_ainv
      integer, intent(in) :: batch_size
      complex*8, intent(in) :: a(stride_a,*)
      integer, intent(in) :: ipiv(stride_ipiv,*)
      complex*8, intent(inout) :: ainv(stride_ainv,*)
      integer, intent(out) :: info(*)
!$omp declare variant (mkl_lapack_cgetri_oop_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_cgetri_oop_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,ipiv,ainv,info)
      end subroutine cgetri_oop_batch_strided

      subroutine dgetri_oop_batch_strided(n, a, lda, stride_a, ipiv,   &
                                          stride_ipiv, ainv, ldainv,   &
                                          stride_ainv, batch_size,     &
                                          info) bind(c)
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: stride_ipiv
      integer, intent(in) :: ldainv
      integer, intent(in) :: stride_ainv
      integer, intent(in) :: batch_size
      double precision, intent(in) :: a(stride_a,*)
      integer, intent(in) :: ipiv(stride_ipiv,*)
      double precision, intent(inout) :: ainv(stride_ainv,*)
      integer, intent(out) :: info(*)
!$omp declare variant (mkl_lapack_dgetri_oop_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dgetri_oop_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,ipiv,ainv,info)
      end subroutine dgetri_oop_batch_strided

      subroutine sgetri_oop_batch_strided(n, a, lda, stride_a, ipiv,   &
                                          stride_ipiv, ainv, ldainv,   &
                                          stride_ainv, batch_size,     &
                                          info) bind(c)
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: stride_ipiv
      integer, intent(in) :: ldainv
      integer, intent(in) :: stride_ainv
      integer, intent(in) :: batch_size
      real, intent(in) :: a(stride_a,*)
      integer, intent(in) :: ipiv(stride_ipiv,*)
      real, intent(inout) :: ainv(stride_ainv,*)
      integer, intent(out) :: info(*)
!$omp declare variant (mkl_lapack_sgetri_oop_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_sgetri_oop_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,ipiv,ainv,info)
      end subroutine sgetri_oop_batch_strided

      subroutine zgetri_oop_batch_strided(n, a, lda, stride_a, ipiv,   &
                                          stride_ipiv, ainv, ldainv,   &
                                          stride_ainv, batch_size,     &
                                          info) bind(c)
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: stride_ipiv
      integer, intent(in) :: ldainv
      integer, intent(in) :: stride_ainv
      integer, intent(in) :: batch_size
      complex*16, intent(in) :: a(stride_a,*)
      integer, intent(in) :: ipiv(stride_ipiv,*)
      complex*16, intent(inout) :: ainv(stride_ainv,*)
      integer, intent(out) :: info(*)
!$omp declare variant (mkl_lapack_zgetri_oop_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_zgetri_oop_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,ipiv,ainv,info)
      end subroutine zgetri_oop_batch_strided

      subroutine cgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb,          &
                        info) bind(c)
      character*1, intent(in) :: trans
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      integer, intent(in) :: lda
      integer, intent(in) :: ldb
      integer, intent(out) :: info
      complex*8, intent(in) :: a(lda,*)
      integer, intent(in) :: ipiv(*)
      complex*8, intent(inout) :: b(ldb,*)
!$omp declare variant (mkl_lapack_cgetrs_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_cgetrs_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,ipiv,b)
      end subroutine cgetrs

      subroutine dgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb,          &
                        info) bind(c)
      character*1, intent(in) :: trans
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      integer, intent(in) :: lda
      integer, intent(in) :: ldb
      integer, intent(out) :: info
      double precision, intent(in) :: a(lda,*)
      integer, intent(in) :: ipiv(*)
      double precision, intent(inout) :: b(ldb,*)
!$omp declare variant (mkl_lapack_dgetrs_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dgetrs_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,ipiv,b)
      end subroutine dgetrs

      subroutine sgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb,          &
                        info) bind(c)
      character*1, intent(in) :: trans
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      integer, intent(in) :: lda
      integer, intent(in) :: ldb
      integer, intent(out) :: info
      real, intent(in) :: a(lda,*)
      integer, intent(in) :: ipiv(*)
      real, intent(inout) :: b(ldb,*)
!$omp declare variant (mkl_lapack_sgetrs_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_sgetrs_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,ipiv,b)
      end subroutine sgetrs

      subroutine zgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb,          &
                        info) bind(c)
      character*1, intent(in) :: trans
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      integer, intent(in) :: lda
      integer, intent(in) :: ldb
      integer, intent(out) :: info
      complex*16, intent(in) :: a(lda,*)
      integer, intent(in) :: ipiv(*)
      complex*16, intent(inout) :: b(ldb,*)
!$omp declare variant (mkl_lapack_zgetrs_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_zgetrs_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,ipiv,b)
      end subroutine zgetrs

      subroutine cheev(jobz, uplo, n, a, lda, w, work, lwork, rwork,   &
                       info) bind(c)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      complex*8, intent(inout) :: a(lda,*)
      real, intent(out) :: w(*)
      complex*8, intent(out) :: work(*)
      real, intent(out) :: rwork(*)
!$omp declare variant (mkl_lapack_cheev_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_cheev_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,w,work,rwork)
      end subroutine cheev

      subroutine zheev(jobz, uplo, n, a, lda, w, work, lwork, rwork,   &
                       info) bind(c)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      complex*16, intent(inout) :: a(lda,*)
      double precision, intent(out) :: w(*)
      complex*16, intent(out) :: work(*)
      double precision, intent(out) :: rwork(*)
!$omp declare variant (mkl_lapack_zheev_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_zheev_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,w,work,rwork)
      end subroutine zheev

      subroutine cheevd(jobz, uplo, n, a, lda, w, work, lwork, rwork,  &
                        lrwork, iwork, liwork, info) bind(c)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(in) :: lrwork
      integer, intent(in) :: liwork
      integer, intent(out) :: info
      complex*8, intent(inout) :: a(lda,*)
      real, intent(out) :: w(*)
      complex*8, intent(out) :: work(*)
      real, intent(out) :: rwork(*)
      integer, intent(out) :: iwork(*)
!$omp declare variant (mkl_lapack_cheevd_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_cheevd_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,w,work,rwork,iwork)
      end subroutine cheevd

      subroutine zheevd(jobz, uplo, n, a, lda, w, work, lwork, rwork,  &
                        lrwork, iwork, liwork, info) bind(c)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(in) :: lrwork
      integer, intent(in) :: liwork
      integer, intent(out) :: info
      complex*16, intent(inout) :: a(lda,*)
      double precision, intent(out) :: w(*)
      complex*16, intent(out) :: work(*)
      double precision, intent(out) :: rwork(*)
      integer, intent(out) :: iwork(*)
!$omp declare variant (mkl_lapack_zheevd_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_zheevd_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,w,work,rwork,iwork)
      end subroutine zheevd

      subroutine cheevx(jobz, range, uplo, n, a, lda, vl, vu, il, iu,  &
                        abstol, m, w, z, ldz, work, lwork, rwork,      &
                        iwork, ifail, info) bind(c)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: range
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      real, intent(in) :: vl
      real, intent(in) :: vu
      integer, intent(in) :: il
      integer, intent(in) :: iu
      real, intent(in) :: abstol
      integer, intent(out) :: m
      integer, intent(in) :: ldz
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      complex*8, intent(inout) :: a(lda,*)
      real, intent(out) :: w(*)
      complex*8, intent(out) :: z(ldz,*)
      complex*8, intent(out) :: work(*)
      real, intent(out) :: rwork(*)
      integer, intent(out) :: iwork(*)
      integer, intent(out) :: ifail(*)
!$omp declare variant (mkl_lapack_cheevx_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_cheevx_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:m,info,a,w,z,work,rwork,iwork,ifail)
      end subroutine cheevx

      subroutine zheevx(jobz, range, uplo, n, a, lda, vl, vu, il, iu,  &
                        abstol, m, w, z, ldz, work, lwork, rwork,      &
                        iwork, ifail, info) bind(c)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: range
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      double precision, intent(in) :: vl
      double precision, intent(in) :: vu
      integer, intent(in) :: il
      integer, intent(in) :: iu
      double precision, intent(in) :: abstol
      integer, intent(out) :: m
      integer, intent(in) :: ldz
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      complex*16, intent(inout) :: a(lda,*)
      double precision, intent(out) :: w(*)
      complex*16, intent(out) :: z(ldz,*)
      complex*16, intent(out) :: work(*)
      double precision, intent(out) :: rwork(*)
      integer, intent(out) :: iwork(*)
      integer, intent(out) :: ifail(*)
!$omp declare variant (mkl_lapack_zheevx_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_zheevx_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:m,info,a,w,z,work,rwork,iwork,ifail)
      end subroutine zheevx

      subroutine chegvd(itype, jobz, uplo, n, a, lda, b, ldb, w, work, &
                        lwork, rwork, lrwork, iwork, liwork,           &
                        info) bind(c)
      integer, intent(in) :: itype
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: ldb
      integer, intent(in) :: lwork
      integer, intent(in) :: lrwork
      integer, intent(in) :: liwork
      integer, intent(out) :: info
      complex*8, intent(inout) :: a(lda,*)
      complex*8, intent(inout) :: b(ldb,*)
      real, intent(out) :: w(*)
      complex*8, intent(out) :: work(*)
      real, intent(out) :: rwork(*)
      integer, intent(out) :: iwork(*)
!$omp declare variant (mkl_lapack_chegvd_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_chegvd_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,b,w,work,rwork,iwork)
      end subroutine chegvd

      subroutine zhegvd(itype, jobz, uplo, n, a, lda, b, ldb, w, work, &
                        lwork, rwork, lrwork, iwork, liwork,           &
                        info) bind(c)
      integer, intent(in) :: itype
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: ldb
      integer, intent(in) :: lwork
      integer, intent(in) :: lrwork
      integer, intent(in) :: liwork
      integer, intent(out) :: info
      complex*16, intent(inout) :: a(lda,*)
      complex*16, intent(inout) :: b(ldb,*)
      double precision, intent(out) :: w(*)
      complex*16, intent(out) :: work(*)
      double precision, intent(out) :: rwork(*)
      integer, intent(out) :: iwork(*)
!$omp declare variant (mkl_lapack_zhegvd_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_zhegvd_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,b,w,work,rwork,iwork)
      end subroutine zhegvd

      subroutine chegvx(itype, jobz, range, uplo, n, a, lda, b, ldb,   &
                        vl, vu, il, iu, abstol, m, w, z, ldz, work,    &
                        lwork, rwork, iwork, ifail, info) bind(c)
      integer, intent(in) :: itype
      character*1, intent(in) :: jobz
      character*1, intent(in) :: range
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: ldb
      real, intent(in) :: vl
      real, intent(in) :: vu
      integer, intent(in) :: il
      integer, intent(in) :: iu
      real, intent(in) :: abstol
      integer, intent(out) :: m
      integer, intent(in) :: ldz
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      complex*8, intent(inout) :: a(lda,*)
      complex*8, intent(inout) :: b(ldb,*)
      real, intent(out) :: w(*)
      complex*8, intent(out) :: z(ldz,*)
      complex*8, intent(out) :: work(*)
      real, intent(out) :: rwork(*)
      integer, intent(out) :: iwork(*)
      integer, intent(out) :: ifail(*)
!$omp declare variant (mkl_lapack_chegvx_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_chegvx_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:m,info,a,b,w,z,work,rwork,iwork,ifail)
      end subroutine chegvx

      subroutine zhegvx(itype, jobz, range, uplo, n, a, lda, b, ldb,   &
                        vl, vu, il, iu, abstol, m, w, z, ldz, work,    &
                        lwork, rwork, iwork, ifail, info) bind(c)
      integer, intent(in) :: itype
      character*1, intent(in) :: jobz
      character*1, intent(in) :: range
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: ldb
      double precision, intent(in) :: vl
      double precision, intent(in) :: vu
      integer, intent(in) :: il
      integer, intent(in) :: iu
      double precision, intent(in) :: abstol
      integer, intent(out) :: m
      integer, intent(in) :: ldz
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      complex*16, intent(inout) :: a(lda,*)
      complex*16, intent(inout) :: b(ldb,*)
      double precision, intent(out) :: w(*)
      complex*16, intent(out) :: z(ldz,*)
      complex*16, intent(out) :: work(*)
      double precision, intent(out) :: rwork(*)
      integer, intent(out) :: iwork(*)
      integer, intent(out) :: ifail(*)
!$omp declare variant (mkl_lapack_zhegvx_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_zhegvx_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:m,info,a,b,w,z,work,rwork,iwork,ifail)
      end subroutine zhegvx

      subroutine chetrd(uplo, n, a, lda, d, e, tau, work, lwork,       &
                        info) bind(c)
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      complex*8, intent(inout) :: a(lda,*)
      real, intent(out) :: d(*)
      real, intent(out) :: e(*)
      complex*8, intent(out) :: tau(*)
      complex*8, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_chetrd_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_chetrd_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,d,e,tau,work)
      end subroutine chetrd

      subroutine zhetrd(uplo, n, a, lda, d, e, tau, work, lwork,       &
                        info) bind(c)
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      complex*16, intent(inout) :: a(lda,*)
      double precision, intent(out) :: d(*)
      double precision, intent(out) :: e(*)
      complex*16, intent(out) :: tau(*)
      complex*16, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_zhetrd_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_zhetrd_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,d,e,tau,work)
      end subroutine zhetrd

      subroutine dorgqr(m, n, k, a, lda, tau, work, lwork,             &
                        info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: k
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      double precision, intent(inout) :: a(lda,*)
      double precision, intent(in) :: tau(*)
      double precision, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_dorgqr_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dorgqr_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,tau,work)
      end subroutine dorgqr

      subroutine sorgqr(m, n, k, a, lda, tau, work, lwork,             &
                        info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: k
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      real, intent(inout) :: a(lda,*)
      real, intent(in) :: tau(*)
      real, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_sorgqr_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_sorgqr_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,tau,work)
      end subroutine sorgqr

      subroutine dormqr(side, trans, m, n, k, a, lda, tau, c, ldc,     &
                        work, lwork, info) bind(c)
      character*1, intent(in) :: side
      character*1, intent(in) :: trans
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: k
      integer, intent(in) :: lda
      integer, intent(in) :: ldc
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      double precision, intent(in) :: a(lda,*)
      double precision, intent(in) :: tau(*)
      double precision, intent(inout) :: c(ldc,*)
      double precision, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_dormqr_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dormqr_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,tau,c,work)
      end subroutine dormqr

      subroutine sormqr(side, trans, m, n, k, a, lda, tau, c, ldc,     &
                        work, lwork, info) bind(c)
      character*1, intent(in) :: side
      character*1, intent(in) :: trans
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: k
      integer, intent(in) :: lda
      integer, intent(in) :: ldc
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      real, intent(in) :: a(lda,*)
      real, intent(in) :: tau(*)
      real, intent(inout) :: c(ldc,*)
      real, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_sormqr_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_sormqr_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,tau,c,work)
      end subroutine sormqr

      subroutine csteqr(compz, n, d, e, z, ldz, work, info) bind(c)
      character*1, intent(in) :: compz
      integer, intent(in) :: n
      integer, intent(in) :: ldz
      integer, intent(out) :: info
      real, intent(inout) :: d(*)
      real, intent(inout) :: e(*)
      complex*8, intent(inout) :: z(ldz,*)
      real, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_csteqr_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_csteqr_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,d,e,z,work)
      end subroutine csteqr

      subroutine dsteqr(compz, n, d, e, z, ldz, work, info) bind(c)
      character*1, intent(in) :: compz
      integer, intent(in) :: n
      integer, intent(in) :: ldz
      integer, intent(out) :: info
      double precision, intent(inout) :: d(*)
      double precision, intent(inout) :: e(*)
      double precision, intent(inout) :: z(ldz,*)
      double precision, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_dsteqr_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dsteqr_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,d,e,z,work)
      end subroutine dsteqr

      subroutine ssteqr(compz, n, d, e, z, ldz, work, info) bind(c)
      character*1, intent(in) :: compz
      integer, intent(in) :: n
      integer, intent(in) :: ldz
      integer, intent(out) :: info
      real, intent(inout) :: d(*)
      real, intent(inout) :: e(*)
      real, intent(inout) :: z(ldz,*)
      real, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_ssteqr_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_ssteqr_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,d,e,z,work)
      end subroutine ssteqr

      subroutine zsteqr(compz, n, d, e, z, ldz, work, info) bind(c)
      character*1, intent(in) :: compz
      integer, intent(in) :: n
      integer, intent(in) :: ldz
      integer, intent(out) :: info
      double precision, intent(inout) :: d(*)
      double precision, intent(inout) :: e(*)
      complex*16, intent(inout) :: z(ldz,*)
      double precision, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_zsteqr_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_zsteqr_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,d,e,z,work)
      end subroutine zsteqr

      subroutine dsyev(jobz, uplo, n, a, lda, w, work, lwork,          &
                       info) bind(c)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      double precision, intent(inout) :: a(lda,*)
      double precision, intent(out) :: w(*)
      double precision, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_dsyev_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dsyev_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,w,work)
      end subroutine dsyev

      subroutine ssyev(jobz, uplo, n, a, lda, w, work, lwork,          &
                       info) bind(c)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      real, intent(inout) :: a(lda,*)
      real, intent(out) :: w(*)
      real, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_ssyev_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_ssyev_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,w,work)
      end subroutine ssyev

      subroutine dsyevd(jobz, uplo, n, a, lda, w, work, lwork, iwork,  &
                        liwork, info) bind(c)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(in) :: liwork
      integer, intent(out) :: info
      double precision, intent(inout) :: a(lda,*)
      double precision, intent(out) :: w(*)
      double precision, intent(out) :: work(*)
      integer, intent(out) :: iwork(*)
!$omp declare variant (mkl_lapack_dsyevd_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dsyevd_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,w,work,iwork)
      end subroutine dsyevd

      subroutine ssyevd(jobz, uplo, n, a, lda, w, work, lwork, iwork,  &
                        liwork, info) bind(c)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(in) :: liwork
      integer, intent(out) :: info
      real, intent(inout) :: a(lda,*)
      real, intent(out) :: w(*)
      real, intent(out) :: work(*)
      integer, intent(out) :: iwork(*)
!$omp declare variant (mkl_lapack_ssyevd_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_ssyevd_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,w,work,iwork)
      end subroutine ssyevd

      subroutine dsyevx(jobz, range, uplo, n, a, lda, vl, vu, il, iu,  &
                        abstol, m, w, z, ldz, work, lwork, iwork,      &
                        ifail, info) bind(c)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: range
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      double precision, intent(in) :: vl
      double precision, intent(in) :: vu
      integer, intent(in) :: il
      integer, intent(in) :: iu
      double precision, intent(in) :: abstol
      integer, intent(out) :: m
      integer, intent(in) :: ldz
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      double precision, intent(inout) :: a(lda,*)
      double precision, intent(out) :: w(*)
      double precision, intent(out) :: z(ldz,*)
      double precision, intent(out) :: work(*)
      integer, intent(out) :: iwork(*)
      integer, intent(out) :: ifail(*)
!$omp declare variant (mkl_lapack_dsyevx_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dsyevx_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:m,info,a,w,z,work,iwork,ifail)
      end subroutine dsyevx

      subroutine ssyevx(jobz, range, uplo, n, a, lda, vl, vu, il, iu,  &
                        abstol, m, w, z, ldz, work, lwork, iwork,      &
                        ifail, info) bind(c)
      character*1, intent(in) :: jobz
      character*1, intent(in) :: range
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      real, intent(in) :: vl
      real, intent(in) :: vu
      integer, intent(in) :: il
      integer, intent(in) :: iu
      real, intent(in) :: abstol
      integer, intent(out) :: m
      integer, intent(in) :: ldz
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      real, intent(inout) :: a(lda,*)
      real, intent(out) :: w(*)
      real, intent(out) :: z(ldz,*)
      real, intent(out) :: work(*)
      integer, intent(out) :: iwork(*)
      integer, intent(out) :: ifail(*)
!$omp declare variant (mkl_lapack_ssyevx_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_ssyevx_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:m,info,a,w,z,work,iwork,ifail)
      end subroutine ssyevx

      subroutine dsygvd(itype, jobz, uplo, n, a, lda, b, ldb, w, work, &
                        lwork, iwork, liwork, info) bind(c)
      integer, intent(in) :: itype
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: ldb
      integer, intent(in) :: lwork
      integer, intent(in) :: liwork
      integer, intent(out) :: info
      double precision, intent(inout) :: a(lda,*)
      double precision, intent(inout) :: b(ldb,*)
      double precision, intent(out) :: w(*)
      double precision, intent(out) :: work(*)
      integer, intent(out) :: iwork(*)
!$omp declare variant (mkl_lapack_dsygvd_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dsygvd_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,b,w,work,iwork)
      end subroutine dsygvd

      subroutine ssygvd(itype, jobz, uplo, n, a, lda, b, ldb, w, work, &
                        lwork, iwork, liwork, info) bind(c)
      integer, intent(in) :: itype
      character*1, intent(in) :: jobz
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: ldb
      integer, intent(in) :: lwork
      integer, intent(in) :: liwork
      integer, intent(out) :: info
      real, intent(inout) :: a(lda,*)
      real, intent(inout) :: b(ldb,*)
      real, intent(out) :: w(*)
      real, intent(out) :: work(*)
      integer, intent(out) :: iwork(*)
!$omp declare variant (mkl_lapack_ssygvd_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_ssygvd_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,b,w,work,iwork)
      end subroutine ssygvd

      subroutine dsygvx(itype, jobz, range, uplo, n, a, lda, b, ldb,   &
                        vl, vu, il, iu, abstol, m, w, z, ldz, work,    &
                        lwork, iwork, ifail, info) bind(c)
      integer, intent(in) :: itype
      character*1, intent(in) :: jobz
      character*1, intent(in) :: range
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: ldb
      double precision, intent(in) :: vl
      double precision, intent(in) :: vu
      integer, intent(in) :: il
      integer, intent(in) :: iu
      double precision, intent(in) :: abstol
      integer, intent(out) :: m
      integer, intent(in) :: ldz
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      double precision, intent(inout) :: a(lda,*)
      double precision, intent(inout) :: b(ldb,*)
      double precision, intent(out) :: w(*)
      double precision, intent(out) :: z(ldz,*)
      double precision, intent(out) :: work(*)
      integer, intent(out) :: iwork(*)
      integer, intent(out) :: ifail(*)
!$omp declare variant (mkl_lapack_dsygvx_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dsygvx_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:m,info,a,b,w,z,work,iwork,ifail)
      end subroutine dsygvx

      subroutine ssygvx(itype, jobz, range, uplo, n, a, lda, b, ldb,   &
                        vl, vu, il, iu, abstol, m, w, z, ldz, work,    &
                        lwork, iwork, ifail, info) bind(c)
      integer, intent(in) :: itype
      character*1, intent(in) :: jobz
      character*1, intent(in) :: range
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: ldb
      real, intent(in) :: vl
      real, intent(in) :: vu
      integer, intent(in) :: il
      integer, intent(in) :: iu
      real, intent(in) :: abstol
      integer, intent(out) :: m
      integer, intent(in) :: ldz
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      real, intent(inout) :: a(lda,*)
      real, intent(inout) :: b(ldb,*)
      real, intent(out) :: w(*)
      real, intent(out) :: z(ldz,*)
      real, intent(out) :: work(*)
      integer, intent(out) :: iwork(*)
      integer, intent(out) :: ifail(*)
!$omp declare variant (mkl_lapack_ssygvx_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_ssygvx_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:m,info,a,b,w,z,work,iwork,ifail)
      end subroutine ssygvx

      subroutine dsytrd(uplo, n, a, lda, d, e, tau, work, lwork,       &
                        info) bind(c)
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      double precision, intent(inout) :: a(lda,*)
      double precision, intent(out) :: d(*)
      double precision, intent(out) :: e(*)
      double precision, intent(out) :: tau(*)
      double precision, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_dsytrd_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dsytrd_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,d,e,tau,work)
      end subroutine dsytrd

      subroutine ssytrd(uplo, n, a, lda, d, e, tau, work, lwork,       &
                        info) bind(c)
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      real, intent(inout) :: a(lda,*)
      real, intent(out) :: d(*)
      real, intent(out) :: e(*)
      real, intent(out) :: tau(*)
      real, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_ssytrd_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_ssytrd_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,d,e,tau,work)
      end subroutine ssytrd

      subroutine ctrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb,    &
                        info) bind(c)
      character*1, intent(in) :: uplo
      character*1, intent(in) :: trans
      character*1, intent(in) :: diag
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      integer, intent(in) :: lda
      integer, intent(in) :: ldb
      integer, intent(out) :: info
      complex*8, intent(in) :: a(lda,*)
      complex*8, intent(inout) :: b(ldb,*)
!$omp declare variant (mkl_lapack_ctrtrs_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_ctrtrs_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,b)
      end subroutine ctrtrs

      subroutine dtrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb,    &
                        info) bind(c)
      character*1, intent(in) :: uplo
      character*1, intent(in) :: trans
      character*1, intent(in) :: diag
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      integer, intent(in) :: lda
      integer, intent(in) :: ldb
      integer, intent(out) :: info
      double precision, intent(in) :: a(lda,*)
      double precision, intent(inout) :: b(ldb,*)
!$omp declare variant (mkl_lapack_dtrtrs_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dtrtrs_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,b)
      end subroutine dtrtrs

      subroutine strtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb,    &
                        info) bind(c)
      character*1, intent(in) :: uplo
      character*1, intent(in) :: trans
      character*1, intent(in) :: diag
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      integer, intent(in) :: lda
      integer, intent(in) :: ldb
      integer, intent(out) :: info
      real, intent(in) :: a(lda,*)
      real, intent(inout) :: b(ldb,*)
!$omp declare variant (mkl_lapack_strtrs_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_strtrs_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,b)
      end subroutine strtrs

      subroutine ztrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb,    &
                        info) bind(c)
      character*1, intent(in) :: uplo
      character*1, intent(in) :: trans
      character*1, intent(in) :: diag
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      integer, intent(in) :: lda
      integer, intent(in) :: ldb
      integer, intent(out) :: info
      complex*16, intent(in) :: a(lda,*)
      complex*16, intent(inout) :: b(ldb,*)
!$omp declare variant (mkl_lapack_ztrtrs_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_ztrtrs_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,b)
      end subroutine ztrtrs

      subroutine cungqr(m, n, k, a, lda, tau, work, lwork,             &
                        info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: k
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      complex*8, intent(inout) :: a(lda,*)
      complex*8, intent(in) :: tau(*)
      complex*8, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_cungqr_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_cungqr_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,tau,work)
      end subroutine cungqr

      subroutine zungqr(m, n, k, a, lda, tau, work, lwork,             &
                        info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: k
      integer, intent(in) :: lda
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      complex*16, intent(inout) :: a(lda,*)
      complex*16, intent(in) :: tau(*)
      complex*16, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_zungqr_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_zungqr_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,tau,work)
      end subroutine zungqr

      subroutine cunmqr(side, trans, m, n, k, a, lda, tau, c, ldc,     &
                        work, lwork, info) bind(c)
      character*1, intent(in) :: side
      character*1, intent(in) :: trans
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: k
      integer, intent(in) :: lda
      integer, intent(in) :: ldc
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      complex*8, intent(in) :: a(lda,*)
      complex*8, intent(in) :: tau(*)
      complex*8, intent(inout) :: c(ldc,*)
      complex*8, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_cunmqr_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_cunmqr_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,tau,c,work)
      end subroutine cunmqr

      subroutine zunmqr(side, trans, m, n, k, a, lda, tau, c, ldc,     &
                        work, lwork, info) bind(c)
      character*1, intent(in) :: side
      character*1, intent(in) :: trans
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: k
      integer, intent(in) :: lda
      integer, intent(in) :: ldc
      integer, intent(in) :: lwork
      integer, intent(out) :: info
      complex*16, intent(in) :: a(lda,*)
      complex*16, intent(in) :: tau(*)
      complex*16, intent(inout) :: c(ldc,*)
      complex*16, intent(out) :: work(*)
!$omp declare variant (mkl_lapack_zunmqr_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_zunmqr_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:info,a,tau,c,work)
      end subroutine zunmqr

      subroutine cgetrs_batch_strided(trans, n, nrhs, a, lda, stride_a,&
                                      ipiv, stride_ipiv, b, ldb,       &
                                      stride_b, batch_size,            &
                                      info) bind(c)
      character*1, intent(in) :: trans
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: stride_ipiv
      integer, intent(in) :: ldb
      integer, intent(in) :: stride_b
      integer, intent(in) :: batch_size
      complex*8, intent(in) :: a(stride_a,*)
      integer, intent(in) :: ipiv(stride_ipiv,*)
      complex*8, intent(inout) :: b(stride_b,*)
      integer, intent(out) :: info(*)
!$omp declare variant (mkl_lapack_cgetrs_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_cgetrs_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,ipiv,b,info)
      end subroutine cgetrs_batch_strided

      subroutine dgetrs_batch_strided(trans, n, nrhs, a, lda, stride_a,&
                                      ipiv, stride_ipiv, b, ldb,       &
                                      stride_b, batch_size,            &
                                      info) bind(c)
      character*1, intent(in) :: trans
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: stride_ipiv
      integer, intent(in) :: ldb
      integer, intent(in) :: stride_b
      integer, intent(in) :: batch_size
      double precision, intent(in) :: a(stride_a,*)
      integer, intent(in) :: ipiv(stride_ipiv,*)
      double precision, intent(inout) :: b(stride_b,*)
      integer, intent(out) :: info(*)
!$omp declare variant (mkl_lapack_dgetrs_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dgetrs_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,ipiv,b,info)
      end subroutine dgetrs_batch_strided

      subroutine sgetrs_batch_strided(trans, n, nrhs, a, lda, stride_a,&
                                      ipiv, stride_ipiv, b, ldb,       &
                                      stride_b, batch_size,            &
                                      info) bind(c)
      character*1, intent(in) :: trans
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: stride_ipiv
      integer, intent(in) :: ldb
      integer, intent(in) :: stride_b
      integer, intent(in) :: batch_size
      real, intent(in) :: a(stride_a,*)
      integer, intent(in) :: ipiv(stride_ipiv,*)
      real, intent(inout) :: b(stride_b,*)
      integer, intent(out) :: info(*)
!$omp declare variant (mkl_lapack_sgetrs_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_sgetrs_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,ipiv,b,info)
      end subroutine sgetrs_batch_strided

      subroutine zgetrs_batch_strided(trans, n, nrhs, a, lda, stride_a,&
                                      ipiv, stride_ipiv, b, ldb,       &
                                      stride_b, batch_size,            &
                                      info) bind(c)
      character*1, intent(in) :: trans
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: stride_ipiv
      integer, intent(in) :: ldb
      integer, intent(in) :: stride_b
      integer, intent(in) :: batch_size
      complex*16, intent(in) :: a(stride_a,*)
      integer, intent(in) :: ipiv(stride_ipiv,*)
      complex*16, intent(inout) :: b(stride_b,*)
      integer, intent(out) :: info(*)
!$omp declare variant (mkl_lapack_zgetrs_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_zgetrs_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,ipiv,b,info)
      end subroutine zgetrs_batch_strided

      subroutine cgetrfnp_batch_strided(m, n, a, lda, stride_a,        &
                                        batch_size, info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: batch_size
      complex*8, intent(inout) :: a(stride_a,*)
      integer, intent(out) :: info(*)
!$omp declare variant (mkl_lapack_cgetrfnp_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_cgetrfnp_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,info)
      end subroutine cgetrfnp_batch_strided

      subroutine dgetrfnp_batch_strided(m, n, a, lda, stride_a,        &
                                        batch_size, info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: batch_size
      double precision, intent(inout) :: a(stride_a,*)
      integer, intent(out) :: info(*)
!$omp declare variant (mkl_lapack_dgetrfnp_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dgetrfnp_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,info)
      end subroutine dgetrfnp_batch_strided

      subroutine sgetrfnp_batch_strided(m, n, a, lda, stride_a,        &
                                        batch_size, info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: batch_size
      real, intent(inout) :: a(stride_a,*)
      integer, intent(out) :: info(*)
!$omp declare variant (mkl_lapack_sgetrfnp_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_sgetrfnp_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,info)
      end subroutine sgetrfnp_batch_strided

      subroutine zgetrfnp_batch_strided(m, n, a, lda, stride_a,        &
                                        batch_size, info) bind(c)
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: batch_size
      complex*16, intent(inout) :: a(stride_a,*)
      integer, intent(out) :: info(*)
!$omp declare variant (mkl_lapack_zgetrfnp_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_zgetrfnp_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,info)
      end subroutine zgetrfnp_batch_strided

      subroutine cgetrsnp_batch_strided(trans, n, nrhs, a, lda,        &
                                        stride_a, b, ldb, stride_b,    &
                                        batch_size, info) bind(c)
      character*1, intent(in) :: trans
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: ldb
      integer, intent(in) :: stride_b
      integer, intent(in) :: batch_size
      complex*8, intent(in) :: a(stride_a,*)
      complex*8, intent(inout) :: b(stride_b,*)
      integer, intent(out) :: info(*)
!$omp declare variant (mkl_lapack_cgetrsnp_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_cgetrsnp_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,b,info)
      end subroutine cgetrsnp_batch_strided

      subroutine dgetrsnp_batch_strided(trans, n, nrhs, a, lda,        &
                                        stride_a, b, ldb, stride_b,    &
                                        batch_size, info) bind(c)
      character*1, intent(in) :: trans
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: ldb
      integer, intent(in) :: stride_b
      integer, intent(in) :: batch_size
      double precision, intent(in) :: a(stride_a,*)
      double precision, intent(inout) :: b(stride_b,*)
      integer, intent(out) :: info(*)
!$omp declare variant (mkl_lapack_dgetrsnp_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_dgetrsnp_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,b,info)
      end subroutine dgetrsnp_batch_strided

      subroutine sgetrsnp_batch_strided(trans, n, nrhs, a, lda,        &
                                        stride_a, b, ldb, stride_b,    &
                                        batch_size, info) bind(c)
      character*1, intent(in) :: trans
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: ldb
      integer, intent(in) :: stride_b
      integer, intent(in) :: batch_size
      real, intent(in) :: a(stride_a,*)
      real, intent(inout) :: b(stride_b,*)
      integer, intent(out) :: info(*)
!$omp declare variant (mkl_lapack_sgetrsnp_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_sgetrsnp_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,b,info)
      end subroutine sgetrsnp_batch_strided

      subroutine zgetrsnp_batch_strided(trans, n, nrhs, a, lda,        &
                                        stride_a, b, ldb, stride_b,    &
                                        batch_size, info) bind(c)
      character*1, intent(in) :: trans
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: ldb
      integer, intent(in) :: stride_b
      integer, intent(in) :: batch_size
      complex*16, intent(in) :: a(stride_a,*)
      complex*16, intent(inout) :: b(stride_b,*)
      integer, intent(out) :: info(*)
!$omp declare variant (mkl_lapack_zgetrsnp_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (mkl_lapack_zgetrsnp_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,b,info)
      end subroutine zgetrsnp_batch_strided

      subroutine dpotrf(uplo, n, a, lda, info) bind(c)
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      double precision, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      integer, intent(out) :: info
!$omp declare variant (dpotrf:mkl_lapack_dpotrf_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (dpotrf:mkl_lapack_dpotrf_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,info)
      end subroutine dpotrf

      subroutine spotrf(uplo, n, a, lda, info) bind(c)
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      real, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      integer, intent(out) :: info
!$omp declare variant (spotrf:mkl_lapack_spotrf_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (spotrf:mkl_lapack_spotrf_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,info)
      end subroutine spotrf

      subroutine cpotrf(uplo, n, a, lda, info) bind(c)
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      complex*8, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      integer, intent(out) :: info
!$omp declare variant (cpotrf:mkl_lapack_cpotrf_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (cpotrf:mkl_lapack_cpotrf_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,info)
      end subroutine cpotrf

      subroutine zpotrf(uplo, n, a, lda, info) bind(c)
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      complex*16, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      integer, intent(out) :: info
!$omp declare variant (zpotrf:mkl_lapack_zpotrf_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (zpotrf:mkl_lapack_zpotrf_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,info)
      end subroutine zpotrf


      subroutine dpotri(uplo, n, a, lda, info) bind(c)
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      double precision, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      integer, intent(out) :: info
!$omp declare variant (dpotri:mkl_lapack_dpotri_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (dpotri:mkl_lapack_dpotri_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,info)
      end subroutine dpotri

      subroutine spotri(uplo, n, a, lda, info) bind(c)
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      real, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      integer, intent(out) :: info
!$omp declare variant (spotri:mkl_lapack_spotri_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (spotri:mkl_lapack_spotri_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,info)
      end subroutine spotri

      subroutine cpotri(uplo, n, a, lda, info) bind(c)
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      complex*8, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      integer, intent(out) :: info
!$omp declare variant (cpotri:mkl_lapack_cpotri_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (cpotri:mkl_lapack_cpotri_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,info)
      end subroutine cpotri

      subroutine zpotri(uplo, n, a, lda, info) bind(c)
      character*1, intent(in) :: uplo
      integer, intent(in) :: n
      complex*16, intent(inout) :: a(lda,*)
      integer, intent(in) :: lda
      integer, intent(out) :: info
!$omp declare variant (zpotri:mkl_lapack_zpotri_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (zpotri:mkl_lapack_zpotri_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,info)
      end subroutine zpotri


      subroutine dpotrs(uplo, n, nrhs, a, lda, b, ldb, info) bind(c)
      character*1, intent(in) :: uplo
      integer, intent(in) :: n, nrhs, lda, ldb
      double precision, intent(in) :: a(lda,*)
      double precision, intent(inout) :: b(ldb,*)
      integer, intent(out) :: info
!$omp declare variant (dpotrs:mkl_lapack_dpotrs_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (dpotrs:mkl_lapack_dpotrs_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,b,info)
      end subroutine dpotrs

      subroutine spotrs(uplo, n, nrhs, a, lda, b, ldb, info) bind(c)
      character*1, intent(in) :: uplo
      integer, intent(in) :: n, nrhs, lda, ldb
      real, intent(in) :: a(lda,*)
      real, intent(inout) :: b(ldb,*)
      integer, intent(out) :: info
!$omp declare variant (spotrs:mkl_lapack_spotrs_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (spotrs:mkl_lapack_spotrs_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,b,info)
      end subroutine spotrs

      subroutine cpotrs(uplo, n, nrhs, a, lda, b, ldb, info) bind(c)
      character*1, intent(in) :: uplo
      integer, intent(in) :: n, nrhs, lda, ldb
      complex*8, intent(in) :: a(lda,*)
      complex*8, intent(inout) :: b(ldb,*)
      integer, intent(out) :: info
!$omp declare variant (cpotrs:mkl_lapack_cpotrs_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (cpotrs:mkl_lapack_cpotrs_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,b,info)
      end subroutine cpotrs

      subroutine zpotrs(uplo, n, nrhs, a, lda, b, ldb, info) bind(c)
      character*1, intent(in) :: uplo
      integer, intent(in) :: n, nrhs, lda, ldb
      complex*16, intent(in) :: a(lda,*)
      complex*16, intent(inout) :: b(ldb,*)
      integer, intent(out) :: info
!$omp declare variant (zpotrs:mkl_lapack_zpotrs_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (zpotrs:mkl_lapack_zpotrs_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,b,info)
      end subroutine zpotrs

      subroutine mkl_lapack_sgels_batch_strided_omp_offload_lp64() bind(c) 
      end subroutine mkl_lapack_sgels_batch_strided_omp_offload_lp64

      subroutine sgels_batch_strided(trans, m, n, nrhs, a, lda, stride_a, b, ldb, stride_b, batch_size, info) bind(c)
      character*1, intent(in) :: trans
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: ldb
      integer, intent(in) :: stride_b
      integer, intent(in) :: batch_size
      real,    intent(inout) :: a(stride_a,*)
      real,    intent(inout) :: b(stride_b,*)
      integer, intent(out) :: info(*)
!$omp declare variant (sgels_batch_strided:mkl_lapack_sgels_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (sgels_batch_strided:mkl_lapack_sgels_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,b,info)
      end subroutine sgels_batch_strided

      subroutine mkl_lapack_dgels_batch_strided_omp_offload_lp64() bind(c)
      end subroutine mkl_lapack_dgels_batch_strided_omp_offload_lp64

      subroutine dgels_batch_strided(trans, m, n, nrhs, a, lda, stride_a, b, ldb, stride_b, batch_size, info) bind(c)
      character*1, intent(in) :: trans
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: ldb
      integer, intent(in) :: stride_b
      integer, intent(in) :: batch_size
      double precision, intent(inout) :: a(stride_a,*)
      double precision, intent(inout) :: b(stride_b,*)
      integer, intent(out) :: info(*)
!$omp declare variant (dgels_batch_strided:mkl_lapack_dgels_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (dgels_batch_strided:mkl_lapack_dgels_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,b,info)
      end subroutine dgels_batch_strided

      subroutine mkl_lapack_cgels_batch_strided_omp_offload_lp64() bind(c) 
      end subroutine mkl_lapack_cgels_batch_strided_omp_offload_lp64

      subroutine cgels_batch_strided(trans, m, n, nrhs, a, lda, stride_a, b, ldb, stride_b, batch_size, info) bind(c)
      character*1, intent(in) :: trans
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: ldb
      integer, intent(in) :: stride_b
      integer, intent(in) :: batch_size
      complex*8, intent(inout) :: a(stride_a,*)
      complex*8, intent(inout) :: b(stride_b,*)
      integer, intent(out) :: info(*)
!$omp declare variant (cgels_batch_strided:mkl_lapack_cgels_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (cgels_batch_strided:mkl_lapack_cgels_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,b,info)
      end subroutine cgels_batch_strided

      subroutine mkl_lapack_zgels_batch_strided_omp_offload_lp64() bind(c)
      end subroutine mkl_lapack_zgels_batch_strided_omp_offload_lp64

      subroutine zgels_batch_strided(trans, m, n, nrhs, a, lda, stride_a, b, ldb, stride_b, batch_size, info) bind(c)
      character*1, intent(in) :: trans
      integer, intent(in) :: m
      integer, intent(in) :: n
      integer, intent(in) :: nrhs
      integer, intent(in) :: lda
      integer, intent(in) :: stride_a
      integer, intent(in) :: ldb
      integer, intent(in) :: stride_b
      integer, intent(in) :: batch_size
      complex*16, intent(inout) :: a(stride_a,*)
      complex*16, intent(inout) :: b(stride_b,*)
      integer, intent(out) :: info(*)
!$omp declare variant (zgels_batch_strided:mkl_lapack_zgels_batch_strided_omp_offload_lp64) match(construct={target variant dispatch}, device={arch(gen)})
!$omp declare variant (zgels_batch_strided:mkl_lapack_zgels_batch_strided_omp_offload_lp64) match(construct={dispatch}, device={arch(gen)}) append_args(interop(targetsync)) adjust_args(need_device_ptr:a,b,info)
      end subroutine zgels_batch_strided

   end interface

end module onemkl_lapack_omp_offload_lp64
