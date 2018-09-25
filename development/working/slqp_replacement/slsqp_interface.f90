!***************************************************************************************************
!***************************************************************************************************
MODULE slsqp_interface

    INTERFACE

        PURE SUBROUTINE SLSQP(M, MEQ, LA, N, X, XL, XU, F, C, G, A, ACC, ITER, MODE, W, LEN_W, JW, LEN_JW)

              !/* external modules    */

              USE shared_constants

              !/* external objects    */

              INTEGER(our_int), INTENT(INOUT)     :: ITER
              INTEGER(our_int), INTENT(INOUT)     :: MODE

              REAL(our_dble), INTENT(INOUT)       :: X(N)
              REAL(our_dble), INTENT(INOUT)       :: ACC

              INTEGER(our_int), INTENT(IN)        :: JW(LEN_W)
              INTEGER(our_int), INTENT(IN)        :: LEN_JW
              INTEGER(our_int), INTENT(IN)        :: LEN_W
              INTEGER(our_int), INTENT(IN)        :: MEQ
              INTEGER(our_int), INTENT(IN)        :: LA
              INTEGER(our_int), INTENT(IN)        :: M
              INTEGER(our_int), INTENT(IN)        :: N

              REAL(our_dble), INTENT(IN)          :: A(LA, N + 1)
              REAL(our_dble), INTENT(IN)          :: G(N + 1)
              REAL(our_dble), INTENT(IN)          :: W(LEN_W)
              REAL(our_dble), INTENT(IN)          :: XL(N)
              REAL(our_dble), INTENT(IN)          :: XU(N)
              REAL(our_dble), INTENT(IN)          :: C(LA)
              REAL(our_dble), INTENT(IN)          :: F

          END SUBROUTINE

    END INTERFACE

!***************************************************************************************************
!***************************************************************************************************
END MODULE
