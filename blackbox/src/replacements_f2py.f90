!*******************************************************************************
!*******************************************************************************
SUBROUTINE minimize_slsqp(x, n)

    !/* setup                   */

    USE slsqp_interface
    USE shared_constants
    IMPLICIT NONE

 !   DOUBLE PRECISION, INTENT(OUT)   :: rslt

 !   INTEGER, INTENT(IN)             :: d
    INTEGER, INTENT(IN)             :: n

    DOUBLE PRECISION, INTENT(IN)    :: x(n)

 !   INTEGER                         :: j
!    INTEGER                         :: i

    ! TODO: These are wrkspace dimensions.
    INTEGER     :: LEN_W, LEN_JW, LA

     INTEGER  :: ITER
     INTEGER  :: MODE

              DOUBLE PRECISION    :: ACC

              INTEGER, ALLOCATABLE      :: JW(:)
             DOUBLE PRECISION, ALLOCATABLE        :: W(:)

              INTEGER       :: MEQ
              INTEGER       :: M, N1, MINEQ

               DOUBLE PRECISION, ALLOCATABLE        :: A(:, :)
              DOUBLE PRECISION, ALLOCATABLE       :: G(:)
              DOUBLE PRECISION        :: XL(n),  XU(n), x_iter(n)
              DOUBLE PRECISION, ALLOCATABLE         :: C(:)
              DOUBLE PRECISION        :: F


    LOGICAL :: is_finished

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! TODO: move outside

    ITER = 10
    ACC = 10e-6
    mode = zero_int
    M = 10000

    ! In this problems there are no equality constraints ... but many many inequality

    MEQ = zero_int
    N1= N+1
    LA = MAX(1, M)
    MINEQ= M-MEQ+N1+N1


    LEN_W = (3*N1+M)*(N1+1) +(N1-MEQ+1)*(MINEQ+2) + 2*MINEQ  +(N1+MINEQ)*(N1-MEQ) + 2*MEQ + N1  +(N+1)*N/2 + 2*M + 3*N + 3*N1 + 1
    LEN_JW = MINEQ

    XL = zero_dble
    XU = one_dble

    is_finished = .False.


    ! These are workspace that are derived
    ALLOCATE(JW(LEN_W))
    ALLOCATE(W(LEN_W))


    ALLOCATE(C(M))          ! hold the values of all constraints
    ALLOCATE(G(N + 1))      ! holds the partials of the constraints, but needs N + 1 from doc
    ALLOCATE(A(LA, N + 1))

    x_iter = x

    CALL SLSQP(m, meq, la, n, x_iter, xl, xu, f, c, g, a, acc, iter, mode, W, LEN_W, JW, LEN_JW)

END SUBROUTINE
