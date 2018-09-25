MODULE replacements

    USE shared_constants

    USE slsqp_interface

    IMPLICIT NONE

CONTAINS

!***************************************************************************************************
!***************************************************************************************************
SUBROUTINE spread(rslt, points, n, d)

    !/* setup                   */

    REAL(our_dble), INTENT(OUT)   :: rslt

    INTEGER(our_int), INTENT(IN)             :: d
    INTEGER(our_int), INTENT(IN)             :: n

    REAL(our_dble), INTENT(IN)    :: points(n, d)

    INTEGER(our_int)                         :: j
    INTEGER(our_int)                         :: i

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    rslt = 0.0
    DO i = 1, n
        DO j = 1, i - 1
            rslt = rslt + 1.0 / NORM2(points(i, :) - points(j, :))
        END DO
    END DO

END SUBROUTINE
!***************************************************************************************************
!***************************************************************************************************
SUBROUTINE get_capital_phi(rslt, points, T, n, d)

    !/* setup                   */

    REAL(our_dble), INTENT(OUT)   :: rslt(n, n)

    REAL(our_dble), INTENT(IN)    :: points(n, d)
    REAL(our_dble), INTENT(IN)    :: T(d, d)

    INTEGER(our_int), INTENT(IN)             :: d
    INTEGER(our_int), INTENT(IN)             :: n

    REAL(our_dble)                :: substract(d)

    INTEGER(our_int)                         :: i
    INTEGER(our_int)                         :: j

!---------------------------------------------------------------------------------------------------
! Algorithm
!---------------------------------------------------------------------------------------------------

    rslt = -99.0
    DO i = 1, n
        DO j = 1, n
            substract(:) = points(i, :) - points(j, :)
            rslt(i, j) = NORM2(MATMUL(T, substract)) ** 3
        END DO
    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
FUNCTION fit_full(lam, b, a, T, points, x) RESULT(rslt)

    !/* setup                   */

    REAL(our_dble)   :: rslt

    REAL(our_dble), INTENT(IN)    :: points(:, :)
    REAL(our_dble), INTENT(IN)    :: x(:, :)
    REAL(our_dble), INTENT(IN)    :: T(:, :)
    REAL(our_dble), INTENT(IN)    :: lam(:)
    REAL(our_dble), INTENT(IN)    :: b(:)
    REAL(our_dble), INTENT(IN)    :: a(:)

    REAL(our_dble)                :: substr(SIZE(T, 1))
    REAL(our_dble)                :: incr(1)

    INTEGER(our_int)                         :: i

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    incr = MATMUL(b, x) + a

    rslt = 0.0
    DO i = 1, SIZE(lam)
        substr = x(:, 1) - points(i, :)
        rslt = rslt + lam(i) * NORM2(MATMUL(T, substr)) ** 3
    END DO
    rslt = rslt + incr(1)

END FUNCTION
!***************************************************************************************************
!***************************************************************************************************
SUBROUTINE constraint(rslt, point, r, x)

    !/* setup                   */

    REAL(our_dble), INTENT(OUT)   :: rslt

    REAL(our_dble), INTENT(IN)    :: point(:)
    REAL(our_dble), INTENT(IN)    :: x(:)
    REAL(our_dble), INTENT(IN)    :: r

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    rslt = NORM2(x - point) - r

END SUBROUTINE
!***************************************************************************************************
!***************************************************************************************************
FUNCTION get_all_constraints(points, r, x, num_constraints) result(constraints)

    !/* setup                   */

    REAL(our_dble)  :: constraints(num_constraints)

    REAL(our_dble), INTENT(IN)      :: points(:, :)
    REAL(our_dble), INTENT(IN)      :: x(:)
    REAL(our_dble), INTENT(IN)      :: r

    INTEGER(our_int), INTENT(IN)    :: num_constraints

    INTEGER(our_int)                :: i

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    DO i = 1, num_constraints
        CALL constraint(constraints(i), points(i, :), r, x)
    END DO

END FUNCTION
!***************************************************************************************************
!***************************************************************************************************
SUBROUTINE minimize_slsqp(x, r, points, lam, b, a_ext, T)

    !/* setup                   */

 !   DOUBLE PRECISION, INTENT(OUT)   :: rslt

 !   INTEGER, INTENT(IN)             :: d

    DOUBLE PRECISION, INTENT(IN)    :: x(:)
    DOUBLE PRECISION, INTENT(IN)    :: points(:, :), r

    REAL(our_dble), INTENT(IN)    :: T(:, :)
    REAL(our_dble), INTENT(IN)    :: lam(:)
    REAL(our_dble), INTENT(IN)    :: b(:)
    REAL(our_dble), INTENT(IN)    :: a_ext(:)


 !   INTEGER                         :: j
!    INTEGER                         :: i

    ! TODO: These are wrkspace dimensions.
    INTEGER     :: LEN_W, LEN_JW, LA

     INTEGER  :: ITER
     INTEGER  :: MODE, num_constraints, n

              DOUBLE PRECISION    :: ACC

              INTEGER, ALLOCATABLE      :: JW(:)
             DOUBLE PRECISION, ALLOCATABLE        :: W(:)

              INTEGER       :: MEQ
              INTEGER       :: M, N1, MINEQ

               DOUBLE PRECISION, ALLOCATABLE        :: A(:, :)
              DOUBLE PRECISION, ALLOCATABLE       :: G(:)
              DOUBLE PRECISION, ALLOCATABLE        :: XL(:),  XU(:), x_iter(:)
              DOUBLE PRECISION, ALLOCATABLE         :: C(:)
              DOUBLE PRECISION        :: F


    LOGICAL :: is_finished

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    PRINT *, 'test'

    ! Some basic setup
    n = SIZE(x)
    num_constraints = SIZE(points, 1)


    ! TODO: move outside

    ITER = 10
    ACC = 10e-6
    mode = zero_int
    M = num_constraints

    ! In this problems there are no equality constraints ... but many many inequality

    ALLOCATE(XL(n), XU(n), x_iter(n))

    MEQ = zero_int
    N1= N+1
    LA = MAX(1, M)
    MINEQ= M-MEQ+N1+N1


    LEN_W = (3 * N1 + M)*(N1 + 1) + (N1 - MEQ + 1) * (MINEQ + 2) + 2 * MINEQ
    LEN_W = LEN_W + (N1 + MINEQ) * (N1 - MEQ) + 2 * MEQ + N1  + (N + 1) * N / 2 + 2 * M + 3 * N
    LEN_W = LEN_W + 3 * N1 + 1

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

    ! We evaluate the criterion function at the starting values.
    f = fit_full(lam, b, a_ext, T, points, x)

    ! We evaluate the constraint at the starting valures.
    c = get_all_constraints(points, r, x, num_constraints)

    CALL SLSQP(m, meq, la, n, x_iter, xl, xu, f, c, g, a, acc, iter, mode, W, LEN_W, JW, LEN_JW)

END SUBROUTINE

END MODULE
