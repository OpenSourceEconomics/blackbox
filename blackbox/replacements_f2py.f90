
!*******************************************************************************
!*******************************************************************************
SUBROUTINE f2py_spread(rslt, points, n, d)

    !/* setup                   */

    IMPLICIT NONE

    DOUBLE PRECISION, INTENT(OUT)   :: rslt


    INTEGER, INTENT(IN)             :: n, d
    DOUBLE PRECISION, INTENT(IN)    :: points(n, d)


    INTEGER                         ::i, j

    !TODO: Review best practices
!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    rslt = 0.0
    DO i = 1, n
        DO j = 1, n
            IF (i .GT. j) THEN
                rslt = rslt + 1.0 / NORM2(points(i, :) - points(j, :))
            END IF
        END DO
    END DO

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************

SUBROUTINE f2py_get_capital_phi(rslt, points, T, n, d)

    !/* setup                   */

    IMPLICIT NONE

    DOUBLE PRECISION, INTENT(OUT)   :: rslt(n, n)
    DOUBLE PRECISION, INTENT(IN)               :: T(d, d)
    INTEGER, INTENT(IN)             :: d

    INTEGER, INTENT(IN)             :: n
    DOUBLE PRECISION, INTENT(IN)    :: points(n, d)


    INTEGER                         ::i, j

    DOUBLE PRECISION :: substract(d)
    !TODO: Review best practices
!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

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

!*******************************************************************************
!*******************************************************************************
SUBROUTINE f2py_fit_full(rslt, lam, b, a, T, points, x)

    !/* setup                   */

    IMPLICIT NONE

    DOUBLE PRECISION, INTENT(OUT):: rslt

    DOUBLE PRECISION, INTENT(IN)          :: lam(:), b(:), a(:), T(:, :), points(:, :), x(:, :)


    DOUBLE PRECISION :: incr(1)

    DOUBLE PRECISION, ALLOCATABLE   :: substr(:)
    INTEGER :: n, i, d
!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------
    n = SIZE(lam)
    incr = MATMUL(b, x) + a
    d = SIZE(T, 1)
    ALLOCATE(substr(d))
rslt = 0.0
    DO i = 1, n
        substr = x(:, 1) - points(i, :)
        rslt = rslt + lam(i) * NORM2(MATMUL(T, substr)) ** 3
    END DO
    rslt = rslt + incr(1)

END SUBROUTINE
