!*******************************************************************************
!*******************************************************************************
SUBROUTINE f2py_spread(rslt, points, n, d)

    !/* setup                   */

    USE blackbox

    IMPLICIT NONE

    DOUBLE PRECISION, INTENT(OUT)   :: rslt

    INTEGER, INTENT(IN)             :: d
    INTEGER, INTENT(IN)             :: n

    DOUBLE PRECISION, INTENT(IN)    :: points(n, d)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    CALL spread(rslt, points, n, d)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE f2py_get_capital_phi(rslt, points, T, n, d)

    !/* setup                   */

    USE blackbox

    IMPLICIT NONE

    DOUBLE PRECISION, INTENT(OUT)   :: rslt(n, n)

    DOUBLE PRECISION, INTENT(IN)    :: points(n, d)
    DOUBLE PRECISION, INTENT(IN)    :: T(d, d)

    INTEGER, INTENT(IN)             :: d
    INTEGER, INTENT(IN)             :: n

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    CALL get_capital_phi(rslt, points, T, n, d)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE f2py_fit_full(rslt, lam, b, a, T, points, x)

    !/* setup                   */
    USE blackbox

    IMPLICIT NONE

    DOUBLE PRECISION, INTENT(OUT)   :: rslt

    DOUBLE PRECISION, INTENT(IN)    :: points(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: x(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: T(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: lam(:)
    DOUBLE PRECISION, INTENT(IN)    :: b(:)
    DOUBLE PRECISION, INTENT(IN)    :: a(:)

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    rslt = fit_full(lam, b, a, T, points, x)

END SUBROUTINE
!***************************************************************************************************
!***************************************************************************************************
SUBROUTINE f2py_constraint_full(rslt, point, r, x)

    !/* setup                   */

    USE blackbox

    IMPLICIT NONE

    DOUBLE PRECISION, INTENT(OUT)   :: rslt

    DOUBLE PRECISION, INTENT(IN)    :: point(:)
    DOUBLE PRECISION, INTENT(IN)    :: x(:)
    DOUBLE PRECISION, INTENT(IN)    :: r

!---------------------------------------------------------------------------------------------------
! Algorithm
!---------------------------------------------------------------------------------------------------

    CALL constraint(rslt, point, r, x)

END SUBROUTINE
!***************************************************************************************************
!***************************************************************************************************
SUBROUTINE f2py_minimize_slsqp(x, r, points)

    !/* setup                   */

    USE blackbox

    IMPLICIT NONE

    DOUBLE PRECISION, INTENT(IN)    :: x(:)
    DOUBLE PRECISION, INTENT(IN)    :: r
    DOUBLE PRECISION, INTENT(IN)    :: points(:, :)

!---------------------------------------------------------------------------------------------------
! Algorithm
!---------------------------------------------------------------------------------------------------

    CALL minimize_slsqp(x, r, points)

END SUBROUTINE
!***************************************************************************************************
!***************************************************************************************************
