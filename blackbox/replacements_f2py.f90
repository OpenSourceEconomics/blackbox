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

    ! Distribute global variable
    num_points = n
    num_params = d

    CALL spread(rslt, points)

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

    ! Distribute global variable
    num_points = n
    num_params = d

    CALL get_capital_phi(rslt, points, T)

END SUBROUTINE
!*******************************************************************************
!*******************************************************************************
SUBROUTINE f2py_fit_full(rslt, lam, b, a, T, points, x)

    !/* setup                   */
    USE blackbox

    IMPLICIT NONE

    DOUBLE PRECISION, INTENT(OUT)   :: rslt

    DOUBLE PRECISION, INTENT(IN)    :: points(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: x(:)
    DOUBLE PRECISION, INTENT(IN)    :: T(:, :)
    DOUBLE PRECISION, INTENT(IN)    :: lam(:)
    DOUBLE PRECISION, INTENT(IN)    :: b(:)
    DOUBLE PRECISION, INTENT(IN)    :: a

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Distribute global variables
    num_params = SIZE(points, 2)
    num_points = SIZE(points, 1)

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

    ! Distribute global variables
    num_points = MISSING_INT
    num_params = SIZE(x)

    CALL constraint(rslt, point, r, x)

END SUBROUTINE
!***************************************************************************************************
!***************************************************************************************************
SUBROUTINE f2py_minimize_slsqp(x, x_start, r, points,  lam, b, a, T, d, n)

    !/* setup                   */

    USE blackbox

    IMPLICIT NONE

    DOUBLE PRECISION, INTENT(OUT)   :: x(d)

    DOUBLE PRECISION, INTENT(IN)    :: x_start(d)
    DOUBLE PRECISION, INTENT(IN)    :: points(n, d)
    DOUBLE PRECISION, INTENT(IN)    :: T(d, d)
    DOUBLE PRECISION, INTENT(IN)    :: lam(n)
    DOUBLE PRECISION, INTENT(IN)    :: b(d)
    DOUBLE PRECISION, INTENT(IN)    :: a
    DOUBLE PRECISION, INTENT(IN)    :: r

    INTEGER, INTENT(IN)             :: n
    INTEGER, INTENT(IN)             :: d

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Distribute global variables
    num_points = SIZE(points, 1)
    num_params = SIZE(points, 2)

    x = x_start

    CALL minimize_slsqp(x, r, points, lam, b, a, T)

END SUBROUTINE
!***************************************************************************************************
!***************************************************************************************************
SUBROUTINE f2py_derivative_function(rslt, x, lam, b, a_ext, T, points, n, d)

    !/* setup                   */

    USE blackbox

    IMPLICIT NONE

    DOUBLE PRECISION, INTENT(OUT)   :: rslt(d)
    DOUBLE PRECISION, INTENT(IN)    :: T(d, d)
    DOUBLE PRECISION, INTENT(IN)    :: lam(n)
    DOUBLE PRECISION, INTENT(IN)    :: b(d)
    DOUBLE PRECISION, INTENT(IN)    :: a_ext

    DOUBLE PRECISION, INTENT(IN)   :: x(d)
    DOUBLE PRECISION, INTENT(IN)   :: points(n, d)

    INTEGER, INTENT(IN)             :: d
    INTEGER, INTENT(IN)             :: n

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Distribute global variables
    num_points = n
    num_params = d

    rslt = derivative_function(x, lam, b, a_ext, T, points)

END SUBROUTINE
!***************************************************************************************************
!***************************************************************************************************
SUBROUTINE f2py_derivative_constraints(rslt, points, r, x, n, d)

    !/* setup                   */

    USE blackbox

    IMPLICIT NONE


    DOUBLE PRECISION, INTENT(OUT)   :: rslt(n, d)
    DOUBLE PRECISION, INTENT(IN)    :: points(n, d)
    DOUBLE PRECISION, INTENT(IN)    :: x(d), r

    INTEGER, INTENT(IN)             :: d
    INTEGER, INTENT(IN)             :: n

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    ! Distribute global variables
    num_points = n
    num_params = d

    rslt = derivative_constraints(points, r, x, n)

END SUBROUTINE
!***************************************************************************************************
!***************************************************************************************************
