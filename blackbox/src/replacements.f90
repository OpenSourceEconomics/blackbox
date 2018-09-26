MODULE replacements

    USE shared_constants

    USE slsqp_interface

    IMPLICIT NONE

CONTAINS
!***************************************************************************************************
!***************************************************************************************************
SUBROUTINE spread(rslt, points)

    !/* setup                   */

    REAL(our_dble), INTENT(OUT)   :: rslt

    REAL(our_dble), INTENT(IN)    :: points(num_points, num_params)

    INTEGER(our_int)                         :: j
    INTEGER(our_int)                         :: i

!-------------------------------------------------------------------------------
! Algorithm
!-------------------------------------------------------------------------------

    rslt = 0.0
    DO i = 1, num_points
        DO j = 1, i - 1
            rslt = rslt + 1.0 / NORM2(points(i, :) - points(j, :))
        END DO
    END DO

END SUBROUTINE
!***************************************************************************************************
!***************************************************************************************************
SUBROUTINE get_capital_phi(rslt, points, T)

    !/* setup                   */

    REAL(our_dble), INTENT(OUT)   :: rslt(num_points, num_points)

    REAL(our_dble), INTENT(IN)    :: points(num_points, num_params)
    REAL(our_dble), INTENT(IN)    :: T(num_params, num_params)

    REAL(our_dble)                :: substract(num_params)

    INTEGER(our_int)                         :: i
    INTEGER(our_int)                         :: j

!---------------------------------------------------------------------------------------------------
! Algorithm
!---------------------------------------------------------------------------------------------------

    rslt = -99.0
    DO i = 1, num_points
        DO j = 1, num_points
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

    REAL(our_dble), INTENT(IN)    :: points(num_points, num_params)
    REAL(our_dble), INTENT(IN)    :: x(num_params)
    REAL(our_dble), INTENT(IN)    :: T(num_params,  num_params)
    REAL(our_dble), INTENT(IN)    :: lam(num_points)
    REAL(our_dble), INTENT(IN)    :: b(num_params)
    REAL(our_dble), INTENT(IN)    :: a

    REAL(our_dble)                :: substr(SIZE(T, 1))
    REAL(our_dble)                :: incr(1), x_extended(SIZE(x), 1)

    INTEGER(our_int)                         :: i

    !-----------------------------------------------------------------------------------------------
    ! Algorithm
    !-----------------------------------------------------------------------------------------------
    x_extended(:, 1) = x
    incr = MATMUL(b, x_extended) + a

    rslt = 0.0
    DO i = 1, SIZE(lam)
        substr = x(:) - points(i, :)
        rslt = rslt + lam(i) * NORM2(MATMUL(T, substr)) ** 3
    END DO
    rslt = rslt + incr(1)

END FUNCTION
!***************************************************************************************************
!***************************************************************************************************
SUBROUTINE constraint(rslt, point, r, x)

    !/* setup                   */

    REAL(our_dble), INTENT(OUT)   :: rslt

    REAL(our_dble), INTENT(IN)    :: point(num_params)
    REAL(our_dble), INTENT(IN)    :: x(num_params)
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

    REAL(our_dble), INTENT(IN)      :: points(num_points, num_params)
    REAL(our_dble), INTENT(IN)      :: x(num_params)
    REAL(our_dble), INTENT(IN)      :: r

    INTEGER(our_int), INTENT(IN)    :: num_constraints

    REAL(our_dble)                  :: point(num_params)
    INTEGER(our_int)                :: i

    !-----------------------------------------------------------------------------------------------
    ! Algorithm
    !-----------------------------------------------------------------------------------------------

    DO i = 1, num_constraints
        point = points(i, :)
        CALL constraint(constraints(i), point, r, x)
    END DO

END FUNCTION
!***************************************************************************************************
!***************************************************************************************************
SUBROUTINE minimize_slsqp(x, r, points, lam, b, a_ext, T)

    !/* setup                   */

 !   DOUBLE PRECISION, INTENT(OUT)   :: rslt

 !   INTEGER, INTENT(IN)             :: d

    DOUBLE PRECISION, INTENT(INOUT)    :: x(:)
    DOUBLE PRECISION, INTENT(IN)    :: points(:, :), r

        REAL(our_dble), INTENT(IN)    :: T(:, :)
        REAL(our_dble), INTENT(IN)    :: lam(:)
        REAL(our_dble), INTENT(IN)    :: b(:)
        REAL(our_dble), INTENT(IN)    :: a_ext

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

    ! Some basic setup
    n = SIZE(x)
    num_constraints = SIZE(points, 1)


    ! TODO: move outside
    ITER = 100
    ACC = 1E-6
    mode = zero_int
    M = num_constraints

    ! In this problems there are no equality constraints ... but many many inequality

    ALLOCATE(XL(n), XU(n), x_iter(n))

    MEQ = zero_int
    N1= N + 1
    LA = MAX(1, M)

    MINEQ= M - MEQ + N1 + N1

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
    c = get_all_constraints(points, r, x, num_constraints)
    f = fit_full(lam, b, a_ext, T, points, x)
    G(1:num_params) = derivative_function(x, lam, b, a_ext, T, points)
    A(:,:num_params) = derivative_constraints(points, r, x, num_constraints)

    ! Iterate until completion
    DO WHILE (.NOT. is_finished)
        ! Evaluate criterion function and constraints
        IF (mode .EQ. one_int) THEN
            c = get_all_constraints(points, r, x_iter, num_constraints)
            f = fit_full(lam, b, a_ext, T, points, x_iter)
        ELSEIF (mode .EQ. - one_int) THEN
            G(1:num_params) = derivative_function(x_iter, lam, b, a_ext, T, points)
            A(:,:num_params) = derivative_constraints(points, r, x_iter, num_constraints)
        END IF

        CALL SLSQP(m, meq, la, n, x_iter, xl, xu, f, c, g, a, acc, iter, mode, W, LEN_W, JW, LEN_JW)

        ! TODO: Stabilization as in a rare number of cases the SLSQP routine returns NAN.
        IF (ANY(ISNAN(X))) THEN
            mode = 17
        END IF

        ! Check if SLSQP has completed
        IF (.NOT. ABS(mode) .EQ. one_int) THEN
                is_finished = .True.
            END IF

    END DO

    ! We only replace the starting values with the final values if the optimization was successful.
    IF(mode .EQ. zero_int) x = x_iter

END SUBROUTINE
!***************************************************************************************************
!***************************************************************************************************
! TODO:  more flexible function setup, that I only pass in a different function each time.
FUNCTION derivative_function(x, lam, b, a, T, points) RESULT(rslt)

    REAL(our_dble), INTENT(IN)      :: points(num_points, num_params)
    REAL(our_dble), INTENT(IN)      :: T(num_params, num_params)
    REAL(our_dble), INTENT(IN)      :: lam(num_points)
    REAL(our_dble), INTENT(IN)      :: x(num_params)
    REAL(our_dble), INTENT(IN)      :: b(num_params)
    REAL(our_dble), INTENT(IN)      :: a

    REAL(our_dble)                  :: rslt(num_params)
    REAL(our_dble)                  :: ei(num_params)
    REAL(our_dble)                  :: d(num_params)
    REAL(our_dble)                  :: f0
    REAL(our_dble)                  :: f1

    INTEGER(our_int)                :: j

    !-----------------------------------------------------------------------------------------------
    ! Algorithm
    !-----------------------------------------------------------------------------------------------
    ei = zero_dble
    f0 = fit_full(lam, b, a, T, points, x)

    DO j = 1, num_params

        ei(j) = one_dble

        d = eps_der_approx * ei

        f1 = fit_full(lam, b, a, T, points, x + d)

        rslt(j) = (f1 - f0) / d(j)

        ei(j) = zero_dble

    END DO

END FUNCTION
!***************************************************************************************************
!***************************************************************************************************
! TODO: This can be tested f2py
FUNCTION derivative_constraints(points, r, x, num_constraints) RESULT(rslt)

    REAL(our_dble), INTENT(IN)      :: points(num_points, num_params)
    REAL(our_dble), INTENT(IN)      :: x(num_params)
    REAL(our_dble), INTENT(IN)      :: r

    INTEGER(our_int), INTENT(IN)    :: num_constraints


        REAL(our_dble) :: rslt(num_constraints, num_params), ei(num_params), f0(num_constraints), d(num_params),   f1(num_constraints)

        INTEGER(our_int)    :: j


        ! Initialize containers
        ei = zero_dble

        ! Evaluate baseline


        f0 = get_all_constraints(points, r, x, num_constraints)

        DO j = 1, num_params

            ei(j) = one_dble

            d = eps_der_approx * ei

            f1 = get_all_constraints(points, r, x + d, num_constraints)

            rslt(:, j) = (f1 - f0) / d(j)

            ei(j) = zero_dble

        END DO

END FUNCTION
!***************************************************************************************************
!***************************************************************************************************
END MODULE
