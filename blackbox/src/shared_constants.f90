MODULE shared_constants

    !/*	setup	                */

    IMPLICIT NONE

!******************************************************************************
!******************************************************************************

    INTEGER, PARAMETER          :: our_int      = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER          :: our_dble     = SELECTED_REAL_KIND(15, 307)

    INTEGER(our_int), PARAMETER :: zero_int     = 0_our_int
    INTEGER(our_int), PARAMETER :: one_int      = 1_our_int

    REAL(our_dble), PARAMETER   :: zero_dble        = 0.00_our_dble
    REAL(our_dble), PARAMETER   :: one_dble         = 1.00_our_dble

    ! Variables that need to be available throughout the library for ease of use.
    INTEGER(our_int)            :: num_points
    INTEGER(our_int)            :: num_params

    REAL(our_dble)              :: eps_der_approx = 1e-6


    ! Variables that need to be aligned across FORTRAN and PYTHON implementations.
    INTEGER(our_int), PARAMETER :: MISSING_INT                  = -99_our_int
!******************************************************************************
!******************************************************************************
END MODULE
