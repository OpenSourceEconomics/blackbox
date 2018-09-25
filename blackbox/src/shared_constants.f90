MODULE shared_constants

    !/*	setup	                */

    IMPLICIT NONE

!******************************************************************************
!******************************************************************************

    INTEGER, PARAMETER          :: our_int      = SELECTED_INT_KIND(9)
    INTEGER, PARAMETER          :: our_dble     = SELECTED_REAL_KIND(15, 307)

    INTEGER(our_int), PARAMETER :: zero_int     = 0_our_int
    INTEGER(our_int), PARAMETER :: one_int      = 1_our_int
    INTEGER(our_int), PARAMETER :: two_int      = 2_our_int
    INTEGER(our_int), PARAMETER :: three_int    = 3_our_int
    INTEGER(our_int), PARAMETER :: four_int     = 4_our_int
    INTEGER(our_int), PARAMETER :: ten_int      = 10_our_int

    REAL(our_dble), PARAMETER   :: zero_dble        = 0.00_our_dble
    REAL(our_dble), PARAMETER   :: half_dble        = 0.50_our_dble
    REAL(our_dble), PARAMETER   :: one_dble         = 1.00_our_dble
    REAL(our_dble), PARAMETER   :: two_dble         = 2.00_our_dble
    REAL(our_dble), PARAMETER   :: three_dble       = 3.00_our_dble
    REAL(our_dble), PARAMETER   :: ten_dble         = 10.00_our_dble
    REAL(our_dble), PARAMETER   :: one_hundred_dble = 100.00_our_dble

    REAL(our_dble), PARAMETER   :: pi           = 3.141592653589793238462643383279502884197_our_dble
    REAL(our_dble), PARAMETER   :: eps          = EPSILON(one_dble)

    ! Variables that need to be available throughout the library for ease of use.
    INTEGER(our_int)            :: num_points
    INTEGER(our_int)            :: num_params

    ! Variables that need to be aligned across FORTRAN and PYTHON implementations.
    INTEGER(our_int), PARAMETER :: MISSING_INT                  = -99_our_int
!******************************************************************************
!******************************************************************************
END MODULE
