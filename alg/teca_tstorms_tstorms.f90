  MODULE TSTORMS_MOD
  implicit none

!=====================================================================
! --- NAMELIST
!=====================================================================

! integer, parameter :: gnx0 = 3   !for c90
!  integer, parameter :: gnx0 = 6   !for c180
 integer, parameter :: gnx0 = 12  !for c360



  real :: crit_vort  =  3.5E-5   
  real :: crit_twc   =  0.5   
  real :: crit_thick = 50.0   
  real :: crit_dist  =  4.0  
  real :: lat_bound_n =  90.0
  real :: lat_bound_s = -90.0
  logical :: do_spline = .false.
  logical :: do_thickness = .false.

  namelist / nml_tstorms / crit_vort, crit_twc, crit_thick, crit_dist, &
                           lat_bound_n, lat_bound_s, do_spline, do_thickness

!=====================================================================
  contains

!######################################################################

  SUBROUTINE SET_TSTORMS
  implicit none
        READ( *, nml_tstorms )
  end SUBROUTINE SET_TSTORMS

!######################################################################

  SUBROUTINE TSTORMS ( Gwind, Gvort, Gtbar, Gpsl, Gthick,      &
                       Grlon, Grlat, iyear, imon, iday, ihour, &
                       iucy   )

!===================================================================
! --- LOCATE TROPICAL STORMS 
!===================================================================

  use TSGPAD_MOD, only : GPAD2,  GPAD1
  use SPLINE_MOD, only : SPLIE2, SPLIE3, FRPRM, SHAPE
  implicit none

!-------------------------------------------------------------------
! --- INPUT ARGUMENTS 
!     Gwind  - wind speed at 850 mb
!     Gvort  - vorticity  at 850 mb
!     Gtbar  - mean temperature for warm core layer
!     Gpsl   - sea level pressure
!     Gthick - thickness of 200 to 1000 mb layer 
!     Grlon  - longitudes
!     Grlat  - latitudes
!     iyear  - year
!     imon   - month
!     iday   - day of month
!     ihour  - hour
!     iucy   - unit for output
!-------------------------------------------------------------------
! --- OUTPUT - file "cyclones" 
!-------------------------------------------------------------------
! --- record # 1 
!     num0   - day
!     imon0  - month
!     iyear  - year
!     number - number of cyclones found
! --- records # 2...number+1 
!     idex, jdex - (i,j) index of cyclone 
!     svort_max  - max vorticity                  
!     swind_max  - max wind              
!      spsl_min  - min sea level pressure                 
!     svort_lon,  svort_lat - longitude & latitude of max vorticity 
!      spsl_lon,   spsl_lat - longitude & latitude of min slp 
!      stwc_lon,   stwc_lat - longitude & latitude of warm core 
!    sthick_lon, sthick_lat - longitude & latitude of max thickness 
!-------------------------------------------------------------------

  real,    intent(in),    dimension(:,:) :: Gwind, Gvort, Gtbar
  real,    intent(in),    dimension(:,:) :: Gpsl,  Gthick
  real,    intent(in),    dimension(:)   :: Grlon, Grlat
  integer, intent(in)                    :: ihour, iday,  imon, iyear, iucy

!-------------------------------------------------------------------
! --- LOCAL
!-------------------------------------------------------------------

  integer, parameter :: nx   = gnx0
! integer, parameter :: nx   = 12
  integer, parameter :: nx2  = 2*nx
  integer, parameter :: nxp1 = nx + 1

  real,    parameter :: ftol  = 0.01   
  integer, parameter :: nsmax = 10000

  real, dimension(SIZE(Grlon)+nx2) :: rlon
  real, dimension(SIZE(Grlat)+nx2) :: rlat

  real, dimension(SIZE(Gwind,1)+nx2,SIZE(Gwind,2)+nx2) ::  &
        vort, wind, psl,    tbar,    thick,                &
                    psl_dx, tbar_dx, thick_dx,             &
                    psl_dy, tbar_dy, thick_dy 

  real    :: vort_max, wind_max, psl_min, twc_max,   thick_max
  real    :: lon_vort,           lon_psl, lon_twc,   lon_thick
  real    :: lat_vort,           lat_psl, lat_twc,   lat_thick            
  logical ::                              exist_twc, exist_thick

  integer, dimension(nsmax) :: idex, jdex

  real,    dimension(nsmax) ::                                    &
           svort_max, swind_max,  spsl_min, stwc_max, sthick_max, &
           svort_lon,             spsl_lon, stwc_lon, sthick_lon, &
           svort_lat,             spsl_lat, stwc_lat, sthick_lat
  logical, dimension(nsmax) ::                                    &
                                            stwc_is,  sthick_is

  real,    dimension(2)  :: p

  real :: xx, yy, rr, fret

  integer :: ierr_pos, ierr_mag

  integer :: i, im, ip, ix, ixp3, ixp6
  integer :: j, jm, jp, jx, jxp3, jxp6, jxp6h 
  integer :: number, iter
  integer :: imm, jmm
  real    :: avg

!===================================================================

  ix    = SIZE( Gwind, 1 )
  jx    = SIZE( Gwind, 2 )
  ixp3  = ix + nx
  jxp3  = jx + nx
  ixp6  = ix + nx2
  jxp6  = jx + nx2
  jxp6h = jxp6 / 2

  number = 0

!-------------------------------------------------------------------
! --- SETUP
!-------------------------------------------------------------------

  CALL GPAD2( Gwind,  wind    )    !  Wind speed at 850 mb 
  CALL GPAD2( Gvort,  vort    )    !  Vorticity  at 850 mb 
  CALL GPAD2( Gtbar,  tbar    )    !  Mean temp for warm core layer
  CALL GPAD2( Gpsl,   psl     )    !  Sea level pressure.
if (do_thickness) then
  CALL GPAD2( Gthick, thick   )    !  Thickness of 200 to 1000 mb layer 
end if
  CALL GPAD1( Grlon,  rlon, 0 )    !  Longitudes
  CALL GPAD1( Grlat,  rlat, 1 )    !  Latitudes
 
! --- change sign of vorticity in southern hemisphere
  vort(:,1:jxp6h) = -1.0 * vort(:,1:jxp6h) 

! --- change sign of temperature & thickness
if (do_spline) then
   tbar(:,:) = -1.0 *  tbar(:,:)
if (do_thickness) then
  thick(:,:) = -1.0 * thick(:,:)
end if
end if

!-------------------------------------------------------------------
! --- INITIALIZE SPLINES
!-------------------------------------------------------------------
if (do_spline) then
  CALL SPLIE2( rlon, rlat, psl,   psl_dy   )
  CALL SPLIE3( rlon, rlat, psl,   psl_dx   )
 
  CALL SPLIE2( rlon, rlat, tbar,  tbar_dy  )
  CALL SPLIE3( rlon, rlat, tbar,  tbar_dx  )
 
  CALL SPLIE2( rlon, rlat, thick, thick_dy )
  CALL SPLIE3( rlon, rlat, thick, thick_dx )
end if

!===================================================================
! -- LOOP OVER GRID & LOOK FOR STORMS
!===================================================================

  do j = nxp1,jxp3
         if( ( rlat(j) > lat_bound_n ) .or. &
             ( rlat(j) < lat_bound_s ) ) CYCLE
  do i = nxp1,ixp3
! zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz

    im = i - nx
    ip = i + nx
    jm = j - nx
    jp = j + nx

!-------------------------------------------------------------------
! --- STEP 1: CHECK FOR VORTICITY MAX
!-------------------------------------------------------------------

   vort_max = MAXVAL( vort(im:ip,jm:jp) )
   wind_max = MAXVAL( wind(im:ip,jm:jp) )

! xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  if( ( vort(i,j) /= vort_max  )  .or. &
      ( vort(i,j) <  crit_vort ) ) CYCLE
! xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

  lon_vort = rlon(i)
  lat_vort = rlat(j)

!-------------------------------------------------------------------
! --- STEP 2: LOCATE LOCAL SEA LEVEL PRESSURE MIN 
!-------------------------------------------------------------------

  ierr_pos  = 0

  p(1) = lon_vort
  p(2) = lat_vort

if (do_spline) then
  CALL FRPRM( rlon, rlat, psl,  psl_dy, psl_dx, &
              p,    ftol, iter, fret,   ierr_pos )

  psl_min = fret
  lon_psl = p(1)
  lat_psl = p(2)
else
  call find_minmax (.true., psl(im:ip,jm:jp), imm, jmm, avg)
  imm=i-(nx+1)+imm
  jmm=j-(nx+1)+jmm
  lon_psl = rlon(imm)
  lat_psl = rlat(jmm)
  psl_min = psl (imm, jmm)
end if
  xx      = lon_psl - lon_vort
  yy      = lat_psl - lat_vort
  rr      = xx*xx + yy*yy

  if( rr >= crit_dist ) ierr_pos = 1

! xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   if ( ierr_pos == 1 ) CYCLE
! xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

!-------------------------------------------------------------------
! --- STEP 3: CHECK FOR PRESENCE OF A WARM CORE
!-------------------------------------------------------------------

! --- location

  ierr_pos = 0
if (do_spline) then
  CALL FRPRM( rlon, rlat, tbar, tbar_dy, tbar_dx, &
              p,    ftol, iter, fret,    ierr_pos )

  if( ierr_pos == 0 ) then
    twc_max = -fret
    lon_twc = p(1)
    lat_twc = p(2)
    xx      = lon_twc - lon_psl
    yy      = lat_twc - lat_psl
    rr      = xx*xx + yy*yy
    if( rr >= crit_dist ) ierr_pos = 1
  else
    twc_max = 0.0
    lon_twc = 0.0
    lat_twc = 0.0
  endif

  exist_twc = (ierr_pos == 0)

! --- magnitude

  if ( exist_twc ) then

  ierr_mag = 0

  CALL SHAPE( rlon, rlat, tbar, tbar_dy, tbar_dx,           &
              p,    ftol, iter, fret,    ierr_mag, crit_twc )
 
  exist_twc = exist_twc .and. ( ierr_mag == 0 )
 
  endif
else

  call find_minmax (.false., tbar(im:ip,jm:jp), imm, jmm, avg)
  imm=i-(nx+1)+imm
  jmm=j-(nx+1)+jmm
  lon_twc = rlon(imm)
  lat_twc = rlat(jmm)
  twc_max = tbar(imm, jmm) - avg
  xx      = lon_twc - lon_psl
  yy      = lat_twc - lat_psl
  rr      = xx*xx + yy*yy
  if( rr >= crit_dist ) ierr_pos = 1

  exist_twc = (ierr_pos == 0)

  exist_twc = exist_twc!miz .and. (twc_max > crit_twc)
!new
  exist_twc = exist_twc .and. (twc_max > crit_twc)
  if ( .not.exist_twc ) CYCLE
!new
end if

!-------------------------------------------------------------------
! --- STEP 4: CHECK FOR THICKNESS MAX
!-------------------------------------------------------------------

! --- location
if (do_thickness) then
  ierr_pos = 0

  CALL FRPRM( rlon, rlat, thick, thick_dy, thick_dx,  &
              p,    ftol, iter,  fret,     ierr_pos )

  if( ierr_pos == 0 ) then
    thick_max = -fret
    lon_thick = p(1)
    lat_thick = p(2)
    xx        = lon_thick - lon_psl
    yy        = lat_thick - lat_psl
    rr        = xx*xx + yy*yy
    if( rr >= crit_dist ) ierr_pos = 1
  else
    thick_max = 0.0
    lon_thick = 0.0
    lat_thick = 0.0
  endif

  exist_thick = (ierr_pos == 0)

! --- magnitude

  if ( exist_thick ) then

  ierr_mag = 0

  CALL SHAPE( rlon, rlat, thick, thick_dy, thick_dx,     &
              p,    ftol, iter,  fret,     ierr_mag,  crit_thick )

  exist_thick = exist_thick .and. ( ierr_mag == 0 )
 
  endif
end if
!-------------------------------------------------------------------
! --- WE HAVE A TROPICAL STORM. SAVE INFO ABOUT STORM
!-------------------------------------------------------------------

    number = number + 1

  if( number > nsmax ) then
    PRINT *, '***************************************'
    PRINT *, '  GOT TOO MANY STORMS - INCREASE nsmax '
    PRINT *, '***************************************'
    STOP
  endif

        idex(number) = i - nx
        jdex(number) = j - nx

   svort_max(number) =  vort_max
   svort_lon(number) =  lon_vort
   svort_lat(number) =  lat_vort

   swind_max(number) =  wind_max

    spsl_min(number) =  psl_min
    spsl_lon(number) =  lon_psl
    spsl_lat(number) =  lat_psl

    stwc_max(number) =  twc_max
    stwc_lon(number) =  lon_twc
    stwc_lat(number) =  lat_twc
    stwc_is (number) =  exist_twc

if (.not.do_thickness) then
  sthick_max(number) =  10000
  sthick_lon(number) =  0.
  sthick_lat(number) =  0.
  sthick_is (number) =  .true.
end if

!miz  WRITE(*,*) i-nx, j-nx
! zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
  end do
  end do

!===================================================================
! ---  OUTPUT
!===================================================================

  WRITE(iucy,*) iday,  imon, iyear, number, ihour
  WRITE(   *,*) iyear, imon, iday,  ihour,  number

  if( number == 0 ) RETURN

  do i = 1,number
    WRITE(iucy,*) idex(i),       jdex(i),              & 
              spsl_lon(i),   spsl_lat(i),              &
              swind_max(i), svort_max(i), spsl_min(i), &
                stwc_is(i), sthick_is(i), stwc_max(i), sthick_max(i)
  end do

!===================================================================
  end SUBROUTINE TSTORMS


!######################################################################
  SUBROUTINE find_minmax (do_min, var, imm, jmm, avg)
    implicit none

    logical, intent(in)                    :: do_min
    real,    intent(in),    dimension(2*gnx0+1,2*gnx0+1) :: var
    integer, intent(out)                   :: imm, jmm
    real,    intent(out)                   :: avg

    real    ::  varminmax
    integer ::  i,j

    if (do_min) then
       varminmax = MINVAL( var(:,:) )
    else
       varminmax = MAXVAL( var(:,:) )
    end if
    avg = 0.
    do i = 1, 2*gnx0+1
       do j = 1, 2*gnx0+1
          avg = avg + var(i,j)
          if (var(i,j) == varminmax) then
             imm = i
             jmm = j
          end if
       end do
    end do
    avg = avg / ((2.*gnx0+1)*(2.*gnx0+1))
  end SUBROUTINE find_minmax

!######################################################################
  end MODULE TSTORMS_MOD
