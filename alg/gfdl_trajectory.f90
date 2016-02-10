  PROGRAM TRAJECTORY
!===================================================================
!  --- DETECT TROPICAL STORM TRAJECTORIES 
!===================================================================
  use TS_TOOLS_MOD
  implicit none
!-------------------------------------------------------------------

  integer, parameter :: numax    = 500  ! cyclones per time instance (typically less than 100 in HiRam)
  integer, parameter :: nrmx     = 50000 ! maximum of 21 years of 1460 time instances

  integer, parameter :: iucyc = 12
  integer, parameter :: iutra = 13
  integer, parameter :: iuori = 14
  integer, parameter :: iusta = 15
  integer, parameter :: iutrv = 16

!-------------------------------------------------------------------

  real,  parameter :: RADIUS = 6371.0
  real :: PI, RADIAN
  real :: rlon_0, rlat_0, rlon_i, rlat_i, dx, dy, dr, xeq

!-------------------------------------------------------------------

  integer :: iday, jcyc, mdays

  integer, dimension(numax)  :: icand, jcand
  integer, dimension(numax)  :: ix, iy, bon_1, bon_2
  real,    dimension(numax)  :: rtot

  integer :: bon, num_traj, long_traj
  integer :: l, i1, j1,  ncand 
  integer :: inc, nwnd, nwndm, inc1, nr, m

  integer, dimension(1) :: imin

!-------------------------------------------------------------------

  integer :: day0, month0, year0, hour0, number0
  integer :: day1, month1, year1, hour1
  integer :: idex, jdex

  real    :: psl_lon, psl_lat, wind_max, vort_max, psl_min, twc_max, thick_max !miz
  logical :: twc_is,  thick_is
  real    :: dday !miz

  integer, dimension(nrmx)        :: day,  month, year, number, hour
  real,    dimension(nrmx,numax)  :: rlon, rlat, wind,  psl, vmax 
  logical, dimension(nrmx,numax)  :: available
  logical, dimension(nrmx,numax)  :: exist_wind, exist_vort, exist_twc, exist_thick, exist_maxw

!-------------------------------------------------------------------

  real     ::  rcrit  = 900.0
  real     ::  wcrit  =  17.0
  real     ::  wcritm =  17.0
  real     ::  vcrit  =  3.5e-5  !miz
  real     ::  twc_crit   =  0.5 !miz
  real     ::  thick_crit =  50. !miz
  real     :: nwcrit  =   2
  logical  :: do_filt = .true.
  real     :: nlat =  40. !miz
  real     :: slat = -40. !miz
  logical  :: do_spline = .false.
  logical  :: do_thickness = .false.

  namelist / input /  rcrit, wcrit, wcritm, nwcrit, do_filt, &
       vcrit, twc_crit, thick_crit, nlat, slat, do_spline, do_thickness !miz

!===================================================================

  PI     = 4.0*ATAN(1.0)
  RADIAN = 180.0/PI

 103 FORMAT( 2f8.2,   4i6 )
 104 FORMAT( 'start', 5i6 )
 105 FORMAT( 4f8.2,   4i6 )
 106 FORMAT( 4f8.2, f13.8, 4i6 )

  READ( *, input )

!===================================================================
! --- ETAPE 1:  LECTURE DU FICHIER DE DONNEES
!===================================================================

           rlon(:,:) = 0.0
           rlat(:,:) = 0.0
           wind(:,:) = 0.0
     exist_wind(:,:) = .false.
     exist_maxw(:,:) = .false.
     exist_vort(:,:) = .false.
      exist_twc(:,:) = .false.
    exist_thick(:,:) = .false.
      available(:,:) = .false.

  OPEN( iutra, FILE = 'traj', STATUS = 'unknown' )
  OPEN( iuori, FILE = 'ori',  STATUS = 'unknown' )
  OPEN( iutrv, FILE = 'trav', STATUS = 'unknown' )

!===================================================================
! --- INPUT DATA
!===================================================================

  OPEN( iucyc, FILE = 'cyclones', STATUS = 'unknown' )

  do iday = 1,nrmx
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  READ(iucyc,*,end=101) day0, month0, year0, number0, hour0
  WRITE(*,*)      iday, hour0, day0, month0, year0, number0

    number(iday) = number0
      year(iday) = year0
     month(iday) = month0
       day(iday) = day0
      hour(iday) = hour0

  if( number0 > 0 ) then
  do jcyc = 1,number0
    READ(iucyc,*,err=201)  idex,     jdex,              &
                           psl_lon,  psl_lat,           &
                           wind_max, vort_max, psl_min, &
                           twc_is,   thick_is, twc_max, thick_max !miz
 
           rlon(iday,jcyc) = psl_lon
           rlat(iday,jcyc) = psl_lat
            psl(iday,jcyc) = psl_min*0.01
           wind(iday,jcyc) = wind_max
           vmax(iday,jcyc) = vort_max
     exist_wind(iday,jcyc) = ( wind_max >= wcrit )
     exist_maxw(iday,jcyc) = ( wind_max >= wcritm )
     exist_vort(iday,jcyc) = ( vort_max >= vcrit) !miz
   if (do_spline) then
      exist_twc(iday,jcyc) = twc_is
   else
      exist_twc(iday,jcyc) = twc_is   .and. (twc_max   >= twc_crit)   !miz: twc_is
   end if
   if (do_spline .and. do_thickness) then
      exist_thick(iday,jcyc) = thick_is
   else
      exist_thick(iday,jcyc) = thick_is .and. (thick_max >= thick_crit) !miz: thick_is
   end if
      available(iday,jcyc) = .true.
      cycle 

      201 continue
      print *, ' BAD DATA AT JCYC: ', jcyc
  
  end do
  end if

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  end do

      mdays = iday 
      go to 102
  101 continue
      PRINT *, '*********************************************'
      PRINT *, '  End of file reading record ', iday
      PRINT *, '*********************************************'
      mdays = iday - 1
  102 continue

  CLOSE(iucyc)

!===================================================================
! --- STEP 2: EVALUATION OF TRAJECTORIES
!===================================================================

  dday=(hour(2)-hour(1))/24.
  if (dday == 0) dday = 1.
  write(*,*) 'dday=',dday
  
 
  num_traj  = 0

  do iday = 1,mdays-1
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    year0 =  year(iday)
   month0 = month(iday)
     day0 =   day(iday)
    hour0 =  hour(iday)

  do jcyc  = 1,number(iday)
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  long_traj = 1
  ix(1)     = iday
  iy(1)     = jcyc

! if( available(iday,jcyc) ) then
  if( available(iday,jcyc) .and. &
     exist_wind(iday,jcyc) ) then
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

      l     = iday + 1
      i1    = iday
      j1    = jcyc
 10   ncand = 0

!$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  if( l > mdays ) go to 999
!$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

! --- check for candidates on following day

  rlon_0 = rlon(i1,j1) / RADIAN
  rlat_0 = rlat(i1,j1) / RADIAN

  do inc = 1,number(l)
  if( available(l,inc) ) then
  rlon_i = rlon(l,inc) / RADIAN
  rlat_i = rlat(l,inc) / RADIAN

! Tim. M
! modify longitudes near 0 deg 
! if ( rlon_0 > 350 degrees ) then
!   if (rlon_i < 10) then
!    rlon_i = rlon_i + 360
!   endif
! else if ( rlon_0 < 10 deg ) then
!    if (rlon_i > 350 deg ) then
!!    rlon_i = rlon_i - 360
!    endif
! end
!  dx = RADIUS * ( rlon_i - rlon_0 ) * cos(rlat_0) 
!  dy = RADIUS * ( rlat_i - rlat_0 )
!  dr = sqrt( dx*dx + dy*dy )
 dr = RADIUS * acos( sin(rlat_0) * sin(rlat_i) +  cos(rlat_0) * cos(rlat_i) * cos(rlon_i - rlon_0) )
 xeq = rlat_i * rlat_0

  if ( (dr <= rcrit*dday) .and. (xeq > 0.0) ) then
           ncand  = ncand + 1
     icand(ncand) = l
     jcand(ncand) = inc
  end if ! end if within rcrit conditional
  end if ! end if before last time instance in cyclone file
  end do ! end do loop over next time instance's storms

!$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
 999 continue       
!$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

! --- no more candidate storms
! this is the condition for the end of the track
  if( ncand == 0 ) then 
! zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz

! --- check winds

                                       nwnd = 0
                                       nwndm = 0
  do inc = 1,long_traj
    if( exist_wind(ix(inc),iy(inc)) .and. &
        exist_vort(ix(inc),iy(inc)) .and. & !miz
         exist_twc(ix(inc),iy(inc)) .and. &
       exist_thick(ix(inc),iy(inc))  ) nwnd = nwnd + 1

    if( exist_maxw(ix(inc),iy(inc)) .and. &
        exist_vort(ix(inc),iy(inc)) .and. & !miz
         exist_twc(ix(inc),iy(inc)) .and. &
       exist_thick(ix(inc),iy(inc))  ) nwndm = nwndm + 1
  end do

  if(( long_traj > 1 ).and.( nwnd  >= nwcrit/dday ).and.(nwndm > 0)) then
!  if(( long_traj > 1 ).and.( nwnd  >= nwcrit/dday )) then
! zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
 
   num_traj  = num_traj + 1

! --- output trajectory info

      WRITE(iutra,104) long_traj, year0, month0, day0, hour0
      WRITE(iutrv,104) long_traj, year0, month0, day0, hour0

    do inc1 = 1,long_traj

         year1 =  year(iday+inc1-1)
        month1 = month(iday+inc1-1)
          day1 =   day(iday+inc1-1)
         hour1 =  hour(iday+inc1-1)

      WRITE(iutra,105) rlon(ix(inc1),iy(inc1)), &
                       rlat(ix(inc1),iy(inc1)), &
                       wind(ix(inc1),iy(inc1)), &
                        psl(ix(inc1),iy(inc1)), &
                       year1, month1, day1, hour1
      WRITE(iutrv,106) rlon(ix(inc1),iy(inc1)), &
                       rlat(ix(inc1),iy(inc1)), &
                       wind(ix(inc1),iy(inc1)), &
                       psl(ix(inc1),iy(inc1)), &
                       vmax(ix(inc1),iy(inc1)), &
                       year1, month1, day1, hour1
    end do
 
    WRITE(iuori,103) rlon(ix(1),iy(1)), rlat(ix(1),iy(1)), &
                   year0, month0, day0, hour0

! --- eliminate storms used for this trajectory
  do inc1 = 1,long_traj
     available(ix(inc1),iy(inc1)) = .false.
  end do

! zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
  end if
  end if

! --- one candidate storm
  if( ncand == 1 ) then
! xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
     long_traj  = long_traj + 1
  ix(long_traj) = icand(1)
  iy(long_traj) = jcand(1)

  l  = l + 1
  i1 = ix(long_traj)
  j1 = iy(long_traj)

  goto 10
! xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  end if

! --- more than one candidate storm
  if( ncand > 1 ) then
! xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

  rlon_0 =     rlon(i1,j1)
  rlat_0 = ABS(rlat(i1,j1))

          bon  = 0
  do inc = 1,ncand
      rlon_i =     rlon(l,inc)
      rlat_i = ABS(rlat(l,inc))
!  if( rlon_i <= rlon_0 ) then
  if( sin(rlon_i) <= sin(rlon_0) ) then ! mod to deal with 0 lon
  if( rlat_i >= rlat_0 ) then
          bon  = bon + 1
    bon_1(bon) = icand(inc)
    bon_2(bon) = jcand(inc)
  end if
  end if
  end do

  if( bon == 1 ) then
! --------------------------------
     long_traj  = long_traj + 1
  ix(long_traj) = bon_1(1)
  iy(long_traj) = bon_2(1)

  l  = l + 1
  i1 = ix(long_traj)
  j1 = iy(long_traj)

  goto 10
! --------------------------------
  end if

  if ( bon >= 2 ) then
! --------------------------------
  do inc = 1,bon
!    dx = ( rlon(bon_1(inc),bon_2(inc)) - rlon(i1,j1) )
!    dy = ( rlat(bon_1(inc),bon_2(inc)) - rlat(i1,j1) )
!    rtot(inc) = sqrt( dx*dx + dy*dy )
     rtot(inc) = RADIUS * acos( sin(rlat(bon_1(inc),bon_2(inc))) * sin(rlat(i1,j1)) + cos(rlat(bon_1(inc),bon_2(inc))) * cos(rlat(i1,j1)) * cos(rlon(bon_1(inc),bon_2(inc)) - rlon(i1,j1)) )
  end do

  imin = MINLOC( rtot(1:bon) )

       long_traj  = long_traj + 1
    ix(long_traj) = bon_1(imin(1))
    iy(long_traj) = bon_2(imin(1))
    l  = l + 1
    i1 = ix(long_traj)
    j1 = iy(long_traj)
  goto  10
! --------------------------------
  end if

  if( bon == 0 ) then
! --------------------------------
  do inc = 1,ncand
!    dx = ( rlon(icand(inc),jcand(inc)) - rlon(i1,j1) )
!    dy = ( rlat(icand(inc),jcand(inc)) - rlat(i1,j1) )
!    rtot(inc) = sqrt( dx*dx + dy*dy )
     rtot(inc) = RADIUS * acos( sin(rlat(icand(inc),jcand(inc))) * sin(rlat(i1,j1)) + cos(rlat(icand(inc),jcand(inc))) * cos(rlat(i1,j1)) * cos(rlon(icand(inc),jcand(inc)) - rlon(i1,j1)) )
  end do

  imin = MINLOC( rtot(1:ncand) )

       long_traj  = long_traj + 1
    ix(long_traj) = icand(imin(1))
    iy(long_traj) = jcand(imin(1))
    l  = l + 1
    i1 = ix(long_traj)
    j1 = iy(long_traj)
  goto  10
! --------------------------------
  endif

! xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  end if

! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  end if
  end do
  end do

!$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
! 999 continue
!     print *, ' STOP 999'
!$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

  CLOSE( iutra )
  CLOSE( iuori )
  CLOSE( iutrv )


!===================================================================
! --- FILTER DATA
!===================================================================

  if( do_filt ) CALL TS_FILTER(nlat,slat)

!===================================================================
! --- STATS
!===================================================================
 
  CALL TS_STATS ( do_filt )

!===================================================================
  end PROGRAM TRAJECTORY
