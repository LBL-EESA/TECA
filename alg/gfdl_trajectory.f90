MODULE TRAJECTORY_MOD
use iso_c_binding
use iso_fortran_env, ONLY : error_unit
implicit none

interface
!------------------------------------------------------------------
function teca_get_number_of_rows(a_table) result(a_size) bind(C)
  use iso_c_binding
  implicit none
  type(c_ptr), intent(in) :: a_table
  integer(c_long) :: a_size
end function

!------------------------------------------------------------------
subroutine teca_append_trajectory_summary_float(a_table, &
      num_steps, year, month, day, hour, start_lon, start_lat) bind(C)
  use iso_c_binding
  implicit none
  type(c_ptr), intent(inout) :: a_table
  integer(c_int), intent(in) :: num_steps
  integer(c_int), intent(in) :: year, month, day, hour
  real(c_float), intent(in) :: start_lon, start_lat
end subroutine

!------------------------------------------------------------------
subroutine teca_append_trajectory_details_float(a_table, &
      lon, lat, wind, psl, vmax, year, month, day, hour) bind(C)
  use iso_c_binding
  implicit none
  type(c_ptr), intent(inout) :: a_table
  integer(c_int), intent(in) :: year, month, day, hour
  real(c_float), intent(in) :: lon, lat, wind, psl, vmax
end subroutine
end interface

!-------------------------------------------------------------------
interface teca_get_column
!-------------------------------------------------------------------
integer(c_int) function teca_get_column_int(a_table, &
      a_row_i, a_n_rows, a_col_id, a_array) bind(C)
  use iso_c_binding
  implicit none
  type(c_ptr), intent(in) :: a_table
  integer(c_long), intent(in) :: a_row_i, a_n_rows
  character(kind=c_char) :: a_col_id(*)
  integer(c_int), pointer, dimension(:), intent(inout) :: a_array
end function

!-------------------------------------------------------------------
integer(c_int) function teca_get_column_long(a_table, &
      a_row_i, a_n_rows, a_col_id, a_array) bind(C)
  use iso_c_binding
  implicit none
  type(c_ptr), intent(in) :: a_table
  integer(c_long), intent(in) :: a_row_i, a_n_rows
  character(kind=c_char) :: a_col_id(*)
  integer(c_long), pointer, dimension(:), intent(inout) :: a_array
end function

!-------------------------------------------------------------------
integer(c_int) function teca_get_column_float(a_table, &
      a_row_i, a_n_rows, a_col_id, a_array) bind(C)
  use iso_c_binding
  implicit none
  type(c_ptr), intent(in) :: a_table
  integer(c_long), intent(in) :: a_row_i, a_n_rows
  character(kind=c_char) :: a_col_id(*)
  real(c_float), pointer, dimension(:), intent(inout) :: a_array
end function

!-------------------------------------------------------------------
integer(c_int) function teca_get_column_double(a_table, &
      a_row_i, a_n_rows, a_col_id, a_array) bind(C)
  use iso_c_binding
  implicit none
  type(c_ptr), intent(in) :: a_table
  integer(c_long), intent(in) :: a_row_i, a_n_rows
  character(kind=c_char) :: a_col_id(*)
  real(c_double), pointer, dimension(:), intent(inout) :: a_array
end function
end interface

! unfortunately fortran doesn't support array's of pointers directly.
! the work around is to use an array of struct's where the struct conatins
! a member to hold the pointer to an array
!-------------------------------------------------------------------
type float_pointer
  real(c_float), pointer, dimension(:) :: m_data
end type

!-------------------------------------------------------------------
type int_pointer
  integer(c_int), pointer, dimension(:) :: m_data
end type

!-------------------------------------------------------------------
type logical_pointer
  logical, pointer, dimension(:) :: m_data
end type

contains
!-------------------------------------------------------------------
subroutine alloc_int_array(ptr, n)
  implicit none
  integer, pointer, dimension(:), intent(inout) :: ptr
  integer*8, intent(in) :: n
  integer :: i_err
  allocate(ptr(n), stat = i_err)
  if (i_err.ne.0) then
    write(error_unit, *)"Error: Failed to allocate int array."
    stop
  end if
end subroutine

!-------------------------------------------------------------------
subroutine alloc_int_pointer_array(ptr, n)
  implicit none
  type(int_pointer), pointer, dimension(:), intent(inout) :: ptr
  integer*8, intent(in) :: n
  integer*8 :: i
  integer :: i_err
  allocate(ptr(n), stat = i_err)
  if (i_err.ne.0) then
    write(error_unit, *)"Error: Failed to allocate int_pointer array."
    stop
  end if
  do i = 1,n
      ptr(i)%m_data => null()
  end do
end subroutine

!-------------------------------------------------------------------
subroutine dealloc_int_pointer_array(ptr, n)
  implicit none
  type(int_pointer), pointer, dimension(:), intent(inout) :: ptr
  integer*8, intent(in) :: n
  integer*8 :: i
  do i = 1,n
    if (associated(ptr(i)%m_data)) then
      deallocate(ptr(i)%m_data)
      ptr(i)%m_data => null()
    end if
  end do
end subroutine

!-------------------------------------------------------------------
subroutine alloc_float_array(ptr, n)
  implicit none
  real(c_float), pointer, dimension(:), intent(inout) :: ptr
  integer*8, intent(in) :: n
  integer :: i_err
  allocate(ptr(n), stat = i_err)
  if (i_err.ne.0) then
    write(error_unit, *)"Error: Failed to allocate float array."
    stop
  end if
end subroutine

!-------------------------------------------------------------------
subroutine alloc_float_pointer_array(ptr, n)
  implicit none
  type(float_pointer), pointer, dimension(:), intent(inout) :: ptr
  integer*8, intent(in) :: n
  integer*8 :: i
  integer :: i_err
  allocate(ptr(n), stat = i_err)
  if (i_err.ne.0) then
    write(error_unit, *)"Error: Failed to allocate float_pointer array."
    stop
  end if
  do i = 1,n
      ptr(i)%m_data => null()
  end do
end subroutine

!-------------------------------------------------------------------
subroutine dealloc_float_pointer_array(ptr, n)
  implicit none
  type(float_pointer), pointer, dimension(:), intent(inout) :: ptr
  integer*8, intent(in) :: n
  integer*8 :: i
  do i = 1,n
    if (associated(ptr(i)%m_data)) then
      deallocate(ptr(i)%m_data)
      ptr(i)%m_data => null()
    end if
  end do
end subroutine

!-------------------------------------------------------------------
subroutine alloc_logical_array(ptr, n)
  implicit none
  logical, pointer, dimension(:), intent(inout) :: ptr
  integer*8, intent(in) :: n
  integer :: i_err
  allocate(ptr(n), stat = i_err)
  if (i_err.ne.0) then
    write(error_unit, *)"Error: Failed to allocate logical array."
    stop
  end if
end subroutine

!-------------------------------------------------------------------
subroutine alloc_logical_pointer_array(ptr, n)
  implicit none
  type(logical_pointer), pointer, dimension(:), intent(inout) :: ptr
  integer*8, intent(in) :: n
  integer*8 :: i
  integer :: i_err
  allocate(ptr(n), stat = i_err)
  if (i_err.ne.0) then
    write(error_unit, *)"Error: Failed to allocate logical_pointer array."
    stop
  end if
  do i = 1,n
      ptr(i)%m_data => null()
  end do
end subroutine

!-------------------------------------------------------------------
subroutine dealloc_logical_pointer_array(ptr, n)
  implicit none
  type(logical_pointer), pointer, dimension(:), intent(inout) :: ptr
  integer*8, intent(in) :: n
  integer*8 :: i
  do i = 1,n
    if (associated(ptr(i)%m_data)) then
      deallocate(ptr(i)%m_data)
      ptr(i)%m_data => null()
    end if
  end do
end subroutine

!-------------------------------------------------------------------
! defaults
!  rcrit  = 900.0
!  wcrit  = 17.0
!  wcritm = 17.0
!  vcrit  = 3.5e-5  !miz
!  twc_crit   =  0.5 !miz
!  thick_crit =  50. !miz
!  nwcrit  =   2
!  do_filt = .true.
!  nlat =  40. !miz
!  slat = -40. !miz
!  do_spline = .false.
!  do_thickness = .false.
integer(c_int) function gfdl_tc_trajectory_float(tc_summary, &
      tc_details, rcrit, wcrit, wcritm, nwcrit, vcrit, &
      twc_crit, thick_crit, nlat, slat, do_spline, &
      do_thickness, traj_summary, traj_details) result(return_value) bind(C)
  use iso_c_binding
  use iso_fortran_env, only : error_unit
  implicit none

  type(c_ptr), intent(in) :: tc_summary, tc_details
  type(c_ptr), intent(inout) :: traj_summary, traj_details
  real(c_float), intent(in) :: rcrit, wcrit, wcritm, &
    vcrit, twc_crit, thick_crit, nwcrit, nlat, slat
  integer(c_int), intent(in) :: do_spline, do_thickness

  real, parameter :: RADIUS = 6371.0
  real, parameter :: PI = 3.14159265358979d0
  real, parameter :: RADIAN = 57.29577951
  real :: rlon_0, rlat_0, rlon_i, rlat_i, dx, dy, dr, xeq

  integer(c_long) :: iday, jcyc, number_of_days

  integer :: bon, num_traj, long_traj
  integer :: l, i1, j1,  ncand
  integer :: inc, nwnd, nwndm, inc1, nr, m
  integer, dimension(1) :: imin

  integer :: day0, month0, year0, hour0, number0
  integer :: day1, month1, year1, hour1
  integer(c_long) :: idex, jdex
  real    :: dday !miz

  ! arrays indexed by day
  integer(c_int), pointer, dimension(:) :: icand, jcand, ix, iy, &
        bon_1, bon_2, day, month, year, hour, event_count

  real(c_float), pointer, dimension(:) :: rtot

  ! array of arrays indexed by day and detected event
  type(float_pointer), pointer, dimension(:) :: rlon, rlat, wind, psl, vmax

  type(logical_pointer), pointer, dimension(:) :: available, exist_wind, &
        exist_vort, exist_twc, exist_thick, exist_maxw

  integer(c_long) :: tc_details_idx, n_events
  integer :: i_err

  integer(c_int), pointer, dimension(:) :: twc_is, thick_is
  real(c_float), pointer, dimension(:) :: twc_max, thick_max

  return_value = 0

  ! --- INPUT DATA
  ! allocate working arrays
  number_of_days = teca_get_number_of_rows(tc_summary)

  call alloc_int_array(icand, number_of_days)
  call alloc_int_array(jcand, number_of_days)
  call alloc_int_array(ix, number_of_days)
  call alloc_int_array(iy, number_of_days)
  call alloc_int_array(bon_1, number_of_days)
  call alloc_int_array(bon_2, number_of_days)
  call alloc_int_array(year, number_of_days)
  call alloc_int_array(month, number_of_days)
  call alloc_int_array(day, number_of_days)
  call alloc_int_array(hour, number_of_days)
  call alloc_int_array(event_count, number_of_days)

  ! pull the summary of events table
  if ((teca_get_column_int(tc_summary, 0_8, number_of_days, "year"//c_null_char, year) .ne. 0) &
    .or. (teca_get_column_int(tc_summary, 0_8, number_of_days, "month"//c_null_char, month) .ne. 0) &
    .or. (teca_get_column_int(tc_summary, 0_8, number_of_days, "day"//c_null_char, day) .ne. 0) &
    .or. (teca_get_column_int(tc_summary, 0_8, number_of_days, "hour"//c_null_char, hour) .ne. 0) &
    .or. (teca_get_column_int(tc_summary, 0_8, number_of_days, "event_count"//c_null_char, event_count) .ne. 0)) &
    then
    write(error_unit, *)"Error: Failed to get the summary table"
    return_value = -1
    return
  end if

  ! allocate the outer event details arrays.
  ! these are arrays of arrays, the inner array holds
  ! data for each day's detections
  call alloc_float_pointer_array(rlon, number_of_days)
  call alloc_float_pointer_array(rlat, number_of_days)
  call alloc_float_pointer_array(wind, number_of_days)
  call alloc_float_pointer_array(psl, number_of_days)
  call alloc_float_pointer_array(vmax, number_of_days)

  call alloc_logical_pointer_array(available, number_of_days)
  call alloc_logical_pointer_array(exist_wind, number_of_days)
  call alloc_logical_pointer_array(exist_vort, number_of_days)
  call alloc_logical_pointer_array(exist_twc, number_of_days)
  call alloc_logical_pointer_array(exist_thick, number_of_days)
  call alloc_logical_pointer_array(exist_maxw, number_of_days)

  tc_details_idx = 0

  do iday = 1,number_of_days

    n_events = event_count(iday)

    ! skip days that have no detections
    if (n_events .lt. 1) cycle

    ! allocate event details arrays for this day
    call alloc_float_array(rlon(iday)%m_data, n_events)
    call alloc_float_array(rlat(iday)%m_data, n_events)
    call alloc_float_array(wind(iday)%m_data, n_events)
    call alloc_float_array(psl(iday)%m_data, n_events)
    call alloc_float_array(vmax(iday)%m_data, n_events)

    call alloc_int_array(twc_is, n_events)
    call alloc_int_array(thick_is, n_events)
    call alloc_float_array(twc_max, n_events)
    call alloc_float_array(thick_max, n_events)

    ! get the event details for this day
    if ((teca_get_column_float(tc_details, tc_details_idx, n_events, "pressure_lon"//c_null_char, rlon(iday)%m_data) .ne. 0) &
      .or. (teca_get_column_float(tc_details, tc_details_idx, n_events, "pressure_lat"//c_null_char, rlat(iday)%m_data) .ne. 0) &
      .or. (teca_get_column_float(tc_details, tc_details_idx, n_events, "wind_max"//c_null_char, wind(iday)%m_data) .ne. 0) &
      .or. (teca_get_column_float(tc_details, tc_details_idx, n_events, "vorticity_max"//c_null_char, vmax(iday)%m_data) .ne. 0) &
      .or. (teca_get_column_float(tc_details, tc_details_idx, n_events, "pressure_min"//c_null_char, psl(iday)%m_data) .ne. 0) &
      .or. (teca_get_column_int(tc_details, tc_details_idx, n_events, "have_core_temp"//c_null_char, twc_is) .ne. 0) &
      .or. (teca_get_column_int(tc_details, tc_details_idx, n_events, "have_thickness"//c_null_char, thick_is) .ne. 0) &
      .or. (teca_get_column_float(tc_details, tc_details_idx, n_events, "core_temp_max"//c_null_char, twc_max) .ne. 0) &
      .or. (teca_get_column_float(tc_details, tc_details_idx, n_events, "thickness_max"//c_null_char, thick_max) .ne. 0)) &
      then
      write(error_unit, *)"Error: Failed to get the event details table at day",iday
      return_value = -1
      return
    end if

    tc_details_idx = tc_details_idx + n_events

    ! scale pressure
    do jcyc = 1,n_events
      psl(iday)%m_data(jcyc) = psl(iday)%m_data(jcyc)*0.01
    end do

    ! allocate trajectory candidate criteria arrays
    ! and compute various criteria
    call alloc_logical_array(exist_wind(iday)%m_data, n_events)
    do jcyc = 1,n_events
      exist_wind(iday)%m_data(jcyc) = (wind(iday)%m_data(jcyc) .ge. wcrit)
    end do

    call alloc_logical_array(exist_maxw(iday)%m_data, n_events)
    do jcyc = 1,n_events
      exist_maxw(iday)%m_data(jcyc) = (wind(iday)%m_data(jcyc) .ge. wcritm)
    end do

    call alloc_logical_array(exist_vort(iday)%m_data, n_events)
    do jcyc = 1,n_events
      exist_vort(iday)%m_data(jcyc) = (vmax(iday)%m_data(jcyc) .ge. vcrit)
    end do

    call alloc_logical_array(exist_twc(iday)%m_data, n_events)
    if (do_spline .ne. 0) then
      do jcyc = 1,n_events
          exist_twc(iday)%m_data(jcyc) = (twc_is(jcyc) .ne. 0)
      end do
    else
      do jcyc = 1,n_events
          exist_twc(iday)%m_data(jcyc) = &
            ((twc_is(jcyc) .ne. 0) .and. (twc_max(jcyc) .ge. twc_crit))
      end do
    end if

    call alloc_logical_array(exist_thick(iday)%m_data, n_events)
    if ((do_spline .ne. 0) .and. (do_thickness .ne. 0)) then
      do jcyc = 1,n_events
        exist_thick(iday)%m_data(jcyc) = (thick_is(jcyc) .ne. 0)
      end do
    else
      do jcyc = 1,n_events
        exist_thick(iday)%m_data(jcyc) = &
          ((thick_is(jcyc) .ne. 0) .and. (thick_max(jcyc) .ge. thick_crit))
      end do
    end if

    call alloc_logical_array(available(iday)%m_data, n_events)
    do jcyc = 1,n_events
      available(iday)%m_data(jcyc) = .true.
    end do

    deallocate(twc_is)
    deallocate(thick_is)
    deallocate(twc_max)
    deallocate(thick_max)

  end do

  ! --- STEP 2: EVALUATION OF TRAJECTORIES
  dday = (hour(2)-hour(1))/24.
  if (dday == 0) dday = 1.
  write(*,*) 'dday=',dday

  num_traj  = 0

  do iday = 1,number_of_days-1
    year0 = year(iday)
    month0 = month(iday)
    day0 = day(iday)
    hour0 = hour(iday)

    do jcyc  = 1,event_count(iday)

      long_traj = 1
      ix(1) = iday
      iy(1) = jcyc

      if (available(iday)%m_data(jcyc) .and. exist_wind(iday)%m_data(jcyc) ) then

        l = iday + 1
        i1 = iday
        j1 = jcyc

10      continue
        ncand = 0

        if( l > number_of_days ) go to 999

        ! --- check for candidates on following day
        rlon_0 = rlon(i1)%m_data(j1) / RADIAN
        rlat_0 = rlat(i1)%m_data(j1) / RADIAN

         ! loop over next time instance's storms
        do inc = 1,event_count(l)

          if (available(l)%m_data(inc)) then
            ! before last time instance in cyclone file
            rlon_i = rlon(l)%m_data(inc) / RADIAN
            rlat_i = rlat(l)%m_data(inc) / RADIAN

            dr = RADIUS * acos( sin(rlat_0) * sin(rlat_i) +  cos(rlat_0) * cos(rlat_i) * cos(rlon_i - rlon_0) )
            xeq = rlat_i * rlat_0

            if ( (dr <= rcrit*dday) .and. (xeq > 0.0) ) then
              ! within rcrit conditional
              ncand = ncand + 1
              icand(ncand) = l
              jcand(ncand) = inc
            end if
          end if
        end do

999     continue

        ! --- no more candidate storms
        ! this is the condition for the end of the track
        if (ncand .eq. 0) then

          ! --- check winds
          nwnd = 0
          nwndm = 0

          do inc = 1,long_traj
            if(exist_wind(ix(inc))%m_data(iy(inc)) .and. &
               exist_vort(ix(inc))%m_data(iy(inc)) .and. & !miz
               exist_twc(ix(inc))%m_data(iy(inc)) .and. &
               exist_thick(ix(inc))%m_data(iy(inc))) nwnd = nwnd + 1

            if(exist_maxw(ix(inc))%m_data(iy(inc)) .and. &
               exist_vort(ix(inc))%m_data(iy(inc)) .and. & !miz
               exist_twc(ix(inc))%m_data(iy(inc)) .and. &
               exist_thick(ix(inc))%m_data(iy(inc))) nwndm = nwndm + 1
          end do

          if ((long_traj > 1) .and. (nwnd .ge. nwcrit/dday) .and. (nwndm .gt. 0)) then

            num_traj  = num_traj + 1

            ! --- output trajectory info
            call teca_append_trajectory_summary_float(traj_summary, &
              long_traj, year0, month0, day0, hour0, &
              rlon(ix(1))%m_data(iy(1)), rlat(ix(1))%m_data(iy(1)))

            do inc1 = 1,long_traj

              year1 = year(iday+inc1-1)
              month1 = month(iday+inc1-1)
              day1 = day(iday+inc1-1)
              hour1 = hour(iday+inc1-1)

              call teca_append_trajectory_details_float(traj_details, &
                rlon(ix(inc1))%m_data(iy(inc1)), rlat(ix(inc1))%m_data(iy(inc1)), &
                wind(ix(inc1))%m_data(iy(inc1)), psl(ix(inc1))%m_data(iy(inc1)), &
                vmax(ix(inc1))%m_data(iy(inc1)), year1, month1, day1, hour1)

            end do

            ! --- eliminate storms used for this trajectory
            do inc1 = 1,long_traj
               available(ix(inc1))%m_data(iy(inc1)) = .false.
            end do

          end if
        end if

        ! --- one candidate storm
        if( ncand .eq. 1 ) then

          long_traj = long_traj + 1
          ix(long_traj) = icand(1)
          iy(long_traj) = jcand(1)

          l = l + 1
          i1 = ix(long_traj)
          j1 = iy(long_traj)

          goto 10
        end if

        ! --- more than one candidate storm
        if (ncand .gt. 1) then
          rlon_0 = rlon(i1)%m_data(j1)
          rlat_0 = ABS(rlat(i1)%m_data(j1))
          bon  = 0

          do inc = 1,ncand
            rlon_i = rlon(l)%m_data(inc)
            rlat_i = ABS(rlat(l)%m_data(inc))

            if( sin(rlon_i) <= sin(rlon_0) ) then ! mod to deal with 0 lon
              if( rlat_i >= rlat_0 ) then

                bon = bon + 1
                bon_1(bon) = icand(inc)
                bon_2(bon) = jcand(inc)

              end if
            end if
          end do

          if( bon == 1 ) then
            long_traj = long_traj + 1
            ix(long_traj) = bon_1(1)
            iy(long_traj) = bon_2(1)

            l  = l + 1
            i1 = ix(long_traj)
            j1 = iy(long_traj)

            goto 10
          end if

          if ( bon >= 2 ) then
            do inc = 1,bon
               rtot(inc) = RADIUS*acos(sin(rlat(bon_1(inc))%m_data(bon_2(inc)))*sin(rlat(i1)%m_data(j1)) &
                 + cos(rlat(bon_1(inc))%m_data(bon_2(inc)))*cos(rlat(i1)%m_data(j1))*cos(rlon(bon_1(inc))%m_data(bon_2(inc)) &
                   - rlon(i1)%m_data(j1)))
            end do

            imin = MINLOC(rtot(1:bon))

            long_traj = long_traj + 1
            ix(long_traj) = bon_1(imin(1))
            iy(long_traj) = bon_2(imin(1))
            l  = l + 1
            i1 = ix(long_traj)
            j1 = iy(long_traj)

            goto  10
          end if

          if( bon == 0 ) then
            do inc = 1,ncand
              rtot(inc) = RADIUS*acos(sin(rlat(icand(inc))%m_data(jcand(inc)))*sin(rlat(i1)%m_data(j1)) &
                + cos(rlat(icand(inc))%m_data(jcand(inc)))*cos(rlat(i1)%m_data(j1))*cos(rlon(icand(inc))%m_data(jcand(inc)) &
                  - rlon(i1)%m_data(j1)) )
            end do

            imin = MINLOC( rtot(1:ncand) )

            long_traj = long_traj + 1
            ix(long_traj) = icand(imin(1))
            iy(long_traj) = jcand(imin(1))
            l = l + 1
            i1 = ix(long_traj)
            j1 = iy(long_traj)

            goto  10
          endif
        end if
      end if
    end do
  end do

  deallocate(icand)
  deallocate(jcand)
  deallocate(ix)
  deallocate(iy)
  deallocate(bon_1)
  deallocate(bon_2)
  deallocate(year)
  deallocate(month)
  deallocate(day)
  deallocate(hour)
  deallocate(event_count)

  call dealloc_float_pointer_array(rlon, number_of_days)
  call dealloc_float_pointer_array(rlat, number_of_days)
  call dealloc_float_pointer_array(wind, number_of_days)
  call dealloc_float_pointer_array(psl, number_of_days)
  call dealloc_float_pointer_array(vmax, number_of_days)

  call dealloc_logical_pointer_array(available, number_of_days)
  call dealloc_logical_pointer_array(exist_wind, number_of_days)
  call dealloc_logical_pointer_array(exist_vort, number_of_days)
  call dealloc_logical_pointer_array(exist_twc, number_of_days)
  call dealloc_logical_pointer_array(exist_thick, number_of_days)
  call dealloc_logical_pointer_array(exist_maxw, number_of_days)

  ! TODO -- these would go in downstream algorithms
  ! --- FILTER DATA
  !if (do_filt) CALL TS_FILTER(nlat,slat)
  ! --- STATS
  !call TS_STATS(do_filt)

  return_value = 0
  return
end function
end module
