module teca_tc_trajectory_util_mod
use iso_c_binding
use iso_fortran_env, ONLY : error_unit
implicit none

contains
!-----------------------------------------------------------------------------
subroutine get_step_offsets(step_ids, n_rows, &
  n_steps, step_counts, step_offsets) bind(C)
  use iso_c_binding
  implicit none
  integer(c_long), intent(in), dimension(n_rows) :: step_ids
  integer(c_long), intent(in) :: n_rows
  integer(c_long), intent(out) :: n_steps
  integer(c_long), intent(out), dimension(:) :: &
    step_counts, step_offsets
  integer(c_long) :: q, i, n_m1

  ! count unique number of steps
  n_steps = 1
  n_m1 = n_rows - 1
  do i = 1,n_m1
    if (step_ids(i) .ne. step_ids(i+1)) &
      n_steps = n_steps + 1
  enddo

  ! compute num storms in each step
  q = 1
  do i = 1,n_steps
    step_counts(i) = 1
    do while ((q .lt. n_rows) .and. (step_ids(q) .eq. step_ids(q+1)))
      step_counts(i) = step_counts(i) + 1
      q = q + 1
    enddo
    q = q + 1
  enddo

  ! compute the offset to the first storm in each step
  step_offsets(1) = 1
  do i = 2,n_steps
    step_offsets(i) = step_offsets(i-1) + step_counts(i-1)
  enddo

  write(error_unit,*)'n_steps=',n_steps
  do i = 1,n_steps
    write(error_unit,*)'step=',i,'step_counts=',step_counts(i), &
      'step_offsets=',step_offsets(i),step_ids(step_offsets(i))
  enddo

end subroutine
end module
