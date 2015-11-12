  MODULE TSGPAD_MOD
  implicit none
  contains

!######################################################################

  SUBROUTINE GPAD2( Gxx, xx )
  implicit none
  real, intent(in),  dimension(:,:) :: Gxx
  real, intent(out), dimension(:,:) :: xx
  integer :: ix, jx
  integer :: nx, nx2, nxp1

  ix = SIZE( Gxx, 1 )
  jx = SIZE( Gxx, 2 )

  nx2  = SIZE( xx, 1 ) - SIZE( Gxx, 1 )
  nx   = nx2 / 2                       
  nxp1 = nx + 1                        

  xx(nxp1:ix+nx,jx+nx:nxp1:-1) = Gxx(:,:) 

  xx(      1:nx,       nxp1:jx+nx ) = xx(ix+1:ix+nx, nxp1:jx+nx) 
  xx(ix+nxp1:ix+nx2,   nxp1:jx+nx ) = xx(nxp1:nx2,   nxp1:jx+nx) 
  xx(      1:ix+nx2,      1:nx    ) = xx(   1:ix+nx2,jx+1:jx+nx) 
  xx(      1:ix+nx2,jx+nxp1:jx+nx2) = xx(   1:ix+nx2,nxp1:nx2  ) 
 
  end SUBROUTINE GPAD2

!######################################################################

  SUBROUTINE GPAD1( Gxx, xx, iflip )
  implicit none
  integer, intent(in)                :: iflip
  real,    intent(in),  dimension(:) :: Gxx
  real,    intent(out), dimension(:) :: xx
  integer :: i, ix, ip 
  integer :: nx, nx2, nxp1
  real    :: dx

  nx2  = SIZE( xx, 1 ) - SIZE( Gxx, 1 )
  nx   = nx2 / 2                       
  nxp1 = nx + 1                        

  ix = SIZE( Gxx )
  ip = SIZE(  xx ) + 1

  if( iflip .eq. 0 ) then
     xx(nxp1:ix+nx) = Gxx(:) 
  else
     xx(ix+nx:nxp1:-1) = Gxx(:) 
  end if

  dx = xx(nx+2) - xx(nx+1)

  do i = 1,nx
     xx(   i) =  xx(   nxp1) - ( nxp1 - i ) * dx
     xx(ip-i) =  xx(ip-nxp1) + ( nxp1 - i ) * dx 
  end do

  end SUBROUTINE GPAD1

!######################################################################
  end MODULE TSGPAD_MOD
