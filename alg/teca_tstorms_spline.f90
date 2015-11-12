  MODULE SPLINE_MOD
!-----------------------------------------------------------------------
! --- SUBROUTINE SPLIE2
! --- SUBROUTINE SPLIE3
! --- SUBROUTINE SPLINE 
! --- SUBROUTINE SPLINT   
! --- SUBROUTINE SPLIN2   
! --- SUBROUTINE SPLIN3   
! --- SUBROUTINE FRPRM    
! --- SUBROUTINE LINMIN     
! --- SUBROUTINE LINMIN1     
! --- SUBROUTINE F1D1M     
! --- SUBROUTINE MNBRAK    
! --- SUBROUTINE BRENT    
! --- SUBROUTINE SHAPE    
!-----------------------------------------------------------------------
  implicit none

  integer, private, parameter       :: nmax = 2
  integer, private                  :: ncom
  real,    private, dimension(nmax) :: pcom, xicom

  contains
!########################################################################

  SUBROUTINE SPLIE2( x1a, x2a, ya, y2a )
!-----------------------------------------------------------------------
! Given an M by N tabulated function YA and tabulated independant 
! variables X1A (M values) and X2A (N values), this routine constructs
! one-dimensional natural cubic splines of the rows and returns 
! the second-derivatives in the array Y2A.
!-----------------------------------------------------------------------
  implicit none

  real, intent(in),  dimension(:)   :: x1a   ! ===> NOT REFERENCED
  real, intent(in),  dimension(:)   :: x2a   
  real, intent(in),  dimension(:,:) :: ya   
  real, intent(out), dimension(:,:) :: y2a   

  real,  dimension(SIZE(ya,2)) :: ytmp, y2tmp   
  integer                      :: m, n, j

  real, parameter :: ypval = 1.e30

!-----------------------------------------------------------------------

  m = SIZE( ya, 1 )
  n = SIZE( ya, 2 )

  do j = 1,m
! %%%%%%%%%%%%%%

  ytmp(:) = ya(j,:)

  CALL SPLINE( x2a, ytmp, ypval, ypval, y2tmp )

  y2a(j,:) = y2tmp(:)

! %%%%%%%%%%%%%%
  end do

!-----------------------------------------------------------------------
  end SUBROUTINE SPLIE2

!########################################################################

  SUBROUTINE SPLIE3( x1a, x2a, ya, y2a )
!-----------------------------------------------------------------------
! Given an M by N tabulated function YA and tabulated independant
! variables X1A (M values) and X2A (N values), this routine constructs
! one-dimensional natural cubic splines of the rows and returns
! the second-derivatives in the array Y2A.
!-----------------------------------------------------------------------
  implicit none

  real, intent(in),  dimension(:)   :: x1a 
  real, intent(in),  dimension(:)   :: x2a   ! ===> NOT REFERENCED   
  real, intent(in),  dimension(:,:) :: ya   
  real, intent(out), dimension(:,:) :: y2a   

  real,  dimension(SIZE(ya,1)) :: ytmp, y2tmp   
  integer                      :: m, n, k

  real, parameter :: ypval = 1.e30

!-----------------------------------------------------------------------

  m = SIZE( ya, 1 )
  n = SIZE( ya, 2 )

  do k = 1,n
! %%%%%%%%%%%%%%

  ytmp(:) = ya(:,k)

  CALL SPLINE( x1a, ytmp, ypval, ypval, y2tmp )

  y2a(:,k) = y2tmp(:)

! %%%%%%%%%%%%%%
  end do

!-----------------------------------------------------------------------
  end SUBROUTINE SPLIE3

!########################################################################


  SUBROUTINE SPLINE( x, y, yp1, ypn, y2 )
!-----------------------------------------------------------------------
! Given arrays X and Y of lenght N containing a tabulated function,i.e. 
! Yi = f(Xi), with X1<X2<...<Xn, and given values YP1 and YPN for the
! first derivative of the interpolating function at points 1 and N, 
! respectively, this routine returns an array Y2 of length N which
! contains the second derivatives of the interpolating function
! of the tabulated points Xi.
! If YP1 and/or YPN are equal to 1.E30 or larger, the routine 
! is signalled to set the corresponding boundary  condition
! for a natural spline, wuth zero derivative on that boundary.
!-----------------------------------------------------------------------
  implicit none

  real, intent(in)                :: yp1, ypn 
  real, intent(in),  dimension(:) :: x,   y 
  real, intent(out), dimension(:) :: y2

  real, dimension(SIZE(x)) :: u,  sig, p  
  real                     :: qn, un
  integer                  :: n,  i

  real, parameter :: yptest = 0.99e30

!-----------------------------------------------------------------------

  n = SIZE( x )

  if ( yp1 > yptest  ) then
       y2(1) = 0.
        u(1) = 0.
  else
       y2(1) = -0.5
        u(1) = ( 3.0 / ( x(2) - x(1) ) ) * &
                     ( ( y(2) - y(1) ) / ( x(2) - x(1) ) - yp1 )
  endif

  do i = 2,n-1
! %%%%%%%%%%%%%%
  sig(i) = ( x(i) - x(i-1) ) / ( x(i+1) - x(i-1) )
    p(i) = sig(i) * y2(i-1) + 2.
   y2(i) = ( sig(i) - 1. ) / p(i)
    u(i) = ( 6.0 * ( ( y(i+1) - y(i)   )    &
                   / ( x(i+1) - x(i)   )    &
                   - ( y(i)   - y(i-1) )    &
                   / ( x(i)   - x(i-1) ) )  &
                   / ( x(i+1) - x(i-1) ) - sig(i) * u(i-1) ) / p(i)
! %%%%%%%%%%%%%%
  end do

  if ( ypn > yptest ) then
       qn = 0.
       un = 0.
  else
       qn = 0.5
       un = ( 3.0 / ( x(n) - x(n-1) ) ) * &
            ( ypn - ( y(n) - y(n-1) ) / ( x(n) - x(n-1) ) )
  endif

       y2(n) = ( un - qn * u(n-1) ) / ( qn * y2(n-1) + 1. )
  do i = n-1,1,-1
       y2(i) = y2(i) * y2(i+1) + u(i)
  end do

!-----------------------------------------------------------------------
  end SUBROUTINE SPLINE

!########################################################################

  SUBROUTINE SPLINT( xa, ya, y2a, x, y, dy, dyy )
!-----------------------------------------------------------------------
! Given the arrays XA and YA of length N, which tabulate a function
! (with the XA's in order), and given the array Y2A, which is the output
! from spline above, and given a value of X, this routine
! returns a cubic-spline interpolated value Y.
!-----------------------------------------------------------------------
  implicit none

  real, intent(in),  dimension(:) :: xa, ya, y2a 
  real, intent(in)                :: x
  real, intent(out)               :: y,  dy, dyy 

  integer :: n, k, klo, khi
  real    :: h, a, b,   ddy

!-----------------------------------------------------------------------

  n = SIZE( xa )

         klo = 1
         khi = n

1 if ( ( khi - klo ) > 1 ) then
         k = ( khi + klo ) / 2
    if ( xa(k) > x ) then
         khi = k
    else
         klo = k
    endif
  GOTO 1
  endif

  h = xa(khi) - xa(klo)

  if ( h == 0. ) PRINT *, ' *** SPLINT: bad xa input'

  a = ( xa(khi) - x ) / h
  b = ( x - xa(klo) ) / h

  y = a * ya(klo) + b * ya(khi)  &
     + ( ( a**3 - a ) * y2a(klo) &
       + ( b**3 - b ) * y2a(khi) ) * ( h**2 ) / 6.

  dy = ( ya(khi) - ya(klo) ) / h                  &
     - ( 3.0 * a**2 - 1.0 ) * h * y2a(klo) / 6.0  &
     + ( 3.0 * b**2 - 1.0 ) * h * y2a(khi) / 6.0

  ddy = a * y2a(klo) + b * y2a(khi)
! dyy = ddy

!-----------------------------------------------------------------------
  end SUBROUTINE SPLINT

!########################################################################

  SUBROUTINE SPLIN2( x1a, x2a, ya, y2a, x1, x2, y, dy, ddy )
!-----------------------------------------------------------------------
! Given X1A, X2A, YA, M, N as described in SPLIE2 and Y2A as produced 
! by that routine; and given a desired interpolating point X1, X2;
! this routine returns an interpolated function value Y by bicubic
! spline interpolation
!-----------------------------------------------------------------------
  implicit none

  real, intent(in)                 :: x1,  x2
  real, intent(in), dimension(:)   :: x1a, x2a
  real, intent(in), dimension(:,:) :: ya,  y2a
  real, intent(out)                :: y,   dy, ddy 

  real, dimension(SIZE(ya,2)) :: ytmp, y2tmp  
  real, dimension(SIZE(ya,1)) :: yytmp, yy2tmp   

  real    :: y1, y2
  integer :: m,  n, j

  real, parameter :: ypval = 1.e30

!-----------------------------------------------------------------------

  m = SIZE( ya, 1 )
  n = SIZE( ya, 2 )

  do j = 1,m
! %%%%%%%%%%%%%%

   ytmp(:) =  ya(j,:)
  y2tmp(:) = y2a(j,:)

  CALL SPLINT( x2a, ytmp, y2tmp, x2, yytmp(j), y1, y2 )

! %%%%%%%%%%%%%%
  end do

  CALL SPLINE( x1a, yytmp, ypval,  ypval, yy2tmp  )
  CALL SPLINT( x1a, yytmp, yy2tmp, x1, y, dy, ddy )

  ddy = 0.

!-----------------------------------------------------------------------
  end SUBROUTINE SPLIN2

!########################################################################

  SUBROUTINE SPLIN3( x1a, x2a, ya, y2a, x1, x2, y, dy, ddy )
!-----------------------------------------------------------------------
! Given X1A, X2A, YA, M, N as described in SPLIE2 and Y2A as produced 
! by that routine; and given a desired interpolating point X1, X2;
! this routine returns an interpolated function value Y by bicubic
! spline interpolation
!-----------------------------------------------------------------------
  implicit none

  real, intent(in)                 :: x1,  x2
  real, intent(in), dimension(:)   :: x1a, x2a
  real, intent(in), dimension(:,:) :: ya,  y2a
  real, intent(out)                :: y,   dy, ddy 

  real, dimension(SIZE(ya,1)) :: ytmp,  y2tmp  
  real, dimension(SIZE(ya,2)) :: yytmp, yy2tmp   

  real    :: y1, y2
  integer :: m,  n, k

  real, parameter :: ypval = 1.e30

!-----------------------------------------------------------------------

  m = SIZE( ya, 1 )
  n = SIZE( ya, 2 )

  do k = 1,n
! %%%%%%%%%%%%%%

   ytmp(:) =  ya(:,k)
  y2tmp(:) = y2a(:,k)

  CALL SPLINT( x1a, ytmp, y2tmp, x1, yytmp(k), y1, y2 )

! %%%%%%%%%%%%%%
  end do

  CALL SPLINE( x2a, yytmp, ypval,  ypval, yy2tmp  )
  CALL SPLINT( x2a, yytmp, yy2tmp, x2, y, dy, ddy )

  ddy = 0.

!-----------------------------------------------------------------------
  end SUBROUTINE SPLIN3

!########################################################################

  SUBROUTINE FRPRM( x1a, x2a, ya, y2a, y3a, p, ftol, iter, fret, ierror)
!-----------------------------------------------------------------------
! Given a starting point P that is a vector of length N, Fletcher-Reeves
! Polak-Ribiere minimization is performed on a function FUNV, using its
! gradient as calculated by a routine DFUNC,. The convergence tolerance
! on the function value is input as FTOL. Returned quantities are P
! (the location of the minimum), ITER (the number of iterations
! that were performed), and FRET(the minimum value of the function).
! The routine LINMIN is called to perform line minimizations.
! Maximum anticipated value of N; maximum allowed number of iterations;
! small number to rectify special case of converging to exactly zero 
! function value
!-----------------------------------------------------------------------
  implicit none

  integer, parameter :: itmax = 20
  real,    parameter :: eps   = 1.0E-10

  real,    intent(in)                    :: ftol
  real,    intent(in),    dimension(:)   :: x1a, x2a
  real,    intent(in),    dimension(:,:) :: ya,  y2a, y3a
  real,    intent(inout), dimension(:)   :: p
  real,    intent(out)                   :: fret
  integer, intent(out)                   :: iter
  integer, intent(inout)                 :: ierror

  real, dimension(SIZE(p)) :: g, h, xi

  integer :: l, m, n, j, its, ier
  real    :: p1, px, pxx, py, pyy, fp, s0, pp1, pp2, gg, dgg, gam

!-----------------------------------------------------------------------

  l = SIZE( ya, 1 )
  m = SIZE( ya, 2 )
  n = SIZE( p     )

  CALL SPLIN2( x1a, x2a, ya, y2a, p(1), p(2), p1, px, pxx )
  CALL SPLIN3( x1a, x2a, ya, y3a, p(1), p(2), p1, py, pyy )

  fp    = p1
  xi(1) = px
  xi(2) = py

  do j = 1,n
     g(j) = -xi(j)
     h(j) =   g(j)
    xi(j) =   h(j)
  end do

  do its = 1,itmax
! %%%%%%%%%%%%%%%%%
  iter = its

  s0    = SQRT( xi(1)**2 + xi(2)**2 )
  xi(1) = xi(1) / s0
  xi(2) = xi(2) / s0

  CALL LINMIN( p, xi, fret, x1a, x2a, ya, y2a, ier )

  if( ier == 1 ) then
    ierror = 1
    RETURN
  endif

  if( ( 2.0*ABS( fret - fp ) ) <=   &
      ( ftol*( ABS( fret ) + ABS( fp ) + eps ) ) ) RETURN

                     pp1 =   p(1)
                     pp2 =   p(2)
  if( p(1) <=   0. ) pp1 =   p(1) + 360.0
  if( p(1) >= 360. ) pp1 =   p(1) - 360.0
  if( p(2) <= -90. ) pp2 = -180.0 - p(2)
  if( p(2) >=  90. ) pp2 =  180.0 - p(2)

  CALL SPLIN2( x1a, x2a, ya, y2a, pp1, pp2, p1, px, pxx )
  CALL SPLIN3( x1a, x2a, ya, y3a, pp1, pp2, p1, py, pyy )

  xi(1) = px
  xi(2) = py
  fp    = p1
  gg    = 0.
  dgg   = 0.

  do j = 1,n
    gg  =  gg + g(j)**2
    dgg = dgg + xi(j)**2
    dgg = dgg + ( xi(j) + g(j) ) * xi(j)
  end do

  if( gg == 0.0 ) RETURN

  gam = dgg / gg

  do j = 1,n
     g(j) =- xi(j)
     h(j) =   g(j) + gam*h(j)
    xi(j) =   h(j)
  end do

! %%%%%%%%%%%%%%%%%
  end do

  ierror = 1
  PRINT *, ' *** FRPRM: maximum iterations exceeded'

!-----------------------------------------------------------------------
  end SUBROUTINE FRPRM

!########################################################################

  SUBROUTINE LINMIN( p, xi, fret, x1a, x2a, ya, y2a, ier )
!-----------------------------------------------------------------------
  implicit none

  real, parameter :: tol  = 1.0E-4

  real,    intent(in),    dimension(:)   :: x1a, x2a
  real,    intent(in),    dimension(:,:) :: ya,  y2a
  real,    intent(inout), dimension(:)   :: p,   xi
  real,    intent(out)                   :: fret
  integer, intent(out)                   :: ier

  integer :: l, m, n, j
  real    :: ax, xx, bx, fa, fx, fb, xmin

!-----------------------------------------------------------------------

     l = SIZE( ya, 1 )
     m = SIZE( ya, 2 )
     n = SIZE( p     )
  ncom = n

  do j = 1,n
     pcom(j) =  p(j)
    xicom(j) = xi(j)
  end do

  ax = 0.
  xx = 1.

  CALL MNBRAK( ax, xx, bx, fa, fx, fb, x1a, x2a, ya, y2a )

  ier = 0

  CALL BRENT( ax, xx, bx, tol, fret, xmin, x1a, x2a, ya, y2a, ier )

  if ( xmin <= 1.E-5 ) xmin = 0.0

  do j = 1,n
     xi(j) = xmin * xi(j)
      p(j) = p(j) + xi(j)
  end do

!-----------------------------------------------------------------------
  end SUBROUTINE LINMIN

!########################################################################

  SUBROUTINE LINMIN1( p, xi, fret, x1a, x2a, ya, y2a, ier, val )
!-----------------------------------------------------------------------
  implicit none

  real, parameter :: tol  = 1.0E-4

  real,    intent(in)                    :: val
  real,    intent(in),    dimension(:)   :: x1a, x2a, xi
  real,    intent(in),    dimension(:,:) :: ya,  y2a
  real,    intent(inout), dimension(:)   :: p
  real,    intent(out)                   :: fret
  integer, intent(out)                   :: ier

  real, dimension(SIZE(p)) :: xt

  integer :: l, m, n, j
  real    :: x20, y20, x, d, y, py, pyy, y0, y1

!-----------------------------------------------------------------------

     l = SIZE( ya, 1 )
     m = SIZE( ya, 2 )
     n = SIZE( p     )
  ncom = n

  x20 = P(1)
  y20 = p(2)

  do j = 1,n
     pcom(j) =  p(j)
    xicom(j) = xi(j)
  end do

  ier = 0
  x  = 1.
  d  = 0.

  CALL SPLIN2( x1a, x2a, ya, y2a, p(1), p(2), y, py, pyy )

  y0 = y

! %%%%%%%%%%%%%%%%%
  1 CONTINUE
! %%%%%%%%%%%%%%%%%

  do j = 1,n
    xt(j) = p(j) + x * xicom(j)
  end do

  d = SQRT( ( xt(1) - x20 )**2 + ( xt(2) - y20 )**2 )

  CALL SPLIN2( x1a, x2a, ya, y2a, xt(1), xt(2), y1, py, pyy )

  if( ABS( d ) >= 8.0 ) then
    ier = 1
    RETURN
  endif

  if( x <= 0.0001 ) then
    fret = y
    RETURN
  endif

  if( ( y0 - y1 ) >= val ) RETURN

  if( ( ( y0 - y1 ) <= 2.0 ) .and. ( y <= 1000000. ) ) then
    p(1) = xt(1)
    p(2) = xt(2)
! %%%%%%%%%%%%%%%%%
  GOTO 1
! %%%%%%%%%%%%%%%%%
  endif

  if(y1 >= 10000000 ) then
    p(1) = xt(1)
    p(2) = xt(2)
  endif

  x = x / 2.0

! %%%%%%%%%%%%%%%%%
  GOTO 1
! %%%%%%%%%%%%%%%%%

!-----------------------------------------------------------------------
  end SUBROUTINE LINMIN1

!########################################################################

  SUBROUTINE F1D1M( y, x, x1a, x2a, ya, y2a )
!-----------------------------------------------------------------------
  implicit none

  real,    intent(in)                    :: x
  real,    intent(in),    dimension(:)   :: x1a, x2a
  real,    intent(in),    dimension(:,:) :: ya,  y2a
  real,    intent(out)                   :: y

  real, dimension(nmax) :: xt

  integer :: l, m, j
  real    :: py, pyy

!-----------------------------------------------------------------------

  l = SIZE( ya, 1 )
  m = SIZE( ya, 2 )

  do j = 1,ncom
    xt(j) = pcom(j) + x * xicom(j)
  end do

  if( xt(1) <=   0.0 ) xt(1) =  360.0 + xt(1)
  if( xt(1) >= 360.0 ) xt(1) =  xt(1) - 360.0
  if( xt(2) <= -90.0 ) xt(2) = -180.0 - xt(2)
  if( xt(2) >=  90.0 ) xt(2) =  180.0 - xt(2)

  CALL SPLIN2( x1a, x2a, ya, y2a, xt(1), xt(2), y, py, pyy )

!-----------------------------------------------------------------------
  end SUBROUTINE F1D1M

!########################################################################

  SUBROUTINE MNBRAK( ax, bx, cx, fa, fb, fc, x1a, x2a, ya, y2a )
!-----------------------------------------------------------------------
  implicit none

  real, parameter :: gold   = 1.618034
  real, parameter :: glimit = 100.0
  real, parameter :: tiny   = 1.0E-20

  real, intent(in), dimension(:)   :: x1a, x2a
  real, intent(in), dimension(:,:) :: ya,  y2a
  real, intent(inout)              :: ax, bx, cx
  real, intent(out)                :: fa, fb, fc

  integer :: l, m
  real    :: dum, r, q, u, ulim, fu
  real    :: uxt, uxb

!-----------------------------------------------------------------------

  l = SIZE( ya, 1 )
  m = SIZE( ya, 2 )

  CALL F1D1M( fa, ax, x1a, x2a, ya, y2a )
  CALL F1D1M( fb, bx, x1a, x2a, ya, y2a )

  if( fb > fa ) then
     dum = ax
     ax  = bx
     bx  = dum
     dum = fb
     fb  = fa
     fa  = dum
  endif

  cx = bx + gold * ( bx - ax )
  CALL F1D1M( fc, cx, x1a, x2a, ya, y2a )

! %%%%%%%%%%%%%%%%%%%%%%
  1 if( fb >= fc ) then
! %%%%%%%%%%%%%%%%%%%%%%

  r    = ( bx - ax ) * ( fb - fc )
  q    = ( bx - cx ) * ( fb - fa )

! u    = bx - ( ( bx - cx ) * q - ( bx - ax ) * r ) /  &
!        ( 2.0 * SIGN( MAX( ABS( q - r ), tiny ), q - r ) )

  uxt  = ( bx - cx ) * q - ( bx - ax ) * r 
  uxb  =  2.0 * SIGN( MAX( ABS( q - r ), tiny ), q - r )
  u    = bx - uxt / uxb
  u    = MIN( cx + gold * ( cx - bx ), u )
  ulim = bx + glimit * ( cx - bx )

! ---------------------------------------------------
  if( ( ( bx - u ) * ( u - cx ) ) > 0.0 ) then
! ---------------------------------------------------
     CALL F1D1M( fu, u, x1a, x2a, ya, y2a )
  if( fu < fc ) then
     ax = bx
     fa = fb
     bx = u
     fb = fu
     RETURN
  else if( fu > fb ) then
     cx = u
     fc = fu
     RETURN
  endif
     u = cx + gold * ( cx - bx )
     CALL F1D1M( fu, u, x1a, x2a, ya, y2a )
! ---------------------------------------------------------
  else if( ( ( cx - u ) * ( u - ulim ) ) > 0.0 ) then
! ---------------------------------------------------------
     CALL F1D1M( fu, u, x1a, x2a, ya, y2a )
  if( fu < fc ) then
     bx = cx
     cx = u
     u  = cx + gold * ( cx - bx )
     fb = fc
     fc = fu
     CALL F1D1M( fu, u, x1a, x2a, ya, y2a )
  endif  
! ---------------------------------------------------------
  else if( ( ( u - ulim ) * ( ulim - cx ) ) >= 0.0 ) then
! ---------------------------------------------------------
     u = ulim
     CALL F1D1M( fu, u, x1a, x2a, ya, y2a )
! ---------------------------------------------------------
  else
! ---------------------------------------------------------
     u = cx + gold * ( cx - bx )
     CALL F1D1M( fu, u, x1a, x2a, ya, y2a )
! ---------------------------------------------------------
  endif
! ---------------------------------------------------------

  ax = bx
  bx = cx
  cx = u
  fa = fb
  fb = fc
  fc = fu

! %%%%%%%%%%%%%%%%%%%%%%
  GOTO 1
  endif
! %%%%%%%%%%%%%%%%%%%%%%

!-----------------------------------------------------------------------
  end SUBROUTINE MNBRAK

!########################################################################

  SUBROUTINE BRENT( ax, bx, cx, tol, fret, xmin, x1a, x2a, ya, y2a, ier)
!-----------------------------------------------------------------------
  implicit none

  real,    parameter :: cgold = 0.3819660
  real,    parameter :: zeps  = 1.0E-10
  integer, parameter :: itmax = 100

  real,    intent(in)                 :: ax,  bx, cx, tol
  real,    intent(in), dimension(:)   :: x1a, x2a
  real,    intent(in), dimension(:,:) :: ya,  y2a
  integer, intent(inout)              :: ier
  real,    intent(out)                :: fret, xmin

  integer :: l, m, iter
  real    :: a, b, v, w, x, e, fx, fv, fw, d, u, fu
  real    :: xm, tol1, tol2, r, q, p, etemp

!-----------------------------------------------------------------------

  l = SIZE( ya, 1 )
  m = SIZE( ya, 2 )

  a = MIN( ax, cx )
  b = MAX( ax, cx )
  v = bx
  w = v
  x = v
  e = 0.
  CALL F1D1M( fx, x, x1a, x2a, ya, y2a )
  fv = fx
  fw = fx

! %%%%%%%%%%%%%%%%%%%%%%
  do iter = 1,itmax
! %%%%%%%%%%%%%%%%%%%%%%

  xm   = 0.5 * ( a + b )
  tol1 = tol * ABS( x ) + zeps
  tol2 = 2.0 * tol1

  if( ABS( x - xm ) <= ( tol2 - 0.5 * ( b - a ) ) ) GOTO 3

! ---------------------------------------------------------
  if( ABS( e ) > tol1 ) then
! ---------------------------------------------------------
                  r = ( x - w ) * ( fx - fv )
                  q = ( x - v ) * ( fx - fw )
                  p = ( x - v ) * q - ( x - w ) * r
                  q = 2.0 * ( q - r )
    if( q > 0.0 ) p = -p
                  q = ABS( q )
              etemp = e
                  e = d

    if( ( ABS( p ) >= ABS( 0.5 * q * etemp ) ) .or.  &
        (      p   <= ( q * ( a - x ) )      ) .or.  &
        (      p   >= ( q * ( b - x ) )      )  ) GOTO 1

    d = p / q
    u = x + d

    if( ( ( u - a ) < tol2 )  .or. &
        ( ( b - u ) < tol2 ) ) d = SIGN( tol1, xm - x )
    GOTO 2

! ---------------------------------------------------------
  endif
! ---------------------------------------------------------

 1 CONTINUE
    if( x >= xm ) then
        e = a - x
    else
        e = b - x
    endif
        d = cgold * e

 2 CONTINUE
    if (abs(d) >= tol1) then
        u = x + d
    else
        u = x + SIGN( tol1, d )
    endif
        CALL F1D1M( fu, u, x1a, x2a, ya, y2a )

! ---------------------------------------------------------
  if( fu <= fx ) then
! ---------------------------------------------------------
    if( u >= x ) then
        a = x
    else
        b = x
    endif
         v =  w
        fv = fw
         w =  x
        fw = fx
         x =  u
        fx = fu
! ---------------------------------------------------------
  else
! ---------------------------------------------------------
    if( u < x ) then
        a = u
    else
        b = u
    endif
    if( ( fu <= fw ) .or. ( w == x ) ) then
         v =  w
        fv = fw
         w =  u
        fw = fu
    else if( ( fu <= fv ) .or. ( v == x ) .or. ( v == w ) ) then
         v =  u
        fv = fu
    endif
! ---------------------------------------------------------
  endif
! ---------------------------------------------------------

! %%%%%%%%%%%%%%%%%%%%%%
  end do
! %%%%%%%%%%%%%%%%%%%%%%

  ier = 1
  PRINT *, ' *** Brent: exceed maximum iterations'

 3 CONTINUE
   xmin = x
   fret = fx

!-----------------------------------------------------------------------
  end SUBROUTINE BRENT

!########################################################################

  SUBROUTINE SHAPE( x1a,  x2a,  ya,   y2a,     y3a, p, &
                    ftol, iter, fret, ierror2, val     )
!-----------------------------------------------------------------------
! Given a starting point P that is a vector of length N, Fletcher-Reeves
! Polak-Ribiere minimization is performed on a function FUNV, using its
! gradient as calculated by a routine DFUNC,. The convergence tolerance
! on the function value is input as FTOL. Returned quantities are P
! (the location of the minimum), ITER (the number of iterations
! that were performed), and FRET(the minimum value of the function).
! The routine LINMIN1 is called to perform line minimizations.
! Maximum anticipated value of N; maximum allowed number of iterations;
! small number to rectify special case of converging to exactly zero 
! function value.
!-----------------------------------------------------------------------
  implicit none

  real,    intent(in),  dimension(:)   :: x1a, x2a, p 
  real,    intent(in),  dimension(:,:) :: ya,  y2a, y3a    
  real,    intent(in)                  :: val         
  integer, intent(inout)               :: ierror2         

  real,    intent(in)  :: ftol         ! ===> NOT REFERENCED
  integer, intent(out) :: iter         ! ===> NOT REFERENCED
  real,    intent(out) :: fret         ! ===> NOT REFERENCED

  real, dimension(SIZE(p))               :: q,  g,   h, xi
  real, dimension(SIZE(ya,1),SIZE(ya,2)) :: ta, t2a, t3a

  integer :: l,  m, n, i, ier
  real    :: dr, fret1

 real, dimension(8) :: xi1 = (/ 0.,  0., 1., -1.,  1., 1., -1., -1. /)
 real, dimension(8) :: xi2 = (/ 1., -1., 0.,  0., -1., 1.,  1., -1. /)

!-----------------------------------------------------------------------

  l = SIZE( ya, 1 )
  m = SIZE( ya, 2 )
  n = SIZE( p     )

   ta(:,:) =  -ya(:,:)
  t2a(:,:) = -y2a(:,:)
  t3a(:,:) = -y3a(:,:)

  do i = 1,8
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if( ierror2 == 0 ) then
         q(1) =   p(1)
         q(2) =   p(2)
        xi(1) = xi1(i)
        xi(2) = xi2(i)
  CALL LINMIN1( q, xi, fret1, x1a, x2a, ta, t2a, ier, val )
        dr = ( p(1) - q(1) )**2 + ( p(2) - q(2) )**2 
    if( dr >= 100.0 ) ierror2 = 1
    if( ier == 1    ) ierror2 = 1
  end if
! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  end do 

!-----------------------------------------------------------------------
  end SUBROUTINE SHAPE

!########################################################################
 end MODULE SPLINE_MOD
