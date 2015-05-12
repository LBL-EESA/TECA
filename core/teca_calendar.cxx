#include "teca_calendar.h"

// --------------------------------------------------------------------------
long gregorian_number(long y, long m, long d)
{
    m = (m + 9) % 12;
    y = y - m/10;
    return 365*y + y/4 - y/100 + y/400 + (m*306 + 5)/10 + (d - 1);
}

// --------------------------------------------------------------------------
void date_from_gregorian_number(long g, long &y, long &m, long &d)
{
    y = (10000*g + 14780)/3652425;
    long ddd = g - (365*y + y/4 - y/100 + y/400);
    if (ddd < 0)
    {
        y = y - 1;
        ddd = g - (365*y + y/4 - y/100 + y/400);
    }

    long mi = (100*ddd + 52)/3060;

    m = (mi + 2)%12 + 1;
    y = y + (mi + 2)/12;
    d = ddd - (mi*306 + 5)/10 + 1;
}

// --------------------------------------------------------------------------
bool valid_gregorian_date(long y, long m, long d)
{
    long g = gregorian_number(y,m,d);
    if (g < 578027) // 578027 = gergorian_number(1582,10,1);
        return false;

    long yy, mm, dd;
    date_from_gregorian_number(g, yy, mm, dd);

    if ((y != yy) || (m != mm) || (d != dd))
        return false;

    return true;
}
