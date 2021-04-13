#ifndef teca_calendar_h
#define teca_calendar_h

namespace teca_calendar_util
{

/** @anchor Gregorian calendar
 * @name Gregorian calendar
 * functions for date computations in gregorian calendar.  to use convert the
 * origin to a gergorian_number do the calculation and convert the number back
 * into a date useing date_from_gregorian_number. for details about the math
 * and an explanation of the errors see
 * http://alcor.concordia.ca/~gpkatch/gdate-algorithm.html
 */
///@{
/** return a date number for the given date that can be used in computations.
 * input:
 *
 * > y : 4 digit year
 * > m : 2 digit month
 * > d : 2 digit day
 *
 */
long gregorian_number(long y, long m, long d);

/** input:
 *
 * > g : date number computed from gregorian_number
 *
 * returns:
 *
 * > y : 4 digit year
 * > m : 2 digit month
 * > d : 2 digit day
 *
 */
void date_from_gregorian_number(long g, long &y, long &m, long &d);

/**
 * input:
 *
 * > y : 4 digit year
 * > m : 2 digit month
 * > d : 2 digit day
 *
 * returns:
 *
 * true if the date is valid in the gregorian calendar and our conversion
 * algorithm.
*/
bool valid_gregorian_date(long y, long m, long d);
///@}

}

#endif
