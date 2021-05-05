/*
A threadsafe port of the calcalcs library
Burlen Loring Thu Apr 22 06:22:16 PM PDT 2021

The CalCalcs routines, a set of C-language routines to perform
calendar calculations.

Version 1.0, released 7 January 2010

Copyright (C) 2010 David W. Pierce, dpierce@ucsd.edu

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <ctype.h>

/* TODO
 * for now we won';t use this in a threaded algorithm
 * if we ever need to we can start with this simple
 * locking but we should actually fix the issues  */
#if defined(CALCALCS_THREAD)
#include <mutex>
#endif

#include "udunits2.h"
#include "teca_calcalcs.h"

namespace teca_calcalcs
{

static int c_isleap_gregorian    ( int year, int *leap );
static int c_isleap_gregorian_y0( int year, int *leap );
static int c_isleap_julian       ( int year, int *leap );
static int c_isleap_never        ( int year, int *leap );

static int c_date2jday_julian      ( int year, int month, int day, int *jday );
static int c_date2jday_gregorian   ( int year, int month, int day, int *jday );
static int c_date2jday_gregorian_y0( int year, int month, int day, int *jday );
static int c_date2jday_noleap      ( int year, int month, int day, int *jday );
static int c_date2jday_360_day     ( int year, int month, int day, int *jday );

static int c_jday2date_julian      ( int jday, int *year, int *month, int *day );
static int c_jday2date_gregorian   ( int jday, int *year, int *month, int *day );
static int c_jday2date_gregorian_y0( int jday, int *year, int *month, int *day );
static int c_jday2date_noleap      ( int jday, int *year, int *month, int *day );
static int c_jday2date_360_day     ( int jday, int *year, int *month, int *day );

static int c_dpm_julian      ( int year, int month, int *dpm );
static int c_dpm_gregorian   ( int year, int month, int *dpm );
static int c_dpm_gregorian_y0( int year, int month, int *dpm );
static int c_dpm_noleap      ( int year, int month, int *dpm );
static int c_dpm_360_day     ( int year, int month, int *dpm );

#define CCS_ERROR_MESSAGE_LEN    8192
static char error_message[CCS_ERROR_MESSAGE_LEN];

/* Following are number of Days Per Month (dpm).  They are called 'idx1' to
 * emphasize that they are intended to be index by a month starting at 1
 * rather than at 0.
 *                     na, jan feb mar apr may jun jul aug sep oct nov dec */
static int dpm_idx1[]      = {-99, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
static int dpm_leap_idx1[] = {-99, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};

/* Same as above, but SUM of previous months.  Indexing starts at 1 for January */
static int spm_idx1[]      = {-99, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365};
/* static int spm_leap_idx1[] = {-99, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366}; note: spm is only used for calendars with no leap years */

static int date_ge( int year, int month, int day, int y2, int m2, int d2 );
static int date_le( int year, int month, int day, int y2, int m2, int d2 );
static int date_lt( int year, int month, int day, int y2, int m2, int d2 );
static int date_gt( int year, int month, int day, int y2, int m2, int d2 );
static int set_xition_extra_info( calcalcs_cal *cal );
static void ccs_dump_xition_dates( void );
static void ccs_gxd_add_country(const char *code, const char *longname, int year, int month, int day );
static void ccs_init_country_database( void );

/* Some arbitrary number that is unlikely to be encounterd in a string of random digits */
#define CCS_VALID_SIG    89132412

/* These implement the database that associates a two-letter country code, such as "UK",
 * with the transition date that the country switched from the Julian to Gregorian calendar
 */
#define CCS_MAX_N_COUNTRY_CODES    5000
static int ccs_n_country_codes = 0;
static ccs_country_code *ccs_xition_dates[ CCS_MAX_N_COUNTRY_CODES ];
static int have_initted_country_codes = 0;

/**********************************************************************************************
 * Initialize a calendar.  The passed argument is the name of the calendar, and may be
 * one of the following character strings:
 *    "standard"
 *    "proleptic_Julian"
 *    "proleptic_Gregorian"
 *    "noleap" (aka "365_day" and "no_leap")
 *    "360_day"
 *
 * As a special hack, a calendar can be named "standard_XX" where XX is a two-letter
 * date code recognized by ccs_get_xition_date, in which case a standard calendar with
 * the specified transition date will be used.
 *
 * Returns a pointer to the new calendar, or NULL on error.
 */
calcalcs_cal *ccs_init_calendar( const char *calname )
{
    calcalcs_cal    *retval;
    int        use_specified_xition_date, spec_year_x, spec_month_x, spec_day_x;

    error_message[0] = '\0';

    if( strncasecmp( calname, "standard", 8 ) == 0 ) {

        if( ! have_initted_country_codes )
            ccs_init_country_database();

        /* See if this is a name of the form "Standard_XX" */
        use_specified_xition_date = 0;
        if( (strlen(calname) >= 11) && (calname[8] == '_')) {
            if( ccs_get_xition_date( calname+9, &spec_year_x, &spec_month_x, &spec_day_x ) != 0 ) {
                fprintf( stderr, "Error, unknown calendar passed to ccs_init_calendar: \"%s\". Returning NULL\n",
                    calname );
                return(NULL);
                }
            use_specified_xition_date = 1;
            }

        retval = (calcalcs_cal *)malloc( sizeof(calcalcs_cal) );
        if( retval == NULL ) {
            fprintf( stderr, "Error, cannot allocate space for the calcalcs calendar. Returning NULL\n" );
            return( NULL );
            }
        retval->sig  = CCS_VALID_SIG;
        retval->name = (char *)malloc( sizeof(char) * (strlen(calname)+1) );
        strcpy( retval->name, calname );

        retval->mixed = 1;
        retval->early_cal = ccs_init_calendar( "proleptic_julian" );
        retval->late_cal  = ccs_init_calendar( "proleptic_gregorian" );

        /* Following are FIRST DAY the "later" calendar should be used */
        if( use_specified_xition_date == 1 ) {
            retval->year_x    = spec_year_x;
            retval->month_x   = spec_month_x;
            retval->day_x     = spec_day_x;
            }
        else
            {
            retval->year_x    = 1582;
            retval->month_x   = 10;
            retval->day_x     = 15;
            }

        /* Set the last date the earlier cal was used, and the transition day's Julian date */
        if( set_xition_extra_info( retval ) != 0 ) {
            fprintf( stderr, "calcalcs_init_cal: Error trying to initialize calendar \"%s\": %s. Returning NULL\n",
                calname, error_message );
            return(NULL);
            }
        }

    else if( (strcasecmp( calname, "gregorian" ) == 0) ||
         (strcasecmp( calname, "proleptic_gregorian" ) == 0)) {

        /* This is a "regular" Gregorian calendar, which does not include "year 0".
         * See also calendar gregorian_y0, which does include a year 0, below
         */
        retval = (calcalcs_cal *)malloc( sizeof(calcalcs_cal) );
        if( retval == NULL ) {
            fprintf( stderr, "Error, cannot allocate space for the calcalcs calendar\n" );
            return( NULL );
            }
        retval->sig  = CCS_VALID_SIG;
        retval->name = (char *)malloc( sizeof(char) * (strlen(calname)+1) );
        strcpy( retval->name, calname );
        retval->ndays_reg  = 365;
        retval->ndays_leap = 366;

        retval->mixed = 0;

        retval->c_isleap    = &c_isleap_gregorian;
        retval->c_date2jday = &c_date2jday_gregorian;
        retval->c_jday2date = &c_jday2date_gregorian;
        retval->c_dpm       = &c_dpm_gregorian;
        }

    else if( (strcasecmp( calname, "gregorian_y0" ) == 0) ||
         (strcasecmp( calname, "proleptic_gregorian_y0" ) == 0)) {

        /* This is a Gregorian calendar that includes "year 0".
         */
        retval = (calcalcs_cal *)malloc( sizeof(calcalcs_cal) );
        if( retval == NULL ) {
            fprintf( stderr, "Error, cannot allocate space for the calcalcs calendar\n" );
            return( NULL );
            }
        retval->sig  = CCS_VALID_SIG;
        retval->name = (char *)malloc( sizeof(char) * (strlen(calname)+1) );
        strcpy( retval->name, calname );
        retval->ndays_reg  = 365;
        retval->ndays_leap = 366;

        retval->mixed = 0;

        retval->c_isleap    = &c_isleap_gregorian_y0;
        retval->c_date2jday = &c_date2jday_gregorian_y0;
        retval->c_jday2date = &c_jday2date_gregorian_y0;
        retval->c_dpm       = &c_dpm_gregorian_y0;
        }

    else if( (strcasecmp( calname, "julian" ) == 0 ) ||
            (strcasecmp( calname, "proleptic_julian" ) == 0 )) {
        retval = (calcalcs_cal *)malloc( sizeof(calcalcs_cal) );
        if( retval == NULL ) {
            fprintf( stderr, "Error, cannot allocate space for the calcalcs calendar\n" );
            return( NULL );
            }
        retval->sig  = CCS_VALID_SIG;
        retval->name = (char *)malloc( sizeof(char) * (strlen(calname)+1) );
        strcpy( retval->name, calname );
        retval->ndays_reg  = 365;
        retval->ndays_leap = 366;

        retval->mixed = 0;

        retval->c_isleap    = &c_isleap_julian;
        retval->c_date2jday = &c_date2jday_julian;
        retval->c_jday2date = &c_jday2date_julian;
        retval->c_dpm       = &c_dpm_julian;
        }

    else if( (strcasecmp(calname,"noleap")==0) ||
             (strcasecmp(calname,"no_leap")==0) ||
         (strcasecmp(calname,"365_day")==0)) {
        retval = (calcalcs_cal *)malloc( sizeof(calcalcs_cal) );
        if( retval == NULL ) {
            fprintf( stderr, "Error, cannot allocate space for the calcalcs calendar\n" );
            return( NULL );
            }
        retval->sig  = CCS_VALID_SIG;
        retval->name = (char *)malloc( sizeof(char) * (strlen("noleap")+1) );
        strcpy( retval->name, "noleap" );
        retval->ndays_reg  = 365;
        retval->ndays_leap = 365;

        retval->mixed  = 0;

        retval->c_isleap    = &c_isleap_never;
        retval->c_date2jday = &c_date2jday_noleap;
        retval->c_jday2date = &c_jday2date_noleap;
        retval->c_dpm       = &c_dpm_noleap;
        }

    else if( strcasecmp(calname,"360_day")==0) {
        retval = (calcalcs_cal *)malloc( sizeof(calcalcs_cal) );
        if( retval == NULL ) {
            fprintf( stderr, "Error, cannot allocate space for the calcalcs calendar\n" );
            return( NULL );
            }
        retval->sig  = CCS_VALID_SIG;
        retval->name = (char *)malloc( sizeof(char) * (strlen(calname)+1) );
        strcpy( retval->name, calname );
        retval->ndays_reg  = 360;
        retval->ndays_leap = 360;

        retval->mixed  = 0;

        retval->c_isleap    = &c_isleap_never;
        retval->c_date2jday = &c_date2jday_360_day;
        retval->c_jday2date = &c_jday2date_360_day;
        retval->c_dpm       = &c_dpm_360_day;
        }

    else
        return( NULL );

    return( retval );
}

/**********************************************************************************************
 *
 * Determine if the passed year is a leap year in the specified calendar.
 * The passed parameter leap is set to '1' if the year is a leap year, and '0' if it is not.
 *
 * Returns 0 on success, and a negative value on error.
 * Errors include the passed year being invalid (before 4713 B.C.) or not existing
 * in the specified calendar (i.e., there is no year 0 in either the Gregorian or
 * Julian calendars).
 *
 */
int ccs_isleap( calcalcs_cal *calendar, int year, int *leap )
{
    int    ierr;
    calcalcs_cal *c2use;

    if( calendar == NULL ) return(CALCALCS_ERR_NULL_CALENDAR);
    if( calendar->sig != CCS_VALID_SIG ) return(CALCALCS_ERR_INVALID_CALENDAR);

    if( year < -4714 ) {
        sprintf( error_message, "ccs_isleap: year %d is out of range for the %s calendar; dates must not be before 4713 B.C.", year, calendar->name );
        return( CALCALCS_ERR_OUT_OF_RANGE );
        }

    if( calendar->mixed ) {
        if( year >= calendar->year_x )    /* Q: did any countries transition during a year that had different leap status before and after??? Let's hope not! */
            c2use = calendar->late_cal;
        else
            c2use = calendar->early_cal;
        }
    else
        c2use = calendar;

    ierr = c2use->c_isleap( year, leap );
    return( ierr );
}

/**********************************************************************************************
 * ccs_dpm: returns the number of days per month for the passed year/month/calendar combo
 *
 * Returns 0 on success, and a negative number on error (for example, an illegal month number)
 */
int ccs_dpm( calcalcs_cal *calendar, int year, int month, int *dpm )
{
    int        ndays_reg, ierr;
    int        overlap_px_month, overlap_x_month;
    calcalcs_cal    *c2use;

    if( calendar->mixed ) {
        /* A calendar transition potentially affects two months -- the month containing the
         * last day of the old calendar, and the month containing the first day of the
         * new calendar.  If we are in either of those months, things get much harder.
         * (Note that these can easily be the same month)
         */
        overlap_px_month = ((year == calendar->year_px) && (month == calendar->month_px));
        overlap_x_month  = ((year == calendar->year_x ) && (month == calendar->month_x ));
        if( overlap_px_month || overlap_x_month ) {
            if( overlap_px_month && (!overlap_x_month)) {
                /* Last day of the month must have been last day the early calendar was used */
                *dpm = calendar->day_px;
                return(0);
                }
            else if( overlap_x_month && (!overlap_px_month)) {
                if( (ierr = ccs_dpm( calendar->late_cal, year, month, &ndays_reg )) != 0 )
                    return( ierr );
                *dpm = ndays_reg - calendar->day_x + 1;
                return(0);
                }

            else    /* overlap_px_month && overlap_x_month */
                {
                if( (ierr = ccs_dpm( calendar->late_cal, year, month, &ndays_reg )) != 0 )
                    return( ierr );
                *dpm = calendar->day_px + (ndays_reg - calendar->day_x + 1);
                return(0);
                }
            }
        else if( date_ge( year, month, 1, calendar->year_x, calendar->month_x, calendar->day_x ))
            c2use = calendar->late_cal;
        else
            c2use = calendar->early_cal;
        }
    else
        c2use = calendar;

    return( c2use->c_dpm( year, month, dpm ));
}

/**********************************************************************************************
 * ccs_jday2date: give a Julian day number, return the corresponding date in the
 *           selected calendar
 *
 * Returns 0 on success, <0 on error and fills string error_message
 */
int ccs_jday2date( calcalcs_cal *calendar, int jday, int *year, int *month, int *day )
{
    calcalcs_cal *c2use;

    if( calendar == NULL ) return(CALCALCS_ERR_NULL_CALENDAR);
    if( calendar->sig != CCS_VALID_SIG ) return(CALCALCS_ERR_INVALID_CALENDAR);

    if( calendar->mixed ) {
        if( jday >= calendar->jday_x )
            c2use = calendar->late_cal;
        else
            c2use = calendar->early_cal;
        }
    else
        c2use = calendar;

    return( c2use->c_jday2date( jday, year, month, day ));
}

/**********************************************************************************************
 * ccs_date2jday: given a date, return the (true) Julian day number
 *
 * Note that "Julian day number" is not the day number of the year, but rather the
 * day number starting on Jan 1st 4713 BC (in the proleptic Julian calendar) and
 * counting consecutively.
 *
 * Returns 0 on success, <0 on error and fills string error_message
 */
int ccs_date2jday( calcalcs_cal *calendar, int year, int month, int day, int *jday )
{
    int        dpm, ierr;
    calcalcs_cal    *c2use;

    if( calendar == NULL ) return(CALCALCS_ERR_NULL_CALENDAR);
    if( calendar->sig != CCS_VALID_SIG ) return(CALCALCS_ERR_INVALID_CALENDAR);

    if( calendar->mixed ) {
        if( date_ge( year, month, day, calendar->year_x, calendar->month_x, calendar->day_x ))
            c2use = calendar->late_cal;
        else if( date_le( year, month, day, calendar->year_px, calendar->month_px, calendar->day_px ))
            c2use = calendar->early_cal;
        else
            {
            sprintf( error_message, "ccs_date2jday: date %04d-%02d-%02d is not a valid date in the %s calendar; it falls between the last date the %s calendar was used (%04d-%02d-%02d) and the first date the %s calendar was used (%04d-%02d-%02d)",
                year, month, day, calendar->name,
                calendar->early_cal->name,
                calendar->year_px, calendar->month_px, calendar->day_px,
                calendar->late_cal->name,
                calendar->year_x, calendar->month_x, calendar->day_x );
            return( CALCALCS_ERR_DATE_NOT_IN_CALENDAR );
            }
        }
    else
        c2use = calendar;

    if( (ierr = ccs_dpm( c2use, year, month, &dpm )) != 0 )
        return( ierr );

    if( (month < 1) || (month > 12) || (day < 1) || (day > dpm)) {
        sprintf( error_message, "date2jday passed an date that is invalid in the %s calendar: %04d-%02d-%02d",
            c2use->name, year, month, day );
        return( CALCALCS_ERR_DATE_NOT_IN_CALENDAR );
        }

    return( c2use->c_date2jday( year, month, day, jday ));
}

/********************************************************************************************
 *
 * ccs_date2doy: given a Y/M/D date, calculates the day number of the year, starting at 1 for
 * January 1st.
 *
 * Returns 0 on success, and a negative value on error (for example, an illegal date [one
 * that does not exist in the specified calendar])
 */
int ccs_date2doy( calcalcs_cal *calendar, int year, int month, int day, int *doy )
{
    int    ierr, jd0, jd1, doy_px, jd_args, xition_date_first_day_of_year,
        ndays_elapsed;
    calcalcs_cal    *c2use;

    if( calendar == NULL ) return(CALCALCS_ERR_NULL_CALENDAR);
    if( calendar->sig != CCS_VALID_SIG ) return(CALCALCS_ERR_INVALID_CALENDAR);

    if( calendar->mixed ) {

        /* If we fall in the twilight zone after the old calendar was stopped but before
         * the new calendar was used, it's an error
         */
        if( date_gt( year, month, day, calendar->year_px, calendar->month_px, calendar->day_px ) &&
            date_lt( year, month, day, calendar->year_x,  calendar->month_x,  calendar->day_x )) {
            sprintf( error_message, "ccs_date2doy: date %04d-%02d-%02d is not a valid date in the %s calendar; it falls between the last date the %s calendar was used (%04d-%02d-%02d) and the first date the %s calendar was used (%04d-%02d-%02d)",
                year, month, day, calendar->name,
                calendar->early_cal->name,
                calendar->year_px, calendar->month_px, calendar->day_px,
                calendar->late_cal->name,
                calendar->year_x, calendar->month_x, calendar->day_x );
            return( CALCALCS_ERR_DATE_NOT_IN_CALENDAR );
            }

        xition_date_first_day_of_year = ((year == calendar->year_x) && (calendar->month_x == 1) && (calendar->day_x == 1));
        if( (year > calendar->year_x) || xition_date_first_day_of_year )
            c2use = calendar->late_cal;
        else if( date_le( year, month, day, calendar->year_px, calendar->month_px, calendar->day_px ))
            c2use = calendar->early_cal;
        else
            {
            /* Complicated if we are asking for the day of the year during
             * the transition year and after the transition date.  I'm choosing
             * to define the day numbering during a transition year as
             * consecutive, which means that the doy for dates
             * after the transition date equals the doy of the last
             * day of the earlier calendar plus the number of days
             * that have elapsed since the transition day.
             */

            /* Get the doy of the day BEFORE the transition day
             * in the earlier calendar
            */
            if( (ierr = ccs_date2doy( calendar->early_cal, calendar->year_px, calendar->month_px, calendar->day_px, &doy_px )) != 0 )
                return( ierr );

            /* Get number of days that have elapsed between the transition day
             * and the requested date
             */
            if( (ierr = ccs_date2jday( calendar->late_cal, year, month, day, &jd_args )) != 0 )
                return( ierr );
            ndays_elapsed = jd_args - calendar->jday_x + 1;        /* if this IS the transition day, ndays_elapsed==1 */

            /* Finally, the day of the year is the day number of the day BEFORE the
             * transition day plus the number of elapsed days since the
             * transition day.
             */
            *doy = doy_px + ndays_elapsed;

            return(0);
            }
        }
    else
        c2use = calendar;

    /* Get Julian day number of Jan 1st of the specified year */
    if( (ierr = c2use->c_date2jday( year, 1, 1, &jd0 )) != 0 )
        return( ierr );

    /* Get Julian day number of the specified date */
    if( (ierr = c2use->c_date2jday( year, month, day, &jd1 )) != 0 )
        return( ierr );

    *doy = jd1 - jd0 + 1;    /* Add 1 because numbering starts at 1 */

    return(0);
}

/********************************************************************************************
 *
 * ccs_doy2date: given a year and a day number in that year (with counting starting at 1 for
 *     Jan 1st), this returns the month and day of the month that the doy refers to.
 *
 * Returns 0 on success, and a negative value on error (for example, a day of the year
 * that is less than 1 or greater than 366).
 */
int ccs_doy2date( calcalcs_cal *calendar, int year, int doy, int *month, int *day )
{
    int    ierr, leap, jd0, jd1, tyear, doy_px, jd_want,
        xition_date_first_day_of_year, ndays_max;
    calcalcs_cal    *c2use;

    if( calendar == NULL ) return(CALCALCS_ERR_NULL_CALENDAR);
    if( calendar->sig != CCS_VALID_SIG ) return(CALCALCS_ERR_INVALID_CALENDAR);

    if( calendar->mixed ) {
        xition_date_first_day_of_year = ((year == calendar->year_x) && (calendar->month_x == 1) && (calendar->day_x == 1));
        if( year < calendar->year_x )
            c2use = calendar->early_cal;
        else if( (year > calendar->year_x) || xition_date_first_day_of_year )
            c2use = calendar->late_cal;
        else
            {
            /* Get the doy of the day BEFORE the transition day
             * in the earlier calendar
            */
            if( (ierr = ccs_date2doy( calendar->early_cal, calendar->year_px, calendar->month_px, calendar->day_px, &doy_px )) != 0 )
                return( ierr );

            /* If our requested doy is before the transition doy, we
             * can just easily calculate it with the early calendar
             */
            if( doy <= doy_px )
                return( ccs_doy2date( calendar->early_cal, year, doy, month, day ));

            /* Finally calculate the Julian day we want, and convert it to a date */
            jd_want = calendar->jday_x + (doy - doy_px - 1);
            if( (ierr = ccs_jday2date( calendar->late_cal, jd_want, &tyear, month, day)) != 0 )
                return(ierr);

            /* If the year we got from that Julian day is different from the original
             * year specified, it means we have gone off the end of the transition year,
             * probably because that year has less days than regular years.  In that
             * event, return an error.
             */
            if( tyear != year ) {
                sprintf( error_message, "year %d in the %s calendar (with transition date %04d-%02d-%02d) has less than %d days, but that was the day-of-year number requested in a call to ccs_doy2date\n",
                    year, calendar->name, calendar->year_x, calendar->month_x, calendar->day_x, doy );
                return( CALCALCS_ERR_INVALID_DAY_OF_YEAR );
                }

            return(0);
            }
        }
    else
        c2use = calendar;

    /* Check to make sure we are not asking for a doy that does not exist,
     * esp. as regards to the number of days in leap vs. non-leap years
     */
    if( (ierr = c2use->c_isleap( year, &leap )) != 0 )
        return( ierr );
    if( leap == 1 )
        ndays_max = c2use->ndays_leap;
    else
        ndays_max = c2use->ndays_reg;

    if( (doy < 1) || (doy > ndays_max)) {
        sprintf( error_message, "routine ccs_doy2date was passed a day-of-year=%d, but for year %d in the %s calendar, the value must be between 1 and %d",
            doy, year, c2use->name, ndays_max );
        return( CALCALCS_ERR_INVALID_DAY_OF_YEAR );
        }

    /* Get Julian day number of Jan 1st of the specified year */
    if( (ierr = c2use->c_date2jday( year, 1, 1, &jd0 )) != 0 )
        return( ierr );

    /* Calculate new Julian day */
    jd1 = jd0 + doy - 1;

    /* Get date for new Julian day */
    if( (ierr = c2use->c_jday2date( jd1, &tyear, month, day )) != 0 )
        return( ierr );

    return(0);
}

/********************************************************************************************
 * ccs_dayssince: Given a Y/M/D date in a specified calendar, and the number of days since
 *    that date, this returns the new Y/M/D date in a (possibly different) calendar.
 *
 * Note that specifying "zero" days since, and giving different calendars as the original
 *    and new calendars, essentially converts dates between calendars.
 *
 * Returns 0 on success, and a negative value on error.
 */
int ccs_dayssince( calcalcs_cal *calendar_orig, int year_orig, int month_orig, int day_orig,
        int ndays_since, calcalcs_cal *calendar_new, int *year_new, int *month_new, int *day_new )
{
    int        ierr, jd0, jd1;
    calcalcs_cal    *c2use_orig, *c2use_new;

    if( calendar_orig == NULL ) return(CALCALCS_ERR_NULL_CALENDAR);
    if( calendar_orig->sig != CCS_VALID_SIG ) return(CALCALCS_ERR_INVALID_CALENDAR);

    if( calendar_new == NULL ) return(CALCALCS_ERR_NULL_CALENDAR);
    if( calendar_new->sig != CCS_VALID_SIG ) return(CALCALCS_ERR_INVALID_CALENDAR);

    /* Figure out which calendar of the ORIGINAL calendar to use if it's a mixed calendar
     */
    if( calendar_orig->mixed ) {
        if( date_ge( year_orig, month_orig, day_orig,
                calendar_orig->year_x, calendar_orig->month_x, calendar_orig->day_x ))
            c2use_orig = calendar_orig->late_cal;
        else if( date_le( year_orig, month_orig, day_orig,
                calendar_orig->year_px, calendar_orig->month_px, calendar_orig->day_px ))
            c2use_orig = calendar_orig->early_cal;
        else
            {
            sprintf( error_message, "ccs_dayssince: date %04d-%02d-%02d is not a valid date in the %s calendar; it falls between the last date the %s calendar was used (%04d-%02d-%02d) and the first date the %s calendar was used (%04d-%02d-%02d)",
                year_orig, month_orig, day_orig, calendar_orig->name,
                calendar_orig->early_cal->name,
                calendar_orig->year_px, calendar_orig->month_px, calendar_orig->day_px,
                calendar_orig->late_cal->name,
                calendar_orig->year_x, calendar_orig->month_x, calendar_orig->day_x );
            return( CALCALCS_ERR_DATE_NOT_IN_CALENDAR );
            }
        }
    else
        c2use_orig = calendar_orig;

    /* Get Julian day in the original calendar and date combo */
    if( (ierr = c2use_orig->c_date2jday( year_orig, month_orig, day_orig, &jd0 )) != 0 )
        return(ierr);

    /* Get new Julian day */
    jd1 = jd0 + ndays_since;

    if( calendar_new->mixed ) {
        /* Figure out which calendar of the NEW calendar to use if it's a mixed calendar.
         */
        if( jd1 >= calendar_new->jday_x )
            c2use_new = calendar_new->late_cal;
        else
            c2use_new = calendar_new->early_cal;
        }
    else
        c2use_new = calendar_new;

    /* Convert the new Julian day to a date in the new calendar */
    if( (ierr = c2use_new->c_jday2date( jd1, year_new, month_new, day_new )) != 0 )
        return( ierr );

    return(0);
}

/********************************************************************************************/
static void ccs_gxd_add_country(const char *code, const char *longname, int year, int month, int day )
{
    if( ccs_n_country_codes >= CCS_MAX_N_COUNTRY_CODES ) {
        fprintf( stderr, "Error, the calcalcs library is attempting to store more country codes than is possible; max is %d\n",
            CCS_MAX_N_COUNTRY_CODES );
        fprintf( stderr, "To fix, recompile with a larger number for CCS_MAX_N_COUNTRY_CODES\n" );
        exit( -1 );
        }

    ccs_xition_dates[ccs_n_country_codes] = (ccs_country_code *)malloc( sizeof( ccs_country_code ));
    if( ccs_xition_dates[ccs_n_country_codes] == NULL ) {
        fprintf( stderr, "calcalcs routine ccs_gxd_add_country: Error trying to allocate space for country code %s\n",
            code );
        exit(-1);
        }

    ccs_xition_dates[ccs_n_country_codes]->code = (char *)malloc( sizeof(char) * (strlen(code)+1) );
    if( ccs_xition_dates[ccs_n_country_codes]->code == NULL ) {
        fprintf( stderr, "calcalcs routine ccs_gxd_add_country: Error trying to allocate space for country code named %s\n",
            code );
        exit(-1);
        }
    strcpy( ccs_xition_dates[ccs_n_country_codes]->code, code );

    ccs_xition_dates[ccs_n_country_codes]->longname = (char *)malloc( sizeof(char) * (strlen(longname)+1) );
    if( ccs_xition_dates[ccs_n_country_codes]->longname == NULL ) {
        fprintf( stderr, "calcalcs routine ccs_gxd_add_country: Error trying to allocate space for country code long name %s\n",
            longname );
        exit(-1);
        }
    strcpy( ccs_xition_dates[ccs_n_country_codes]->longname, longname );

     ccs_xition_dates[ccs_n_country_codes]->year  = year;
     ccs_xition_dates[ccs_n_country_codes]->month = month;
     ccs_xition_dates[ccs_n_country_codes]->day   = day;

    ccs_n_country_codes++;
}

/********************************************************************************************/
static void ccs_init_country_database()
{
    ccs_gxd_add_country( "AK", "Alaska",         1867, 10, 18 );
    ccs_gxd_add_country( "AL", "Albania",         1912, 12,  1 );
    ccs_gxd_add_country( "AT", "Austria",         1583, 10, 16 );
    ccs_gxd_add_country( "BE", "Belgium",         1582, 12, 25 );
    ccs_gxd_add_country( "BG", "Bulgaria",         1916,  4,  1 );
    ccs_gxd_add_country( "CN", "China",           1929,  1,  1 );
    ccs_gxd_add_country( "CZ", "Czechoslovakia",     1584,  1, 17 );
    ccs_gxd_add_country( "DK", "Denmark",            1700,  3,  1 );
    ccs_gxd_add_country( "NO", "Norway",         1700,  3,  1 );
    ccs_gxd_add_country( "EG", "Egypt",        1875,  1,  1 );
    ccs_gxd_add_country( "EE", "Estonia",        1918,  1,  1 );
    ccs_gxd_add_country( "FI", "Finland",        1753,  3,  1 );
    ccs_gxd_add_country( "FR", "France",         1582, 12, 20 );
    ccs_gxd_add_country( "DE", "Germany",        1583, 11, 22 );
    ccs_gxd_add_country( "UK", "United Kingdom",    1752,  9, 14 );
    ccs_gxd_add_country( "GR", "Greece",        1924,  3, 23 );
    ccs_gxd_add_country( "HU", "Hungary",        1587, 11,  1 );
    ccs_gxd_add_country( "IT", "Italy",        1582, 10, 15 );
    ccs_gxd_add_country( "JP", "Japan",        1918,  1,  1 );
    ccs_gxd_add_country( "LV", "Latvia",        1915,  1,  1 );
    ccs_gxd_add_country( "LT", "Lithuania",        1915,  1,  1 );
    ccs_gxd_add_country( "LU", "Luxemburg",        1582, 12, 15 );
    ccs_gxd_add_country( "NL", "Netherlands",    1582, 10, 15 );
    ccs_gxd_add_country( "PL", "Poland",        1582, 10, 15 );
    ccs_gxd_add_country( "PT", "Portugal",        1582, 10, 15 );
    ccs_gxd_add_country( "RO", "Romania",        1919,  4, 14 );
    ccs_gxd_add_country( "ES", "Spain",        1582, 10, 15 );
    ccs_gxd_add_country( "SE", "Sweden",        1753,  3,  1 );
    ccs_gxd_add_country( "CH", "Switzerland",    1584,  1, 22 );
    ccs_gxd_add_country( "TR", "Turkey",        1927,  1,  1 );
    ccs_gxd_add_country( "YU", "Yugoslavia",    1919,  1,  1 );
    ccs_gxd_add_country( "US", "United States",    1752,  9, 14 );
    ccs_gxd_add_country( "SU", "Soviet Union",    1918,  2,  1 );
    ccs_gxd_add_country( "RU", "Russia",        1918,  2,  1 );

    have_initted_country_codes = 1;
}

/********************************************************************************************/
int ccs_get_xition_date( const char *country_code, int *year, int *month, int *day )
{
    int    i;

    if( ! have_initted_country_codes )
        ccs_init_country_database();

    if( strcmp( country_code, "??" ) == 0 ) {
        ccs_dump_xition_dates();
        *year  = 0;
        *month = 0;
        *day   = 0;
        return(0);
        }

    /* Find the passed country code in our list */
    for( i=0; i<ccs_n_country_codes; i++ ) {
        if( strcmp( country_code, ccs_xition_dates[i]->code ) == 0 ) {
            *year  =  ccs_xition_dates[i]->year;
            *month =  ccs_xition_dates[i]->month;
            *day   =  ccs_xition_dates[i]->day;
            return(0);
            }
        }

    /* Maybe they passed a longname? */
    for( i=0; i<ccs_n_country_codes; i++ ) {
        if( strcmp( country_code, ccs_xition_dates[i]->longname ) == 0 ) {
            *year  =  ccs_xition_dates[i]->year;
            *month =  ccs_xition_dates[i]->month;
            *day   =  ccs_xition_dates[i]->day;
            return(0);
            }
        }

    sprintf( error_message, "ccs_get_xition_date: unknown calendar country/region code: \"%s\". Known codes: ", country_code );
    for( i=0; i<ccs_n_country_codes; i++ ) {
        if( (strlen(error_message) + strlen(ccs_xition_dates[i]->code) + strlen(ccs_xition_dates[i]->longname) + 10) < CCS_ERROR_MESSAGE_LEN ) {
            strcat( error_message, ccs_xition_dates[i]->code );
            strcat( error_message, " (" );
            strcat( error_message, ccs_xition_dates[i]->longname );
            strcat( error_message, ") " );
            }
        }

    return(CALCALCS_ERR_UNKNOWN_COUNTRY_CODE);
}

/********************************************************************************************/
static void ccs_dump_xition_dates( void )
{
    int    i;

    printf( "Calcalcs library known country codes:\n" );
    for( i=0; i<ccs_n_country_codes; i++ ) {
        printf( "Code: %s     Transition date: %04d-%02d-%02d   Country/Region: %s\n",
            ccs_xition_dates[i]->code,
            ccs_xition_dates[i]->year,
            ccs_xition_dates[i]->month,
            ccs_xition_dates[i]->day,
            ccs_xition_dates[i]->longname );
        if( i%3 == 2 )
            printf( "\n" );
        }
}

/********************************************************************************************/
int ccs_set_xition_date( calcalcs_cal *calendar, int year, int month, int day )
{
    int    ierr, dpm;

    if( calendar == NULL ) return(CALCALCS_ERR_NULL_CALENDAR);
    if( calendar->sig != CCS_VALID_SIG ) return(CALCALCS_ERR_INVALID_CALENDAR);

    if( calendar->mixed == 0 )
        return( CALCALCS_ERR_NOT_A_MIXED_CALENDAR );

    /* Check to make sure the specified date is a valid date in the
     * LATE calendar (since the transition date is the first date
     * that the late calendar is used)
     */
     if( (ierr = ccs_dpm( calendar->late_cal, year, month, &dpm )) != 0 )
        return( ierr );

    if( (month < 1) || (month > 12) || (day < 1) || (day > dpm)) {
        fprintf( stderr, "Error in routine set_cal_xition_date: trying to set the calendar Julian/Gregorian transition date to an illegal date: %04d-%02d-%02d\n", year, month, day );
        return( CALCALCS_ERR_DATE_NOT_IN_CALENDAR );
        }

    calendar->year_x    = year;
    calendar->month_x   = month;
    calendar->day_x     = day;

    if( (ierr = set_xition_extra_info( calendar )) != 0 )
        return( ierr );

    return(0);
}

/********************************************************************************************/
char *ccs_err_str(int ccs_errno)
{
    if (!ccs_errno)
    {
        sprintf(error_message, "no error from calcalcs routines version %f", CALCALCS_VERSION_NUMBER );
    }
    else if (ccs_errno == CALCALCS_ERR_NULL_CALENDAR)
    {
        sprintf( error_message, "a NULL calendar was passed to the calcalcs routine");
    }
    else if (ccs_errno == CALCALCS_ERR_INVALID_CALENDAR)
    {
        sprintf(error_message, "an invalid, malformed, previously-freed, "
            "or uninitialized calendar was passed to the calcalcs routine");
    }
    else
    {
        sprintf(error_message, "unknown error");
    }

    return error_message;
}

/*==================================================================================================
 * Returns 0 on success, <0 on error.
 */
int c_isleap_julian( int year, int *leap )
{
    int    tyear;

    if( year == 0 ) {
        sprintf( error_message, "the Julian calendar has no year 0" );
        return( CALCALCS_ERR_DATE_NOT_IN_CALENDAR );
        }

    /* Because there is no year 0 in the Julian calendar, years -1, -5, -9, etc
     * are leap years.
     */
    if( year < 0 )
        tyear = year + 1;
    else
        tyear = year;

    *leap = ((tyear % 4) == 0);

    return(0);
}

/*==================================================================================================
 * Returns 0 on success, <0 on error.
 */
int c_isleap_gregorian( int year, int *leap )
{
    int    tyear;

    if( year == 0 ) {
        sprintf( error_message, "the Gregorian calendar has no year 0. Use the \"Gregorian_y0\" calendar if you want to include year 0." );
        return( CALCALCS_ERR_DATE_NOT_IN_CALENDAR );
        }

    /* Because there is no year 0 in the gregorian calendar, years -1, -5, -9, etc
     * are leap years.
     */
    if( year < 0 )
        tyear = year + 1;
    else
        tyear = year;

    *leap = (((tyear % 4) == 0) && ((tyear % 100) != 0)) || ((tyear % 400) == 0);

    return(0);
}

/*==================================================================================================
 * Returns 0 on success, <0 on error.
 */
int c_isleap_gregorian_y0( int year, int *leap )
{
    *leap = (((year % 4) == 0) && ((year % 100) != 0)) || ((year % 400) == 0);

    return(0);
}

/*==================================================================================================
 * Given a Y/M/D in the Gregorian calendar, this computes the (true) Julian day number of the
 * specified date.  Julian days are counted starting at 0 on 1 Jan 4713 BC using a proleptic Julian
 * calendar.  The algorithm is based on the "counting" algorithm in the C++ code I obtained from
 * Edward M. Reingold's web site at http://emr.cs.uiuc.edu/~reingold/calendar.C.
 * In that file, the work is declared to be in the public domain.  I modified it by
 * extending it to negative years (years BC) in addition to positive years, and to use
 * actual Julian Days as the counter.  Otherwise, the spirit of the algorithm is similar.
 *
 * Returns 0 on success, <0 on error.
 */
int c_date2jday_gregorian( int year, int month, int day, int *jday )
{
    int    m, leap, *dpm2use, err;

    if( (month < 1) || (month > 12) || (day < 1) || (day > 31)) {
        sprintf( error_message, "date %04d-%02d-%02d does not exist in the Gregorian calendar",
            year, month, day );
        return( CALCALCS_ERR_DATE_NOT_IN_CALENDAR );
        }

    if( year == 0 ) {
        sprintf( error_message, "year 0 does not exist in the Gregorian calendar.  Use the \"Gregorian_y0\" calendar if you want to include year 0" );
        return( CALCALCS_ERR_NO_YEAR_ZERO );
        }

    /* Limit ourselves to positive Julian Days */
    if( year < -4714 ) {
        sprintf( error_message, "year %d is out of range of the Gregorian calendar routines; must have year >= -4714", year );
        return( CALCALCS_ERR_OUT_OF_RANGE );
        }

    /* Following is necessary because Gregorian calendar skips year 0, so the
     * offst for negative years is different than offset for positive years
     */
    if( year < 0 )
        year += 4801;
    else
        year += 4800;

    if( (err = c_isleap_gregorian( year, &leap )) != 0 )
        return( err );

    if( leap )
        dpm2use = dpm_leap_idx1;
    else
        dpm2use = dpm_idx1;

    *jday = day;
    for( m=month-1; m>0; m-- )
        *jday += dpm2use[m];

    *jday += 365*(year-1) + (year-1)/4 - (year-1)/100 + (year-1)/400;

    /* Ajust to "true" Julian days. This constant is how many days difference there is
     * between the Julian Day origin date of 4713 BC and our offset date of 4800 BC
     */
    *jday -= 31739;

    return(0);
}

/*==================================================================================================
 * Given a Y/M/D in the Gregorian calendar, this computes the (true) Julian day number of the
 * specified date.  Julian days are counted starting at 0 on 1 Jan 4713 BC using a proleptic Julian
 * calendar.  The algorithm is based on the "counting" algorithm in the C++ code I obtained from
 * Edward M. Reingold's web site at http://emr.cs.uiuc.edu/~reingold/calendar.C.
 * In that file, the work is declared to be in the public domain.  I modified it by
 * extending it to negative years (years BC) in addition to positive years, and to use
 * actual Julian Days as the counter.  Otherwise, the spirit of the algorithm is similar.
 *
 * Returns 0 on success, <0 on error.
 */
int c_date2jday_gregorian_y0( int year, int month, int day, int *jday )
{
    int    m, leap, *dpm2use, err;

    if( (month < 1) || (month > 12) || (day < 1) || (day > 31)) {
        sprintf( error_message, "date %04d-%02d-%02d does not exist in the Gregorian calendar",
            year, month, day );
        return( CALCALCS_ERR_DATE_NOT_IN_CALENDAR );
        }

    /* Limit ourselves to positive Julian Days */
    if( year < -4714 ) {
        sprintf( error_message, "year %d is out of range of the Gregorian calendar routines; must have year >= -4714", year );
        return( CALCALCS_ERR_OUT_OF_RANGE );
        }

    year += 4800;

    if( (err = c_isleap_gregorian_y0( year, &leap )) != 0 )
        return( err );

    if( leap )
        dpm2use = dpm_leap_idx1;
    else
        dpm2use = dpm_idx1;

    *jday = day;
    for( m=month-1; m>0; m-- )
        *jday += dpm2use[m];

    *jday += 365*(year-1) + (year-1)/4 - (year-1)/100 + (year-1)/400;

    /* Ajust to "true" Julian days. This constant is how many days difference there is
     * between the Julian Day origin date of 4713 BC and our offset date of 4800 BC
     */
    *jday -= 31739;

    return(0);
}

/*==========================================================================================
 * Given a (true) Julian Day, this converts to a date in the Gregorian calendar.
 * Technically, in the proleptic Gregorian calendar, since this works for dates
 * back to 4713 BC.  Again based on the same public domain code from Edward Reingold's
 * web site as the date2jday routine, extended by me to apply to negative years (years BC).
 *
 * Returns 0 on success, <0 on error.
 */
int c_jday2date_gregorian( int jday, int *year, int *month, int *day )
{
    int    tjday, leap, *dpm2use, ierr, yp1;

    /* Make first estimate for year. We subtract 4714 because Julian Day number
     * 0 occurs in year 4714 BC in the Gregorian calendar (recall that it occurs
     * in year 4713 BC in the JULIAN calendar
     */
    *year = jday/366 - 4714;

    /* Advance years until we find the right one */
    yp1 = *year + 1;
    if( yp1 == 0 ) yp1 = 1;    /* no year 0 in the Gregorian calendar */
    if( (ierr = c_date2jday_gregorian( yp1, 1, 1, &tjday )) != 0 )
        return( ierr );
    while( jday >= tjday ) {
        (*year)++;
        if( *year == 0 )
            *year = 1;    /* no year 0 in the Gregorian calendar */
        yp1 = *year + 1;
        if( yp1 == 0 ) yp1 = 1;    /* no year 0 in the Gregorian calendar */
        if( (ierr = c_date2jday_gregorian( yp1, 1, 1, &tjday )) != 0 )
            return( ierr );
        }

    if( (ierr = c_isleap_gregorian( *year, &leap )) != 0 )
        return( ierr );
    if( leap )
        dpm2use = dpm_leap_idx1;
    else
        dpm2use = dpm_idx1;

    *month = 1;
    if( (ierr = c_date2jday_gregorian( *year, *month, dpm2use[*month], &tjday)) != 0)
        return( ierr );
    while( jday > tjday ) {
        (*month)++;
        if( (ierr = c_date2jday_gregorian( *year, *month, dpm2use[*month], &tjday) != 0))
            return( ierr );
        }

    if( (ierr = c_date2jday_gregorian( *year, *month, 1, &tjday)) != 0 )
        return( ierr );
    *day = jday - tjday + 1;

    return(0);
}

/*==========================================================================================
 * Given a (true) Julian Day, this converts to a date in the Gregorian calendar.
 * Technically, in the proleptic Gregorian calendar, since this works for dates
 * back to 4713 BC.  Again based on the same public domain code from Edward Reingold's
 * web site as the date2jday routine, extended by me to apply to negative years (years BC).
 *
 * Returns 0 on success, <0 on error.
 */
int c_jday2date_gregorian_y0( int jday, int *year, int *month, int *day )
{
    int    tjday, leap, *dpm2use, ierr, yp1;

    /* Make first estimate for year. We subtract 4714 because Julian Day number
     * 0 occurs in year 4714 BC in the Gregorian calendar (recall that it occurs
     * in year 4713 BC in the JULIAN calendar
     */
    *year = jday/366 - 4715;

    /* Advance years until we find the right one */
    yp1 = *year + 1;
    if( (ierr = c_date2jday_gregorian_y0( yp1, 1, 1, &tjday )) != 0 )
        return( ierr );
    while( jday >= tjday ) {
        (*year)++;
        yp1 = *year + 1;
        if( (ierr = c_date2jday_gregorian_y0( yp1, 1, 1, &tjday )) != 0 )
            return( ierr );
        }

    if( (ierr = c_isleap_gregorian_y0( *year, &leap )) != 0 )
        return( ierr );
    if( leap )
        dpm2use = dpm_leap_idx1;
    else
        dpm2use = dpm_idx1;

    *month = 1;
    if( (ierr = c_date2jday_gregorian_y0( *year, *month, dpm2use[*month], &tjday)) != 0)
        return( ierr );
    while( jday > tjday ) {
        (*month)++;
        if( (ierr = c_date2jday_gregorian_y0( *year, *month, dpm2use[*month], &tjday) != 0))
            return( ierr );
        }

    if( (ierr = c_date2jday_gregorian_y0( *year, *month, 1, &tjday)) != 0 )
        return( ierr );
    *day = jday - tjday + 1;

    return(0);
}

/*==================================================================================================
 * Given a Y/M/D in the Julian calendar, this computes the (true) Julian day number of the
 * specified date.  Julian days are counted starting at 0 on 1 Jan 4713 BC using a proleptic Julian
 * calendar.  The algorithm is based on the "counting" algorithm in the C++ code I obtained from
 * Edward M. Reingold's web site at http://emr.cs.uiuc.edu/~reingold/calendar.C.
 * In that file, the work is declared to be in the public domain.  I modified it by
 * extending it to negative years (years BC) in addition to positive years, and to use
 * actual Julian Days as the counter.  Otherwise, the spirit of the algorithm is similar.
 *
 * Returns 0 on success, <0 on error.
 */
int c_date2jday_julian( int year, int month, int day, int *jday )
{
    int    m, leap, *dpm2use, err;

    if( (month < 1) || (month > 12) || (day < 1) || (day > 31)) {
        sprintf( error_message, "date %04d-%02d-%02d does not exist in the Julian calendar",
            year, month, day );
        return( CALCALCS_ERR_DATE_NOT_IN_CALENDAR );
        }

    if( year == 0 ) {
        sprintf( error_message, "year 0 does not exist in the Julian calendar" );
        return( CALCALCS_ERR_NO_YEAR_ZERO );
        }

    /* Limit ourselves to positive Julian Days */
    if( year < -4713 ) {
        sprintf( error_message, "year %d is out of range of the Julian calendar routines; must have year >= -4713", year );
        return( CALCALCS_ERR_OUT_OF_RANGE );
        }

    /* Following is necessary because Julian calendar skips year 0, so the
     * offst for negative years is different than offset for positive years
     */
    if( year < 0 )
        year += 4801;
    else
        year += 4800;

    if( (err = c_isleap_julian( year, &leap )) != 0 )
        return( err );

    if( leap )
        dpm2use = dpm_leap_idx1;
    else
        dpm2use = dpm_idx1;

    *jday = day;
    for( m=month-1; m>0; m-- )
        *jday += dpm2use[m];

    *jday += 365*(year-1) + (year-1)/4;

    /* Ajust to "true" Julian days. This constant is how many days difference there is
     * between the Julian Day origin date of 4713 BC and our offset date of 4800 BC
     */
    *jday -= 31777;

    return(0);
}

/*==========================================================================================
 * Given a (true) Julian Day, this converts to a date in the Julian calendar.
 * Technically, in the proleptic Julian calendar, since this works for dates
 * back to 4713 BC.  Again based on the same public domain code from Edward Reingold's
 * web site as the date2jday routine, extended by me to apply to negative years (years BC).
 *
 * Returns 0 on success, <0 on error.
 */
int c_jday2date_julian( int jday, int *year, int *month, int *day )
{
    int    tjday, leap, *dpm2use, ierr, yp1;

    /* Make first estimate for year. We subtract 4713 because Julian Day number
     * 0 occurs in year 4713 BC in the Julian calendar
     */
    *year = jday/366 - 4713;

    /* Advance years until we find the right one */
    yp1 = *year + 1;
    if( yp1 == 0 ) yp1 = 1;    /* no year 0 in the Julian calendar */
    if( (ierr = c_date2jday_julian( yp1, 1, 1, &tjday )) != 0 )
        return( ierr );
    while( jday >= tjday ) {
        (*year)++;
        if( *year == 0 )
            *year = 1;    /* no year 0 in the Julian calendar */
        yp1 = *year + 1;
        if( yp1 == 0 ) yp1 = 1;    /* no year 0 in the Julian calendar */
        if( (ierr = c_date2jday_julian( yp1, 1, 1, &tjday )) != 0 )
            return( ierr );
        }

    if( (ierr = c_isleap_julian( *year, &leap )) != 0 )
        return( ierr );
    if( leap )
        dpm2use = dpm_leap_idx1;
    else
        dpm2use = dpm_idx1;

    *month = 1;
    if( (ierr = c_date2jday_julian( *year, *month, dpm2use[*month], &tjday)) != 0)
        return( ierr );
    while( jday > tjday ) {
        (*month)++;
        if( (ierr = c_date2jday_julian( *year, *month, dpm2use[*month], &tjday) != 0))
            return( ierr );
        }

    if( (ierr = c_date2jday_julian( *year, *month, 1, &tjday)) != 0 )
        return( ierr );
    *day = jday - tjday + 1;

    return(0);
}

/*==================================================================================================
 * Free the storage associated with a calendar
 */
void ccs_free_calendar( calcalcs_cal *cc )
{
    if( cc == NULL )
        return;

    if( cc->mixed == 1 ) {
        ccs_free_calendar( cc->early_cal );
        ccs_free_calendar( cc->late_cal );
        }

    if( cc->sig != CCS_VALID_SIG ) {
        fprintf( stderr, "Warning: invalid calendar passed to routine ccs_free_calendar!\n" );
        return;
        }

    cc->sig = 0;

    if( cc->name != NULL )
        free( cc->name );

    free( cc );
}

/**********************************************************************************************/
static int date_gt( int year, int month, int day, int y2, int m2, int d2 )
{
    return( ! date_le( year, month, day, y2, m2, d2 ));
}

/**********************************************************************************************/
static int date_lt( int year, int month, int day, int y2, int m2, int d2 )
{
    return( ! date_ge( year, month, day, y2, m2, d2 ));
}

/**********************************************************************************************/
static int date_le( int year, int month, int day, int y2, int m2, int d2 ) {

    if( year > y2 )
        return(0);

    if( year < y2 )
        return(1);

    /* If get here, must be same year */
    if( month > m2 )
        return(0);
    if( month < m2)
        return(1);

    /* If get here, must be same month */
    if( day > d2 )
        return(0);

    return(1);
}

/**********************************************************************************************/
static int date_ge( int year, int month, int day, int y2, int m2, int d2 ) {

    if( year < y2 )
        return(0);

    if( year > y2 )
        return(1);

    /* If get here, must be same year */
    if( month < m2 )
        return(0);
    if( month > m2)
        return(1);

    /* If get here, must be same month */
    if( day < d2 )
        return(0);

    return(1);
}

/**********************************************************************************************/
int c_isleap_never( int year, int *leap )
{
    (void)year;
    *leap = 0;
    return( 0 );
}

/**********************************************************************************************/
int c_date2jday_360_day( int year, int month, int day, int *jday )
{
    int spm;

    if( day > 30) {
        sprintf( error_message, "date %04d-%02d-%02d does not exist in the 360_day calendar",
            year, month, day );
        return( CALCALCS_ERR_DATE_NOT_IN_CALENDAR );
        }

    spm = (month-1)*30;    /* sum of days in the previous months */

    *jday = year*360 + spm + (day-1);

    return( 0 );
}

/**********************************************************************************************/
int c_date2jday_noleap( int year, int month, int day, int *jday )
{
    if( (month == 2) && (day == 29)) {
        sprintf( error_message, "date %04d-%02d-%02d does not exist in the noleap calendar",
            year, month, day );
        return( CALCALCS_ERR_DATE_NOT_IN_CALENDAR );
        }

    *jday = year*365 + spm_idx1[month] + (day-1);

    return( 0 );
}

/**********************************************************************************************/
int c_jday2date_360_day( int jday, int *year, int *month, int *day )
{
    int    nextra, yr_offset, doy;

    yr_offset = 0;
    if( jday < 0 ) {
        yr_offset = (-jday)/360+1;
        jday += 360*yr_offset;
        }

    *year = jday/360;

    nextra = jday - *year*360;
    doy    = nextra + 1;    /* Julday numbering starts at 0, doy starts at 1 */
    *month = nextra/30 + 1;
    *day   = doy - (*month-1)*30;

    *year -= yr_offset;

    return(0);
}

/**********************************************************************************************/
int c_jday2date_noleap( int jday, int *year, int *month, int *day )
{
    int    nextra, yr_offset, doy;

    yr_offset = 0;
    if( jday < 0 ) {
        yr_offset = (-jday)/365+1;
        jday += 365*yr_offset;
        }

    *year = jday/365;

    nextra = jday - *year*365;
    doy    = nextra + 1;    /* Julday numbering starts at 0, doy starts at 1 */
    *month = 1;
    while( doy > spm_idx1[*month + 1] )
        *month += 1;

    *day = doy - spm_idx1[*month];

    *year -= yr_offset;

    return(0);
}

/**********************************************************************************************/
int c_dpm_gregorian( int year, int month, int *dpm )
{
    int    ierr, leap;

    if( (month<1) || (month>12)) {
        sprintf( error_message, "month %d does not exist in the Gregorian calendar", month );
        return( CALCALCS_ERR_DATE_NOT_IN_CALENDAR );
        }

    if( (ierr = c_isleap_gregorian( year, &leap )) != 0 )
        return( ierr );

    if( leap )
        *dpm = dpm_leap_idx1[month];
    else
        *dpm = dpm_idx1[month];

    return(0);
}


/**********************************************************************************************/
int c_dpm_gregorian_y0( int year, int month, int *dpm )
{
    int    ierr, leap;

    if( (month<1) || (month>12)) {
        sprintf( error_message, "month %d does not exist in the Gregorian calendar", month );
        return( CALCALCS_ERR_DATE_NOT_IN_CALENDAR );
        }

    if( (ierr = c_isleap_gregorian_y0( year, &leap )) != 0 )
        return( ierr );

    if( leap )
        *dpm = dpm_leap_idx1[month];
    else
        *dpm = dpm_idx1[month];

    return(0);
}

/**********************************************************************************************/
int c_dpm_julian( int year, int month, int *dpm )
{
    int    ierr, leap;

    if( (month<1) || (month>12)) {
        sprintf( error_message, "month %d does not exist in the Julian calendar", month );
        return( CALCALCS_ERR_DATE_NOT_IN_CALENDAR );
        }

    if( (ierr = c_isleap_julian( year, &leap )) != 0 )
        return( ierr );

    if( leap )
        *dpm = dpm_leap_idx1[month];
    else
        *dpm = dpm_idx1[month];

    return(0);
}

/**********************************************************************************************/
int c_dpm_360_day( int year, int month, int *dpm )
{
    (void)year;
    if( (month<1) || (month>12)) {
        sprintf( error_message, "month %d does not exist in the 360_day calendar", month );
        return( CALCALCS_ERR_DATE_NOT_IN_CALENDAR );
        }

    *dpm = 30;

    return(0);
}

/**********************************************************************************************/
int c_dpm_noleap( int year, int month, int *dpm )
{
    (void)year;
    if( (month<1) || (month>12)) {
        sprintf( error_message, "month %d does not exist in the noleap calendar", month );
        return( CALCALCS_ERR_DATE_NOT_IN_CALENDAR );
        }

    *dpm = dpm_idx1[month];

    return(0);
}

/*******************************************************************************************/
static int set_xition_extra_info( calcalcs_cal *cal )
{
    int ierr = 0;

    /* This is the Julian Day of the transition date */
    if ((ierr = ccs_date2jday( cal->late_cal, cal->year_x, cal->month_x, cal->day_x, &(cal->jday_x))))
    {
        sprintf(error_message, "Failed to turn the mixed calendar transition "
            "day %04d-%02d-%02d in the %s calendar into a Julian day!\n", cal->year_x,
            cal->month_x, cal->day_x, cal->name);
        return ierr;
     }

    /* This is the date of the day BEFORE the transition day,
     * i.e., the last day that the early calendar was used
     */
    if ((ierr = ccs_jday2date(cal->early_cal, (cal->jday_x-1), &(cal->year_px), &(cal->month_px), &(cal->day_px))))
    {
        const char *estr = ccs_err_str(ierr);
        sprintf(error_message, "Failed to turn the day before the mixed "
            "calendar transition day into a date! %s\n", estr);
        return ierr;
    }

    return 0;
}

/*
 * code below is from utCalendar2_cal
 * we're including it here so that all the calendaring utility
 * is in one h,c source
 */
#if defined(CALCALCS_THREAD)
static std::mutex calcalcs_mutex;
#endif

static int have_initted = 0;
static calcalcs_cal *cal_std = nullptr;
static ut_system *u_system = nullptr;

/* Following controls the rounding precision of the routines. I.e., if we end up with a value
   such as 59.99999999 seconds, we are going to round it to 60 seconds. The value is given
   in seconds, so, for example, 1.e-3 means round up to 1 second if the value is 0.999 seconds or greater,
   and 1.e-6 means round up to 1 second if the value is 0.999999 seconds or greater.
*/
static double sec_rounding_value = 1.e-8;

/* Internal to this file only */

static int initialize();
static void get_origin( ut_unit *dataunits, int *y0, int *mon0, int *d0, int *h0, int *min0, double *s0 );
static cv_converter *get_day_to_user_converter( ut_unit *uu, int y0, int mon0, int d0, int h0, int min0, double s0 );
static cv_converter *get_user_to_day_converter( ut_unit *uu, int y0, int mon0, int d0, int h0, int min0, double s0 );
static calcalcs_cal *getcal( const char *name );

/* Stores previuosly initialized calendars and their names */
static int maxcals_known=100;
static int ncals_known=0;
static calcalcs_cal **known_cal;        /* ptr to array of calcals_cal ptrs */
static char **known_cal_name;

/* for some oiptimizations that skip initialization of
 * calendar when the calendar is not changed in between
 * invocations */
static char current_calendar_name[1024] = {'\0'};
static char current_unit_str[1024] = {'\0'};
static calcalcs_cal *current_calendar = nullptr;
static ut_unit *current_units = nullptr;
static cv_converter *conv_user_units_to_days = nullptr;
static cv_converter *conv_days_to_user_units = nullptr;
static int y0=0, mon0=0, d0=0, h0=0, min0=0, jday0=0;
static double s0=0.0, fpartday0=0.0, extra_seconds0=0.0;

/* Stores previously emitted "unknown calendar" warnings
static void unknown_cal_emit_warning( const char *calendar_name );
#define UTC2_MAX_UNKCAL_WARNS     1000
static char *unknown_cal_emitted_warning_for[ UTC2_MAX_UNKCAL_WARNS ];
static int n_unkcal=0; */

/*========================================================================================
 * Turns the passed value into a Y/M/D date
 */
int date(double val, int *year, int *month, int *day, int *hour,
    int *minute, double *second, const char *unit_str, const char *calendar_name)
{
    int jdnew, ndinc, ierr, iorig, iround;
    double fdays, extra_seconds, tot_extra_seconds;
    int ndays;

    // initialize and select the calendar
    if ((ierr = set_current_calendar(calendar_name, unit_str)))
        return ierr;

    /* Convert user value of offset to floating point days */
    fdays = cv_convert_double( conv_user_units_to_days, val );

    /* Get integer number of days and seconds past that */
    ndays = fdays;
    extra_seconds = (fdays - ndays)*86400.0;

    /* Get new Julian day */
    jdnew = jday0 + ndays;

    /* Handle the sub-day part */
    tot_extra_seconds = extra_seconds0 + extra_seconds;
    ndinc = tot_extra_seconds / 86400.0;
    jdnew += ndinc;
    tot_extra_seconds -= ndinc*86400.0;
    if( tot_extra_seconds < 0.0 )
    {
        tot_extra_seconds += 86400.0;
        jdnew--;
    }

    /* Convert to a date */
    if ((ierr = ccs_jday2date( current_calendar, jdnew, year, month, day )))
    {
        fprintf(stderr, "Error in utCalendar2: %s\n", ccs_err_str(ierr));
        return UT_EINVALID;
    }

    *hour = tot_extra_seconds / 3600.0;
    tot_extra_seconds -= *hour * 3600.0;
    *minute = tot_extra_seconds / 60.0;
    tot_extra_seconds -= *minute * 60.0;
    *second = tot_extra_seconds;

    /* Handle the rouding issues */
    iorig  = *second;            /* Integer conversion */
    iround = *second + sec_rounding_value;
    if( iround > iorig ) {
        /* printf( "rounding alg invoked, orig date: %04d-%02d-%02d %02d:%02d:%.20lf\n", *year, *month, *day, *hour, *minute, *second ); */
        *second = (double)iround;
        if( *second >= 60.0 ) {
            *second -= 60.0;
            *minute += 1.0;
            if( *minute >= 60.0 ) {
                *minute -= 60.0;
                *hour += 1.0;
                if( *hour >= 24.0 ) {
                    *hour -= 24.0;
                    if( (ierr = ccs_jday2date( current_calendar, jdnew+1, year, month, day )) != 0 ) {
                        fprintf( stderr, "Error in utCalendar2: %s\n", ccs_err_str(ierr) );
                        return( UT_EINVALID );
                        }
                    }
                }
            }
        /* printf( "after rounding alg here is new date: %04d-%02d-%02d %02d:%02d:%02.20lf\n", *year, *month, *day, *hour, *minute, *second ); */
        }

    return 0;
}

/*========================================================================================
 * Turn the passed Y/M/D date into a value in the user's units
 */
int coordinate(int year, int month, int day, int hour, int minute,
    double second, const char *unit_str, const char *calendar_name, double *value)
{
    int jday=0, ierr=0, diff_in_days=0;

    double fdiff_in_days=0.0, val_days=0.0,
           val_partdays=0.0, fdiff_in_partdays=0.0,
           fpartday=0.0;

    // initialize and select the calendar
    if ((ierr = set_current_calendar(calendar_name, unit_str)))
        return ierr;

    /* Turn passed date into a Julian day */
    if((ierr = ccs_date2jday( current_calendar, year, month, day, &jday )))
    {
        fprintf( stderr, "Error in utInvCalendar2: %s\n", ccs_err_str(ierr));
        return UT_EINVALID;
    }

    /* jday and jday0 can be very large and nearly equal, so we difference
     * them first to keep the precision high
     */
    diff_in_days = jday - jday0;
    fdiff_in_days = (double)diff_in_days;

    /* Get the fractional (floating point) part of a Julian day difference
     */
    fpartday = (double)hour/24.0 + (double)minute/1440.0 + second/86400.0;
    fdiff_in_partdays = fpartday - fpartday0;

    /* Convert days and partial days to user's units */
    val_days     = cv_convert_double( conv_days_to_user_units, fdiff_in_days     );
    val_partdays = cv_convert_double( conv_days_to_user_units, fdiff_in_partdays );

    /* Hopefully this will minimize the roundoff errors */
    *value = val_days + val_partdays;

    return 0;
}

/*==============================================================================================
 * Get a converter that turns the user's units into days
 */
static cv_converter *get_user_to_day_converter( ut_unit *uu, int y0, int mon0, int d0, int h0, int min0, double s0 )
{
    char        daystr[1024];
    ut_unit     *udu_days;
    cv_converter    *conv = nullptr;

    sprintf( daystr, "days since %04d-%02d-%02d %02d:%02d:%f",
        y0, mon0, d0, h0, min0, s0 );

    udu_days = ut_parse( ut_get_system(uu), daystr, UT_ASCII );
    if( udu_days == NULL ) {
        fprintf( stderr, "internal error in utCalendar2/conv_to_days: failed to parse following string: \"%s\"\n",
            daystr );
        exit(-1);
        }
    conv = ut_get_converter( uu, udu_days );
    if( conv == NULL ) {
        fprintf( stderr, "internal error in utCalendar2/conv_to_days: cannot convert from \"%s\" to user units\n",
             daystr );
        exit(-1);
        }

    ut_free( udu_days );
    return( conv );
}

/*==============================================================================================
 * Get a converter that turns days into the user's units
 */
static cv_converter *get_day_to_user_converter( ut_unit *uu, int y0, int mon0, int d0, int h0, int min0, double s0 )
{
    char        daystr[1024];
    ut_unit     *udu_days;
    cv_converter    *conv;

    sprintf( daystr, "days since %04d-%02d-%02d %02d:%02d:%f",
        y0, mon0, d0, h0, min0, s0 );

    udu_days = ut_parse( ut_get_system(uu), daystr, UT_ASCII );
    if( udu_days == NULL ) {
        fprintf( stderr, "internal error in utCalendar2/conv_to_user_units: failed to parse following string: \"%s\"\n",
            daystr );
        exit(-1);
        }
    conv = ut_get_converter( udu_days, uu );
    if( conv == NULL ) {
        fprintf( stderr, "internal error in utCalendar2/conv_to_user_units: cannot convert from user units to \"%s\"\n",
             daystr );
        exit(-1);
        }

    free( udu_days );
    return( conv );
}

/*==========================================================================================
 * The user specified some origin to the time units. For example, if the units string
 * were "days since 2005-10-15", then the origin date is 2005-10-15.  This routine
 * deduces the specified origin date from the passed units structure
 */
static void get_origin( ut_unit *dataunits, int *y0, int *mon0, int *d0, int *h0, int *min0, double *s0 )
{
    double        s0lib, rez, tval, tval_conv, resolution;
    cv_converter    *conv_user_date_to_ref_date;
    ut_unit        *tmpu;
    int        y0lib, mon0lib, d0lib, h0lib, min0lib;
    char        ustr[1024];

    static ut_unit     *udu_ref_date=NULL;    /* Saved between invocations */

    if( udu_ref_date == NULL ) {

        /* Make a timestamp units that refers to the udunits2 library's intrinsic
         * time origin.  The first thing we do is parse a timestampe unit
         * (any old timestamp unit) and immediately discard the result, to account
         * for a bug in the udunits-2 library that fails to set the library's
         * time origin unless this step is done first.  Without this, a call to
         * ut_decode_time with tval==0 returns year=-4713 rather than 2001.
         */
        if( (tmpu = ut_parse( ut_get_system(dataunits), "days since 2010-01-09", UT_ASCII )) == NULL ) {
            fprintf( stderr, "Internal error in routnie utCalendar2/get_origin: failed to parse temp timestamp string\n" );
            exit(-1);
            }
        ut_free( tmpu );

        tval = 0.0;
        ut_decode_time( tval, &y0lib, &mon0lib, &d0lib, &h0lib, &min0lib, &s0lib, &rez );
        sprintf( ustr, "seconds since %04d-%02d-%02d %02d:%02d:%f",
            y0lib, mon0lib, d0lib, h0lib, min0lib, s0lib );
        udu_ref_date = ut_parse( ut_get_system(dataunits), ustr, UT_ASCII );
        if( udu_ref_date == NULL ) {
            fprintf( stderr, "internal error in routine utCalendar2/get_origin: could not parse origin string \"%s\"\n",
                ustr );
            exit(-1);
            }
        }

    /* Get converter from passed units to the library's intrinsic time units */
    conv_user_date_to_ref_date = ut_get_converter( dataunits, udu_ref_date );

    /* convert tval=0 to ref date */
    tval = 0.0;
    tval_conv = cv_convert_double( conv_user_date_to_ref_date, tval );

    /* Now decode the converted value */
    ut_decode_time( tval_conv, y0, mon0, d0, h0, min0, s0, &resolution );

    cv_free( conv_user_date_to_ref_date );
}

/*========================================================================================*/
static int initialize()
{
    ut_set_error_message_handler(ut_ignore);
    if (!(u_system = ut_read_xml(nullptr)))
        return -1;
    ut_set_error_message_handler(ut_write_to_stderr);

    /* Make space for our saved calendars */
    known_cal = (calcalcs_cal **)malloc(sizeof(calcalcs_cal*)*maxcals_known);
    if (!known_cal)
    {
        fprintf(stderr, "Error in utCalendar2 routines, could not allocate internal storage\n");
        return -1;
    }

    for (int i=0; i<maxcals_known; i++ )
        known_cal[i] = NULL;

    known_cal_name = (char **)malloc(sizeof(char *)*maxcals_known);
    if (!known_cal_name)
    {
        fprintf(stderr, "Error in utCalendar2 routines, could not allocate internal storage for calendar names\n");
        return -1;
    }

    for (int i=0; i<maxcals_known; i++ )
        known_cal_name[i] = NULL;

    /* Make a standard calendar */
    cal_std = ccs_init_calendar( "Standard" );

    have_initted = 1;    /* a global */
    return 0;
}

/*========================================================================================
 * Returns NULL if the passed calendar name is both not found and not creatable
 */
static calcalcs_cal *getcal( const char *name )
{
    int i, new_index;
    calcalcs_cal *new_cal;

    if( cal_std == NULL ) {
        fprintf( stderr, "Coding error in utCalendar2_cal routines, cal_std is null!\n" );
        exit(-1);
        }

    if( strcasecmp( name, "standard" ) == 0 )
        return( cal_std );

    /* See if it is one of the previously known calendars */
    for( i=0; i<ncals_known; i++ ) {
        if( strcmp( known_cal_name[i], name ) == 0 )
            return( known_cal[i] );
        }

    /* If we get here, the cal is not known, so create it if possible */
    new_cal = ccs_init_calendar( name );
    if( new_cal == NULL )
        return( NULL );        /* unknown name */

    /* We now know a new calendar */
    new_index = ncals_known;
    ncals_known++;

    /* new_index is where we will be storing the new calendar */
    if( ncals_known > maxcals_known ) {
        ncals_known = maxcals_known;
        new_index = strlen(name);    /* some arbitrary value */
        if( new_index >= maxcals_known )
            new_index = 10;
        }

    /* If there was already a calendar stored in this slot
     * (because we might be reusing slots) then clear it out
     */
    if( known_cal[new_index] != NULL )
        ccs_free_calendar( known_cal[new_index] );

    if( known_cal_name[new_index] != NULL )
        free( known_cal_name[new_index] );

    known_cal[new_index] = new_cal;
    known_cal_name[new_index] = (char *)malloc( sizeof(char) * (strlen(name)+1 ));
    strcpy( known_cal_name[new_index], name );

    return( new_cal );
}

/*=============================================================================================
static void unknown_cal_emit_warning( const char *calendar_name )
{
    int    i;

    for( i=0; i<n_unkcal; i++ ) {
        if( strcmp( calendar_name, unknown_cal_emitted_warning_for[i] ) == 0 )
            // Already emitted a warning for this calendar
            return;
        }

    if( n_unkcal == UTC2_MAX_UNKCAL_WARNS )
        // Too many warnings already given, give up
        return;

    // OK, emit the warning now
    fprintf( stderr, "Error in utCalendar2_cal/utInvCalendar2_cal: unknown calendar \"%s\".  Using Standard calendar instead\n",
        calendar_name );

    // Save the fact that we have warned for this calendar
    unknown_cal_emitted_warning_for[ n_unkcal ] = (char *)malloc( sizeof(char) * (strlen(calendar_name)+1 ));
    if( unknown_cal_emitted_warning_for[ n_unkcal ] == NULL ) {
        fprintf( stderr, "Error in utCalendar_cal: could not allocate internal memory to store string \"%s\"\n",
            calendar_name );
        return;
        }

    strcpy( unknown_cal_emitted_warning_for[ n_unkcal ], calendar_name );
    n_unkcal++;
}*/


// ==========================================================================
int set_current_calendar( const char *calendar_name, const char *unit_str )
{
#if defined(CALCALCS_THREAD)
    {
    const std::lock_guard<std::mutex> lock(calcalcs_mutex);
#endif
    int ierr = 0;

    /* See if we are being passed the same units and calendar as last time.  If so,
     * we can optimize by not recomputing all this junk
     */
    if (strncmp(current_calendar_name, calendar_name, 1024)
        || strncmp(current_unit_str, unit_str, 1024))
    {
        // initialize
        if ((!have_initted) && initialize())
        {
            fprintf(stderr, "Error, failed to initialized");
            return -1;
        }

        /* Get the calendar we will be using, based on the passed name */
        if (!(current_calendar = getcal(calendar_name)))
        {
            fprintf(stderr, "Error, unknown calendar %s\n", calendar_name);
            return UT_EINVALID;
        }

        /* create units object from the string, and update the cached string */
        ut_unit *dataunits = ut_parse(u_system, unit_str, UT_ASCII);
        if (!dataunits)
        {
            fprintf(stderr, "Error, bad units %s\n", unit_str);
            return UT_EINVALID;
        }
        strncpy(current_unit_str, unit_str, 1023);

        /* Get origin day of the data units */
        get_origin(dataunits, &y0, &mon0, &d0, &h0, &min0, &s0);    /* Note: static vars */

        /* Number of seconds into the specified origin day */
        extra_seconds0 = h0*3600.0 + min0*60.0 + s0;            /* Note: static vars */

        /* Convert the origin day to Julian Day number in the specified calendar */
        if( (ierr = ccs_date2jday( current_calendar, y0, mon0, d0, &jday0 )) != 0 )
        {
            fprintf( stderr, "Error in utCalendar2: %s\n", ccs_err_str(ierr) );
            return UT_EINVALID;
        }

        /* Get the origin's HMS in fractional (floating point) part of a Julian day */
        fpartday0 = (double)h0/24.0 + (double)min0/1440.0 + s0/86400.0;

        /* Get converter from user-specified units to "days" */
        if (conv_user_units_to_days)
            cv_free( conv_user_units_to_days );

        conv_user_units_to_days =
            get_user_to_day_converter(dataunits, y0, mon0, d0, h0, min0, s0);

        /* Get converter for turning days into user's units */
        if (conv_days_to_user_units)
            cv_free(conv_days_to_user_units);

        conv_days_to_user_units =
            get_day_to_user_converter(dataunits, y0, mon0, d0, h0, min0, s0);

        /* Save these units so we can reuse our time-consuming
         * calculations next time if they are the same units
         */
        if (current_units)
            ut_free(current_units);

        current_units = dataunits;

        strncpy(current_calendar_name, current_calendar->name, 1023);
    }
#if defined(CALCALCS_THREAD)
    }
#endif
    return 0;
}

// ==========================================================================
int is_leap_year( const char *calendar_name, const char *unit_str,
    int year, int &leap )
{
    // initialize and select the calendar
    int ierr = 0;
    if ((ierr = set_current_calendar(calendar_name, unit_str)))
    {
        fprintf(stderr, "Error: system initialization failed\n");
        return ierr;
    }

    // calculate leap year
    if ((ierr = ccs_isleap(current_calendar, year, &leap)))
    {
        fprintf(stderr, "Error, failed to determine if %d in the \"%s\" "
            "calendar with units \"%s\" is a leap year\n", year,
            calendar_name, unit_str);
        return ierr;
    }

    return 0;
}

// ==========================================================================
int days_in_month( const char *calendar_name, const char *unit_str,
    int year, int month, int &dpm )
{
    int ierr = 0;

    // initialize and select the calendar
    if ((ierr = set_current_calendar(calendar_name, unit_str)))
    {
        fprintf(stderr, "Error: system initialization failed");
        return ierr;
    }

    // calculate days in the month
    if ((ierr = ccs_dpm(current_calendar, year, month, &dpm)))
    {
        fprintf(stderr, "Error: failed to get days per month");
        return ierr;
    }

    return 0;
}

};
