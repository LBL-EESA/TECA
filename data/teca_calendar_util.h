#ifndef teca_calendar_h
#define teca_calendar_h

/// @file

#include "teca_config.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"

#include <string>
#include <ostream>
#include <cstring>
#include <memory>

/// Codes dealing with calendaring
namespace teca_calendar_util
{

/** @name Gregorian calendar
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
TECA_EXPORT
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
TECA_EXPORT
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
TECA_EXPORT
bool valid_gregorian_date(long y, long m, long d);
///@}


/// returns one of DJF,MAM,JJA,SON based on the month passed in
TECA_EXPORT
const char *get_season_name(int month);

/** brief
 * A floating point time value and its corresponding year, month day, hour
 * minute and second
 */
struct TECA_EXPORT time_point
{
    time_point() : index(-1), time(0.0), year(0), month(1), day(1),
        hour(0), minute(0), second(0.0)
    {}

    /** Initialize explicitly.
     * @param[in] i the index of the time value
     * @param[in] t the time value
     * @param[in] YYYY the year
     * @param[in] MM the month
     * @param[in] DD the day
     * @param[in] hh the hour
     * @param[in] mm the minute
     * @param[in] ss the second
     */
    time_point(long i, double t, int YYYY=0, int MM=1, int DD=1,
        int hh=0, int mm=0, double ss=0.0) : index(i), time(t),
        year(YYYY), month(MM), day(DD), hour(hh), minute(mm),
        second(ss)
    {}


    /** Initialize from a floating point time value. The calendar and units
     * must be provided.
     * @param[in] i the index of the time value
     * @param[in] t the time value
     * @param[in] units the units t is in
     * @param[in] calendar the calendar system the units are in
     */
    time_point(long i, double t,
        const std::string &units, const std::string &calendar);

    long index;
    double time;
    int year;
    int month;
    int day;
    int hour;
    int minute;
    double second;
};

/// An iterator over a series of time intervals
class TECA_EXPORT interval_iterator
{
public:

    interval_iterator() : time(), units(), calendar(),
        begin(), end(), valid(false)
    {}

    virtual ~interval_iterator() {}

    /** Initialize the iterator from a metadata object following the
     * conventions defined by the teca_cf_reader.
     * @returns 0 if successfully initialized
     */
    virtual int initialize(const teca_metadata &md);

    /** Initialize the iterator from a metadata object following the
     * conventions defined by the teca_cf_reader.
     * @param[in] md a metadata object
     * @param[in] first_step the first step to include in the series or 0 to use all
     * @param[in] last_step the last step to include in the series or -1 to use all
     * @returns 0 if successfully initialized
     */
    virtual int initialize(const teca_metadata &md,
        long first_step, long last_step);

    /** Initialize the iterator.
     * @param[in] t  An array of time values
     * @param[in] units A string units of the time values
     * @param[in] calendar A string name of the calendar system
     * @param[in] first_step the first step to include in the series or 0 to use all
     * @param[in] last_step the last step to include in the series or -1 to use all
     * @returns 0 if successfully initialized
     */
    virtual int initialize(const const_p_teca_variant_array &t,
        const std::string &units, const std::string &calendar,
        long first_step, long last_step) = 0;

    /// return true if there are more time steps in the sequence
    virtual bool is_valid() const = 0;

    /** Get the next interval in the series.
     * @param[out] first_step The first step in the next element of the series
     * @param[out] last_step The last step in the next element of the series
     * @returns 0 if successfully initialized
     */
    virtual int get_next_interval(time_point &first_step,
        time_point &last_step) = 0;

    /// @returns true if there are more intervals in the series
    operator bool() const
    {
        return this->is_valid();
    }

    /// return the first time point in the series
    const time_point &get_begin() const { return this->begin; }

    /// return the last time point in the series
    const time_point &get_end() const { return this->end; }

protected:
    const_p_teca_variant_array time;
    std::string units;
    std::string calendar;
    time_point begin;
    time_point end;
    bool valid;
};

/// Enumerate ranges of time steps bracketing seasons
/**
 * An iterator over seasons (DJF, MAM, JJA, SON) between 2 time_point's.  A
 * pair of time steps bracketing the current season are returned at each
 * iteration. Only full seasonal intervals are processed. If the input data
 * doesn't start or end on a seasonal boundary, the data from the start to the
 * first full season, and the data from the end of the last full season to the
 * end is skipped.
 */
class TECA_EXPORT season_iterator : public interval_iterator
{
public:
    season_iterator() : year(-1), month(-1) {}

    /// return true if there are more time steps in the sequence
    bool is_valid() const override;

    /** Initialize the iterator.
     *
     * @param[in] t  An array of time values
     * @param[in] units A string units of the time values
     * @param[in] calendar A string name of the calendar system
     * @param[in] first_step the first step to include in the series or 0 to use all
     * @param[in] last_step the last step to include in the series or -1 to use all
     * @returns 0 if successfully initialized
     */
    int initialize(const const_p_teca_variant_array &t,
        const std::string &units, const std::string &calendar,
        long first_step, long last_step) override;

    using interval_iterator::initialize;

    /** return a pair of time steps bracketing the current season.
     * both returned time steps belong to the current season.
     */
    int get_next_interval(time_point &first_step,
        time_point &last_step) override;

private:
    /** given a year and month, checks that the values fall on a seasonal
     * boundary. if not, returns the year and month of the start of the next
     * season.
     */
    int get_first_season(int y_in, int m_in, int &y_out, int &m_out) const;

    /** Given a year and month returns the year month and day of the end of the
     * season. the input month need not be on a seasonal boundary.
     */
    int get_season_end(int y_in, int m_in,
        int &y_out, int &m_out, int &d_out) const;

    /** Given a year and month returns the year and month of the next season.
     * the input momnth doesn't need to be on a seasonal boundary.
     */
    int get_next_season(int y_in, int m_in, int &y_out, int &m_out) const;

protected:
    int year;
    int month;
};

/// Enumerate ranges of time steps bracketing months
/** An iterator over all months between 2 time_point's. A pair
 * of time steps bracketing the current month are returned at
 * each iteration.
 */
class TECA_EXPORT year_iterator : public interval_iterator
{
public:
    year_iterator() : year(-1) {}

    /// return true if there are more time steps in the sequence
    bool is_valid() const override;

    /** Initialize the iterator.
     *
     * @param[in] t  An array of time values
     * @param[in] units A string units of the time values
     * @param[in] calendar A string name of the calendar system
     * @param[in] first_step the first step to include in the series or 0 to use all
     * @param[in] last_step the last step to include in the series or -1 to use all
     * @returns 0 if successfully initialized
     */
    int initialize(const const_p_teca_variant_array &t,
        const std::string &units, const std::string &calendar,
        long first_step, long last_step) override;

    using interval_iterator::initialize;

    /** Return a pair of time steps bracketing the current year.
     * Both returned time steps belong to the current year.
     */
    int get_next_interval(time_point &first_step,
        time_point &last_step) override;

protected:
    int year;
};

/// Enumerate ranges of time steps bracketing months
/** An iterator over all months between 2 time_point's. A pair
 * of time steps bracketing the current month are returned at
 * each iteration.
 */
class TECA_EXPORT month_iterator : public interval_iterator
{
public:
    month_iterator() : year(-1), month(-1) {}

    /// return true if there are more time steps in the sequence
    bool is_valid() const override;

    /** Initialize the iterator.
     *
     * @param[in] t  An array of time values
     * @param[in] units A string units of the time values
     * @param[in] calendar A string name of the calendar system
     * @param[in] first_step the first step to include in the series or 0 to use all
     * @param[in] last_step the last step to include in the series or -1 to use all
     * @returns 0 if successfully initialized
     */
    int initialize(const const_p_teca_variant_array &t,
        const std::string &units, const std::string &calendar,
        long first_step, long last_step) override;

    using interval_iterator::initialize;

    /** Return a pair of time steps bracketing the current month.
     * Both returned time steps belong to the current season.
     */
    int get_next_interval(time_point &first_step,
        time_point &last_step) override;

protected:
    int year;
    int month;
};

/// Enumerate ranges of time steps bracketing days
/** An iterator over all days between 2 time_point's. A pair
 * of time steps bracketing the current day are returned at
 * each iteration.
 */
class TECA_EXPORT day_iterator : public interval_iterator
{
public:
    day_iterator() : year(-1), month(-1), day(-1) {}

    /// return true if there are more time steps in the sequence
    bool is_valid() const override;

    /** Initialize the iterator.
     *
     * @param[in] t  An array of time values
     * @param[in] units A string units of the time values
     * @param[in] calendar A string name of the calendar system
     * @param[in] first_step the first step to include in the series or 0 to use all
     * @param[in] last_step the last step to include in the series or -1 to use all
     * @returns 0 if successfully initialized
     */
    int initialize(const const_p_teca_variant_array &t,
        const std::string &units, const std::string &calendar,
        long first_step, long last_step) override;

    using interval_iterator::initialize;

    /** Return a pair of time steps bracketing the current day
     * Both returned time steps belong to the current day.
     */
    int get_next_interval(time_point &first_step,
        time_point &last_step) override;

protected:
    int year;
    int month;
    int day;
};

class TECA_EXPORT n_steps_iterator : public interval_iterator
{
public:
    n_steps_iterator() : index(-1), number_of_steps(0) {}

    /// return true if there are more time steps in the sequence
    bool is_valid() const override;

    /** Initialize the iterator.
     *
     * @param[in] t  An array of time values
     * @param[in] units A string units of the time values
     * @param[in] calendar A string name of the calendar system
     * @param[in] first_step the first step to include in the series or 0 to use all
     * @param[in] last_step the last step to include in the series or -1 to use all
     * @returns 0 if successfully initialized
     */
    int initialize(const const_p_teca_variant_array &t,
        const std::string &units, const std::string &calendar,
        long first_step, long last_step) override;

    using interval_iterator::initialize;

    /** Return a pair of time steps bracketing the current block of n steps.
     * Both returned time steps belong to the current block of steps.
     */
    int get_next_interval(time_point &first_step,
        time_point &last_step) override;

    void set_number_of_steps(long n);

protected:
    long index;
    long number_of_steps;
};

class TECA_EXPORT all_iterator : public interval_iterator
{
public:
    all_iterator() : index(-1) {}

    /// return true if there are more time steps in the sequence
    bool is_valid() const override;

    /** Initialize the iterator.
     *
     * @param[in] t  An array of time values
     * @param[in] units A string units of the time values
     * @param[in] calendar A string name of the calendar system
     * @param[in] first_step the first step to include in the series or 0 to use all
     * @param[in] last_step the last step to include in the series or -1 to use all
     * @returns 0 if successfully initialized
     */
    int initialize(const const_p_teca_variant_array &t,
        const std::string &units, const std::string &calendar,
        long first_step, long last_step) override;

    using interval_iterator::initialize;

    /** Return a pair of time steps bracketing the current and only block of
     * steps.  Both returned time steps belong to the current block of steps.
     */
    int get_next_interval(time_point &first_step,
        time_point &last_step) override;

protected:
    long index;
};

using p_interval_iterator = std::shared_ptr<interval_iterator>;

/// A factory for interval_iterator
class TECA_EXPORT interval_iterator_factory
{
public:
    /** Allocate and return an instance of the named iterator
     * @param[in] interval Name of the desired interval iterator. One of daily,
     *                     monthly, seasonal, yearly, n_steps, or all
     * @returns an instance of interval_iterator
     */
    static p_interval_iterator New(const std::string &interval);

    /// The available intervals
    enum {invalid = 0, daily = 2, monthly = 3, seasonal = 4, yearly = 5, n_steps = 6, all = 7};

    /** Allocate and return an instance of the named iterator
     * @param[in] interval Id of the desired interval iterator. One of daily,
     *            monthly, seasonal, or yearly
     * @returns an instance of interval_iterator
     */
    static p_interval_iterator New(int interval);
};

}

/// send the time_point to a stream in humnan readable form
TECA_EXPORT
std::ostream &operator<<(std::ostream &os,
    const teca_calendar_util::time_point &tpt);

#endif
