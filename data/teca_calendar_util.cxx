#include "teca_calendar_util.h"

#include "teca_common.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_coordinate_util.h"
#include "teca_calcalcs.h"

#include <algorithm>

#include <string>

using namespace teca_variant_array_util;


// TODO - With 23:59:59, sometimes we select the next day.  What is the right
// end-of-day time so that all use cases are satisfied without selecting the
// next day?
#define END_OF_DAY_HH 23
#define END_OF_DAY_MM 30
#define END_OF_DAY_SS 0
#define END_OF_DAY "23:30:00"

// --------------------------------------------------------------------------
std::ostream &operator<<(std::ostream &os,
    const teca_calendar_util::time_point &tpt)
{
    os << "time[" << tpt.index << "] = " << tpt.time << ", \""
        << tpt.year << "-" << tpt.month << "-"
        << tpt.day << " " << tpt.hour << ":" << tpt.minute
        << ":" << tpt.second << "\"";
    return os;
}



namespace teca_calendar_util
{

// **************************************************************************
long gregorian_number(long y, long m, long d)
{
    m = (m + 9) % 12;
    y = y - m/10;
    return 365*y + y/4 - y/100 + y/400 + (m*306 + 5)/10 + (d - 1);
}

// **************************************************************************
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

// **************************************************************************
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


/// returns one of DJF,MAM,JJA,SON based on the month passed in
const char *get_season_name(int month)
{
    if ((month == 12) || ((month >= 1) && (month <= 2)))
    {
        return "DJF";
    }
    else if ((month >= 3) && (month <= 5))
    {
        return "MAM";
    }
    else if ((month >= 6) && (month <= 8))
    {
        return "JJA";
    }
    else if ((month >= 9) && (month <= 11))
    {
        return "SON";
    }

    TECA_ERROR("Failed to get the season name for month " << month)
    return "invalid";
}

// --------------------------------------------------------------------------
time_point::time_point(long i, double t, const std::string &units,
    const std::string &calendar) : index(i), time(t), year(0), month(0),
    day(0), hour(0), minute(0), second(0)

{
    if (teca_calcalcs::date(t, &this->year, &this->month, &this->day,
        &this->hour, &this->minute, &this->second, units.c_str(),
        calendar.c_str()))
    {
        TECA_ERROR("Failed to convert the time value " << t
            << " \"" << units << "\" in the \"" << calendar << "\"")
    }
}

// --------------------------------------------------------------------------
bool season_iterator::is_valid() const
{
    if (!this->valid)
        return false;

    // get the end of the current season
    int ey = -1;
    int em = -1;
    int ed = -1;
    if (this->get_season_end(this->year, this->month, ey, em, ed))
    {
        TECA_ERROR("Failed to get season end")
        return false;
    }

    // verify that we have data for the current season
    if ((ey > this->end.year) ||
        ((ey == this->end.year) && (em > this->end.month)) ||
        ((ey == this->end.year) && (em == this->end.month) &&
        (ed > this->end.day)))
    {
        return false;
    }

    return true;
}

// --------------------------------------------------------------------------
int season_iterator::initialize(const const_p_teca_variant_array &t,
    const std::string &units, const std::string &calendar,
    long first_step, long last_step)
{
    this->time = t;
    this->units = units;
    this->calendar = calendar;

    if (t->size() == 0)
    {
        TECA_ERROR("The array of time values can't be empty")
        return -1;
    }

    if (first_step >= (long)t->size())
    {
        TECA_ERROR("first_step " << first_step
            << " output of bounds with " << t->size() << " time values")
        return -1;
    }

    if (last_step < 0)
        last_step = t->size() - 1;

    if ((last_step < first_step) || (last_step >= (long)t->size()))
    {
        TECA_ERROR("invalid last_step " << last_step << " with first_step "
            << first_step << " and " << t->size() << " time values")
        return -1;
    }

    // initialize the time range to iterate over
    VARIANT_ARRAY_DISPATCH(t.get(),
        auto [sp_t, p_t] = get_host_accessible<CTT>(t);
        sync_host_access_any(t);
        this->begin = time_point(first_step, p_t[first_step], this->units, this->calendar);
        this->end = time_point(last_step, p_t[last_step], this->units, this->calendar);
        )

    // skip ahead to the first season.
    if (this->get_first_season(this->begin.year,
        this->begin.month, this->year, this->month))
    {
        TECA_ERROR("Failed to determine the first season")
        return -1;
    }

    this->valid = true;

    return 0;
}


// --------------------------------------------------------------------------
int season_iterator::get_first_season(int y_in, int m_in, int &y_out,
    int &m_out) const
{
    if ((m_in == 12) or (m_in == 3) or (m_in == 6) or (m_in == 9))
    {
        y_out = y_in;
        m_out = m_in;
        return 0;
    }

    return this->get_next_season(y_in, m_in, y_out, m_out);
}

// --------------------------------------------------------------------------
int season_iterator::get_season_end(int y_in, int m_in, int &y_out,
    int &m_out, int &d_out) const
{
    if (m_in == 12)
    {
        y_out = y_in + 1;
        m_out = 2;
    }
    else if ((m_in >= 1) and (m_in <= 2))
    {
        y_out = y_in;
        m_out = 2;
    }
    else if ((m_in >= 3) and (m_in <= 5))
    {
        y_out = y_in;
        m_out = 5;
    }
    else if ((m_in >= 6) and (m_in <= 8))
    {
        y_out = y_in;
        m_out = 8;
    }
    else if ((m_in >= 9) and (m_in <= 11))
    {
        y_out = y_in;
        m_out = 11;
    }
    else
    {
        TECA_ERROR("Failed to get the end of the season from month "
            << m_in)
        return -1;
    }

    if (teca_calcalcs::days_in_month(this->calendar.c_str(),
        this->units.c_str(), y_out, m_out, d_out))
    {
        TECA_ERROR("Failed to get the last day of the month "
            << y_out << " " << m_out)
        return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
int season_iterator::get_next_season(int y_in, int m_in, int &y_out,
    int &m_out) const
{
    if (m_in == 12)
    {
        y_out = y_in + 1;
        m_out = 3;
    }
    else if ((m_in >= 1) and (m_in <= 2))
    {
        y_out = y_in;
        m_out = 3;
    }
    else if ((m_in >= 3) and (m_in <= 5))
    {
        y_out = y_in;
        m_out = 6;
    }
    else if ((m_in >= 6) and (m_in <= 8))
    {
        y_out = y_in;
        m_out = 9;
    }
    else if ((m_in >= 9) and (m_in <= 11))
    {
        y_out = y_in;
        m_out = 12;
    }
    else
    {
        TECA_ERROR("Failed to get the next season from m_in "
            << m_in)
        return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
int season_iterator::get_next_interval(time_point &first_step,
    time_point &last_step)
{
    // get the end of the current season
    int end_year = -1;
    int end_month = -1;
    int end_day = -1;
    if (this->get_season_end(this->year, this->month, end_year, end_month, end_day))
    {
        TECA_ERROR("Failed to get season end")
        return -1;
    }

    // verify that we have data for the current season
    if ((end_year > this->end.year) ||
        ((end_year == this->end.year) && (end_month > this->end.month)) ||
        ((end_year == this->end.year) && (end_month == this->end.month) &&
        (end_day > this->end.day)))
    {
        return -1;
    }

    // find the time step of the first day
    int sy = this->year;
    int sm = this->month;

    char t0[21] = {'\0'};
    snprintf(t0, 21, "%04d-%02d-01 00:00:00", this->year, this->month);

    unsigned long i0 = 0;
    if (teca_coordinate_util::time_step_of(this->time,
        false, true, this->calendar, this->units, t0, i0))
    {
        TECA_ERROR("Failed to get the tme step of " << t0)
        return -1;
    }

    double ti = 0.0;
    this->time->get(i0, ti);
    first_step = time_point(i0, ti, this->year, this->month);

    // find the time step of the last day
    char t1[21] = {'\0'};
    snprintf(t1, 21, "%04d-%02d-%02d " END_OF_DAY, end_year, end_month, end_day);

    unsigned long i1 = 0;
    if (teca_coordinate_util::time_step_of(this->time,
        true, true, this->calendar, this->units, t1, i1))
    {
        TECA_ERROR("Failed to get the time step of " << t1)
        return -1;
    }

    this->time->get(i1, ti);
    last_step = time_point(i1, ti, end_year, end_month,
        end_day, END_OF_DAY_HH, END_OF_DAY_MM, END_OF_DAY_SS);

    // move to next season
    if (this->get_next_season(sy, sm, this->year, this->month))
    {
        TECA_ERROR("Failed to get the next season from "
            << sy << "-" << sm)
        return -1;
    }

    if (!this->is_valid())
        this->valid = false;

    return 0;
}

// --------------------------------------------------------------------------
bool year_iterator::is_valid() const
{
    if (!this->valid)
        return false;

    // check for more months to process
    if (this->year > this->end.year)
    {
        return false;
    }

    return true;
}

// --------------------------------------------------------------------------
int year_iterator::initialize(const const_p_teca_variant_array &t,
    const std::string &units, const std::string &calendar,
    long first_step, long last_step)
{
    this->time = t;
    this->units = units;
    this->calendar = calendar;

    if (t->size() == 0)
    {
        TECA_ERROR("The array of time values can't be empty")
        return -1;
    }

    if (first_step >= (long)t->size())
    {
        TECA_ERROR("first_step " << first_step
            << " output of bounds with " << t->size() << " time values")
        return -1;
    }

    if (last_step < 0)
        last_step = t->size() - 1;

    if ((last_step < first_step) || (last_step >= (long)t->size()))
    {
        TECA_ERROR("invalid last_step " << last_step << " with first_step "
            << first_step << " and " << t->size() << " time values")
        return -1;
    }

    // current time state
    VARIANT_ARRAY_DISPATCH(t.get(),
        auto [sp_t, p_t] = get_host_accessible<CTT>(t);
        sync_host_access_any(t);
        this->begin = time_point(first_step, p_t[first_step], this->units, this->calendar);
        this->end = time_point(last_step, p_t[last_step], this->units, this->calendar);
        )

    this->valid = true;

    this->year = this->begin.year;

    return 0;
}

// --------------------------------------------------------------------------
int year_iterator::get_next_interval(time_point &first_step,
    time_point &last_step)
{
    // check for more months to process
    if (!this->is_valid())
        return -1;

    // find the time step of the first day
    char t0[21] = {'\0'};
    snprintf(t0, 21, "%04d-01-01 00:00:00", this->year);

    unsigned long i0 = 0;
    if (teca_coordinate_util::time_step_of(this->time,
        false, true, this->calendar, this->units, t0, i0))
    {
        TECA_ERROR("Failed to locate a time step for " << t0)
        return -1;
    }

    double ti = 0.0;
    this->time->get(i0, ti);
    first_step = time_point(i0, ti, this->year);

    // find the time step of the last day
    int n_days = 0;
    if (teca_calcalcs::days_in_month(this->calendar.c_str(),
        this->units.c_str(), this->year, 12, n_days))
    {
        TECA_ERROR("Failed to get the last day of the month "
            << this->year << " 12")
        return -1;
    }

    char t1[21] = {'\0'};
    snprintf(t1, 21, "%04d-12-%02d " END_OF_DAY, this->year, n_days);

    unsigned long i1 = 0;
    if (teca_coordinate_util::time_step_of(this->time,
        true, true, this->calendar, this->units, t1, i1))
    {
        TECA_ERROR("Failed to locate a time step for " << t1)
        return -1;
    }

    this->time->get(i1, ti);
    last_step = time_point(i1, ti, this->year, 12,
        n_days, END_OF_DAY_HH, END_OF_DAY_MM, END_OF_DAY_SS);

    // move to next year
    this->year += 1;

    // if we're at the  end of the sequence mark the iterator invalid
    if (!this->is_valid())
        this->valid = false;

    return 0;
}

// --------------------------------------------------------------------------
bool month_iterator::is_valid() const
{
    if (!this->valid)
        return false;

    // check for more months to process
    if ((this->year > this->end.year) ||
        ((this->year == this->end.year) &&
        (this->month > this->end.month)))
    {
        return false;
    }

    return true;
}

// --------------------------------------------------------------------------
int month_iterator::initialize(const const_p_teca_variant_array &t,
    const std::string &units, const std::string &calendar,
    long first_step, long last_step)
{
    this->time = t;
    this->units = units;
    this->calendar = calendar;

    if (t->size() == 0)
    {
        TECA_ERROR("The array of time values can't be empty")
        return -1;
    }

    if (first_step >= (long)t->size())
    {
        TECA_ERROR("first_step " << first_step
            << " output of bounds with " << t->size() << " time values")
        return -1;
    }

    if (last_step < 0)
        last_step = t->size() - 1;

    if ((last_step < first_step) || (last_step >= (long)t->size()))
    {
        TECA_ERROR("invalid last_step " << last_step << " with first_step "
            << first_step << " and " << t->size() << " time values")
        return -1;
    }

    // time point's to iterate between
    VARIANT_ARRAY_DISPATCH(t.get(),
        auto [sp_t, p_t] = get_host_accessible<CTT>(t);
        sync_host_access_any(t);
        this->begin = time_point(first_step, p_t[first_step], this->units, this->calendar);
        this->end = time_point(last_step, p_t[last_step], this->units, this->calendar);
        )

    this->valid = true;

    this->year = this->begin.year;
    this->month = this->begin.month;

    return 0;
}

// --------------------------------------------------------------------------
int month_iterator::get_next_interval(time_point &first_step,
    time_point &last_step)
{
    // check for more months to process
    if (!this->is_valid())
        return -1;

    // find the time step of the first day
    char t0[21] = {'\0'};
    snprintf(t0, 21, "%04d-%02d-01 00:00:00", this->year, this->month);

    unsigned long i0 = 0;
    if (teca_coordinate_util::time_step_of(this->time,
        false, true, this->calendar, this->units, t0, i0))
    {
        TECA_ERROR("Failed to locate a time step for " << t0)
        return -1;
    }

    double ti = 0.0;
    this->time->get(i0, ti);
    first_step = time_point(i0, ti, this->year, this->month);

    // find the time step of the last day
    int n_days = 0;
    if (teca_calcalcs::days_in_month(this->calendar.c_str(),
        this->units.c_str(), this->year, this->month, n_days))
    {
        TECA_ERROR("Failed to get the last day of the month "
            << this->year << " " << this->month)
        return -1;
    }

    char t1[21] = {'\0'};
    snprintf(t1, 21, "%04d-%02d-%02d " END_OF_DAY, this->year, this->month, n_days);

    unsigned long i1 = 0;
    if (teca_coordinate_util::time_step_of(this->time,
        true, true, this->calendar, this->units, t1, i1))
    {
        TECA_ERROR("Failed to locate a time step for " << t1)
        return -1;
    }

    this->time->get(i1, ti);
    last_step = time_point(i1, ti, this->year, this->month,
        n_days, END_OF_DAY_HH, END_OF_DAY_MM, END_OF_DAY_SS);

    // move to next month
    this->month += 1;

    // move to next year
    if (this->month == 13)
    {
        this->month = 1;
        this->year += 1;
    }

    // if we're at the  end of the sequence mark the iterator invalid
    if (!this->is_valid())
        this->valid = false;

    return 0;
}

// --------------------------------------------------------------------------
bool day_iterator::is_valid() const
{
    if (!this->valid)
        return false;

    // check for more days to process
    if ((this->year > this->end.year) ||
        ((this->year == this->end.year) && (this->month > this->end.month)) ||
        ((this->year == this->end.year) && (this->month == this->end.month) &&
        (this->day > this->end.day)))
    {
        return false;
    }

    return true;
}

// --------------------------------------------------------------------------
int day_iterator::initialize(const const_p_teca_variant_array &t,
    const std::string &units, const std::string &calendar,
    long first_step, long last_step)
{
    this->time = t;
    this->units = units;
    this->calendar = calendar;

    if (t->size() == 0)
    {
        TECA_ERROR("The array of time values can't be empty")
        return -1;
    }

    if (first_step >= (long)t->size())
    {
        TECA_ERROR("first_step " << first_step
            << " output of bounds with " << t->size() << " time values")
        return -1;
    }

    if (last_step < 0)
        last_step = t->size() - 1;

    if ((last_step < first_step) || (last_step >= (long)t->size()))
    {
        TECA_ERROR("invalid last_step " << last_step << " with first_step "
            << first_step << " and " << t->size() << " time values")
        return -1;
    }

    // current time state
    VARIANT_ARRAY_DISPATCH(t.get(),
        auto [sp_t, p_t] = get_host_accessible<CTT>(t);
        sync_host_access_any(t);
        this->begin = time_point(first_step, p_t[first_step], this->units, this->calendar);
        this->end = time_point(last_step, p_t[last_step], this->units, this->calendar);
        )

    this->valid = true;

    // current time state
    this->year = this->begin.year;
    this->month = this->begin.month;
    this->day = this->begin.day;

    return 0;
}

// --------------------------------------------------------------------------
int day_iterator::get_next_interval(time_point &first_step,
    time_point &last_step)
{
    // check for more days to process
    if (!this->is_valid())
        return -1;

    // find the time step of the first hour of the day
    char tstr0[21] = {'\0'};
    snprintf(tstr0, 21, "%04d-%02d-%02d 00:00:00",
        this->year, this->month, this->day);

    unsigned long i0 = 0;
    if (teca_coordinate_util::time_step_of(this->time,
        false, true, this->calendar, this->units, tstr0, i0))
    {
        TECA_ERROR("Failed to locate a time step for " << tstr0)
        return -1;
    }

    double ti = 0.0;
    this->time->get(i0, ti);
    first_step = time_point(i0, ti, this->year, this->month, this->day);

    // find the time step of the last hour of the day
    char t1[21] = {'\0'};
    snprintf(t1, 21, "%04d-%02d-%02d " END_OF_DAY,
        this->year, this->month, this->day);

    unsigned long i1 = 0;
    if (teca_coordinate_util::time_step_of(this->time,
        true, true, this->calendar, this->units, t1, i1))
    {
        TECA_ERROR("Failed to locate a time step for " << t1)
        return -1;
    }

    this->time->get(i1, ti);
    last_step = time_point(i1, ti, this->year, this->month,
        this->day, END_OF_DAY_HH, END_OF_DAY_MM, END_OF_DAY_SS);

    // move to next day
    int n_days = 0;
    if (teca_calcalcs::days_in_month(this->calendar.c_str(),
        this->units.c_str(), this->year, this->month, n_days))
    {
        TECA_ERROR("Failed to get the last day of the month "
            << this->year << " " << this->month)
        return -1;
    }

    this->day += 1;

    // move to next month
    if (this->day > n_days)
    {
        this->month += 1;
        this->day = 1;
    }

    // move to next year
    if (this->month == 13)
    {
        this->month = 1;
        this->year += 1;
    }

    // if we're at the  end of the sequence mark the iterator invalid
    if (!this->is_valid())
        this->valid = false;

    return 0;
}

// --------------------------------------------------------------------------
bool n_steps_iterator::is_valid() const
{
    if (!this->valid)
        return false;

    if ((unsigned long)(this->index + this->number_of_steps -1) >= this->time->size())
    {
        return false;
    }

    return true;
}

void n_steps_iterator::set_number_of_steps(long n)
{
    this->number_of_steps = n;
}

// --------------------------------------------------------------------------
int n_steps_iterator::initialize(const const_p_teca_variant_array &t,
    const std::string &units, const std::string &calendar,
    long first_step, long last_step)
{
    (void)units;
    (void)calendar;

    this->time = t;
    this->index = 0;

    if (this->number_of_steps == 0)
    {
        TECA_ERROR("Please set the number of steps")
        return -1;
    }

    if (t->size() == 0)
    {
        TECA_ERROR("The array of time values can't be empty")
        return -1;
    }

    if (first_step >= (long)t->size())
    {
        TECA_ERROR("first_step " << first_step
            << " output of bounds with " << t->size() << " time values")
        return -1;
    }

    if (last_step < 0)
        last_step = t->size() - 1;

    if ((last_step < first_step) || (last_step >= (long)t->size()))
    {
        TECA_ERROR("invalid last_step " << last_step << " with first_step "
            << first_step << " and " << t->size() << " time values")
        return -1;
    }

    this->valid = true;

    return 0;
}

// --------------------------------------------------------------------------
int n_steps_iterator::get_next_interval(time_point &first_step,
    time_point &last_step)
{
    if (!this->is_valid())
        return -1;

    unsigned long i0 = this->index;
    unsigned long i1 = this->index + this->number_of_steps - 1;

    double ti = 0.0;
    this->time->get(i0, ti);
    first_step = time_point(i0, ti);

    this->time->get(i1, ti);
    last_step = time_point(i1, ti);

    this->index = i1 + 1;

    if (!this->is_valid())
        this->valid = false;

    return 0;
}

// --------------------------------------------------------------------------
bool all_iterator::is_valid() const
{
    if (!this->valid)
        return false;

    if (this->index != 0)
    {
        return false;
    }

    return true;
}

// --------------------------------------------------------------------------
int all_iterator::initialize(const const_p_teca_variant_array &t,
    const std::string &units, const std::string &calendar,
    long first_step, long last_step)
{
    (void)units;
    (void)calendar;

    this->time = t;
    this->index = 0;

    if (t->size() == 0)
    {
        TECA_ERROR("The array of time values can't be empty")
        return -1;
    }

    if (first_step >= (long)t->size())
    {
        TECA_ERROR("first_step " << first_step
            << " output of bounds with " << t->size() << " time values")
        return -1;
    }

    if (last_step < 0)
        last_step = t->size() - 1;

    if ((last_step < first_step) || (last_step >= (long)t->size()))
    {
        TECA_ERROR("invalid last_step " << last_step << " with first_step "
            << first_step << " and " << t->size() << " time values")
        return -1;
    }

    this->valid = true;

    return 0;
}

// --------------------------------------------------------------------------
int all_iterator::get_next_interval(time_point &first_step,
    time_point &last_step)
{
    if (!this->is_valid())
        return -1;

    unsigned long i0 = this->index;
    unsigned long i1 = this->index + this->time->size() - 1;

    double ti = 0.0;
    this->time->get(i0, ti);
    first_step = time_point(i0, ti);

    this->time->get(i1, ti);
    last_step = time_point(i1, ti);

    this->index = i1;

    if (!this->is_valid())
        this->valid = false;

    return 0;
}

// --------------------------------------------------------------------------
int interval_iterator::initialize(const teca_metadata &md)
{
    return this->initialize(md, 0, -1);
}

// --------------------------------------------------------------------------
int interval_iterator::initialize(const teca_metadata &md,
    long first_step, long last_step)
{
    // get the time axis and calendar
    teca_metadata attributes;
    std::string t_var;
    teca_metadata t_atts;
    std::string calendar;
    std::string units;
    teca_metadata coords;
    p_teca_variant_array t;
    if (md.get("attributes", attributes) || md.get("coordinates", coords) ||
        !(t = coords.get("t")) || coords.get("t_variable", t_var) ||
        attributes.get(t_var, t_atts) || t_atts.get("calendar", calendar) ||
        t_atts.get("units", units))
    {
        TECA_ERROR("Failed to get the time axis from the available metadata."
            << (attributes.empty() ? "missing" : "has") << " attributes. "
            << (coords.empty() ? "missing" : "has") << " coordinates. "
            << (t_var.empty() ? "missing" : "has") << " t_variable. "
            << (t_atts.empty() ? "missing" : "has") << " time attributes. "
            << (calendar.empty() ? "missing" : "has") << " calendar ."
            << (units.empty() ? "missing" : "has") << " time units. "
            << (t ? "has" : "missing") << " time values.")
        return -1;
    }

    return this->initialize(t, units, calendar, first_step, last_step);
}

// --------------------------------------------------------------------------
p_interval_iterator interval_iterator_factory::New(const std::string &interval)
{
    if (interval == "daily")
    {
        return std::make_shared<day_iterator>();
    }
    else if (interval == "monthly")
    {
        return std::make_shared<month_iterator>();
    }
    else if (interval == "seasonal")
    {
        return std::make_shared<season_iterator>();
    }
    else if (interval == "yearly")
    {
        return std::make_shared<year_iterator>();
    }
    else if (interval == "n_steps")
    {
        return std::make_shared<n_steps_iterator>();
    }
    else if (interval == "all")
    {
        return std::make_shared<all_iterator>();
    }

    TECA_ERROR("Failed to construct a \""
        << interval << "\" interval iterator")
    return nullptr;
}

// --------------------------------------------------------------------------
p_interval_iterator interval_iterator_factory::New(int interval)
{
    if (interval == daily)
    {
        return std::make_shared<day_iterator>();
    }
    else if (interval == monthly)
    {
        return std::make_shared<month_iterator>();
    }
    else if (interval == seasonal)
    {
        return std::make_shared<season_iterator>();
    }
    else if (interval == yearly)
    {
        return std::make_shared<year_iterator>();
    }
    else if (interval == n_steps)
    {
        return std::make_shared<n_steps_iterator>();
    }
    else if (interval == all)
    {
        return std::make_shared<all_iterator>();
    }

    TECA_ERROR("Failed to construct a \""
        << interval << "\" interval iterator")
    return nullptr;
}
}
