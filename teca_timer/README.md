# Timing and Profiling
The timer library record the times of user defined events and sample memory
ata user specified interval. The resulting data is written in paralel to
a CSV file in rank order. Times are stored in one file and memory use samples
in another. Each memory use sample includes the time it was taken, so that
memory use can be mapped back to corresponding events.


## Runtime controls

One can use the Timer API defined in `teca_timer.h` or the following environment variables
which are accessed during `teca_timer::initialize`

| Variable         | Description                                       |
|------------------|---------------------------------------------------|
| TIMER_ENABLE     | integer turns on or off logging                   |
| TIMER_LOG_FILE   | path to write timer log to                        |
| MEMPROF_LOG_FILE | path to write memory profiler log to              |
| MEMPROF_INTERVAL | float number of seconds between memory recordings |

## Usage

```C++
#include <teca_timer.h>

void myfunc()
{
    // a scoped timer, times its life span
    teca_timer::mark_event("myfunc");
    ...
}

int main(int ac, char** av)
{
    MPI_Init(&ac, &av);
    teca_timer::initialize();
    ...
    // explicit timed event
    teca_timer::mar_start_event("event_1");
    ...
    teca_timer::mark_end_event("event_1");
    ...
    myfunc();
    ...
    // another explicit timed event
    teca_timer::mar_start_event("event_2");
    ...
    teca_timer::mark_end_event("event_2");
    ...
    teca_timer::finalize();
    MPI_Finalize();
    return 0;
}

```






