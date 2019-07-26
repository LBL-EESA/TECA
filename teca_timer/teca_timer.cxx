#include "teca_timer.h"
#include "teca_memory_profiler.h"

#if defined(TECA_HAS_VTK)
#include <vtksys/SystemInformation.hxx>
#endif

#include <sys/time.h>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdint.h>
#include <strings.h>
#include <cstdlib>
#include <cstdlib>

#include <map>
#include <list>
#include <vector>
#include <iomanip>
#include <limits>


extern "C"
void mpi_error_handler(MPI_Comm *comm, int *code, ...)
{
    int rank = 0;
    MPI_Comm_rank(*comm, &rank);

    int estrLen = 0;
    char estr[MPI_MAX_ERROR_STRING] = {'\0'};
    MPI_Error_string(*code, estr, &estrLen);

    std::ostringstream oss;
    #if defined(TECA_HAS_VTK)
    oss
        << "+--------------------------------------------------------+" << std::endl
        << "MPI rank " << rank  << " encountered error " << *code << std::endl
        << std::endl
        << estr << std::endl
        << std::endl
        << vtksys::SystemInformation::GetProgramStack(3,0)
        << "+--------------------------------------------------------+" << std::endl;
    #else
    oss
        << "+--------------------------------------------------------+" << std::endl
        << "MPI rank " << rank  << " encountered error " << *code << std::endl
        << std::endl
        << estr << std::endl
        << std::endl
        << "+--------------------------------------------------------+" << std::endl;
    #endif

    std::cerr << oss.str() << std::endl;

    MPI_Abort(*comm, -1);
}

namespace teca_timer
{
    namespace impl
    {
        // return high res system time relative to system epoch
        static double get_system_time()
        {
            struct timeval tv;
            gettimeofday(&tv, nullptr);
            return tv.tv_sec + tv.tv_usec/1.0e6;
        }

        // helper for pretty printing events
        struct indent
        {
            int count;

            explicit indent(int ind=0): count(ind) {}

            indent get_next_indent() const
            { return indent(this->count+1); }
        };

        std::ostream &operator<<(std::ostream &os, const indent &i)
        {
            for (int cc=0; cc < i.count; ++cc)
                os << "  ";

            return os;
        }

        // container for data captured in a timing event
        struct event
        {
            event();

            bool empty() const
            { return this->count < 1; }

            // merges or computes summary of the two events
            void add_to_summary(const event& other);

            // prints the log in a human readbale format
            void pretty_print(std::ostream& stream, indent ind) const;

            // serializes the event in CSV format into the stream.
            void to_stream(std::ostream &str) const;

            enum { START=0, END=1, DELTA=2 }; // record fields
            enum { MIN=0, MAX=1, SUM=2 };        // summary fields

            // user provided identifier for the record
            std::string name;

            // for nesting of events
            std::list<event> sub_events;

            // event duration, initially start time, end time and duration
            // are recorded. when summarizing this contains min,max and sum
            // of the summariezed set of events
            double time[3];

            // number of events in the summary, or 1 for a single event
            int count;
        };

        // timer controls and data
        static MPI_Comm comm = MPI_COMM_NULL;
        static bool logging_enabled = false;
        static bool summarize = false;
        static int summary_modulus = 100000000;
        static std::string timer_log_file = "Timer.csv";

        static std::list<event> mark;
        static std::list<event> global_events;

        static int active_time_step = -1;
        static double active_time = 0.0;

        // memory profiler
        static teca_memory_profiler mem_prof;

        // --------------------------------------------------------------------------
        event::event() : count(0)
        {
            this->time[0] = this->time[1] = this->time[2] = 0;
        }

        // --------------------------------------------------------------------------
        void event::add_to_summary(const event& other)
        {
            // convert or add to summary
            // first three cases handle conversion of this or other or both
            // into the summary form. the last case merges the summaries
            if ((this->count == 1) && (other.count == 1))
            {
                this->time[MIN] = std::min(this->time[DELTA], other.time[DELTA]);
                this->time[MAX] = std::max(this->time[DELTA], other.time[DELTA]);
                this->time[SUM] = this->time[DELTA] + other.time[DELTA];
            }
            else if (this->count == 1)
            {
                this->time[MIN] = std::min(this->time[DELTA], other.time[MIN]);
                this->time[MAX] = std::max(this->time[DELTA], other.time[MAX]);
                this->time[SUM] = this->time[DELTA] + other.time[SUM];
            }
            else if (other.count == 1)
            {
                this->time[MIN] = std::min(this->time[MIN], other.time[DELTA]);
                this->time[MAX] = std::max(this->time[MAX], other.time[DELTA]);
                this->time[SUM] = this->time[SUM] + other.time[DELTA];
            }
            else
            {
                this->time[MIN] = std::min(this->time[MIN], other.time[MIN]);
                this->time[MAX] = std::max(this->time[MAX], other.time[MAX]);
                this->time[SUM] += other.time[SUM];
            }

            this->count += other.count;

            // process nested events
            if (this->sub_events.size() == other.sub_events.size())
            {
                auto it = this->sub_events.begin();
                auto end = this->sub_events.end();

                auto oit = other.sub_events.begin();

                for (; it != end; ++it, ++oit)
                    it->add_to_summary(*oit);
            }
        }

        //-----------------------------------------------------------------------------
        void event::to_stream(std::ostream &str) const
        {
            #ifndef NDEBUG
            if (this->empty())
            {
                std::cerr << "Empty event detected" << std::endl;
                abort();
            }
            #endif

            int rank = 0;
            MPI_Comm_rank(impl::comm, &rank);

            str << rank << ", \"" << this->name << "\", " << this->time[START]
                << ", " << this->time[END] << ", " << this->time[DELTA] << std::endl;

            // handle nested events
            auto iter = this->sub_events.begin();
            auto end = this->sub_events.end();

            for (; iter != end; ++iter)
                iter->to_stream(str);
        }


        //-----------------------------------------------------------------------------
        void event::pretty_print(std::ostream& stream, indent ind) const
        {
            #ifndef NDEBUG
            if (this->empty())
            {
                std::cerr << "Empty event detected" << std::endl;
                abort();
            }
            #endif

            if (this->count == 1)
            {
                stream << ind << this->name
                    << " = (" << this->time[DELTA] <<  " s)" << std::endl;
            }
            else
            {
                stream << ind << this->name << " = ( min: "
                    << this->time[MIN] << " s, max: " << this->time[MAX]
                    << " s, avg:" << this->time[SUM]/this->count << " s )" << std::endl;
            }

            // handle nested events
            auto iter = this->sub_events.begin();
            auto  end = this->sub_events.end();

            for (; iter != end; ++iter)
                iter->pretty_print(stream, ind.get_next_indent());
        }

        //-----------------------------------------------------------------------------
        void print_summary(std::ostream& stream, indent ind)
        {
            auto iter = global_events.begin();
            auto end = global_events.end();

            for (; iter != end; ++iter)
                iter->pretty_print(stream, ind);
        }

        //-----------------------------------------------------------------------------
        void print_summary(std::ostream& stream)
        {
            if (!impl::logging_enabled)
                return;

            int nprocs = 1;
            int rank = 0;

            MPI_Comm_size(impl::comm, &nprocs);
            MPI_Comm_rank(impl::comm, &rank);

            std::ostringstream tmp;

            std::ostream &output = (rank == 0)? stream : tmp;
            if (rank == 0)
                output << "\n"
                    << "=================================================================\n"
                    << "  Time/Memory log (rank: 0) \n"
                    << "  -------------------------------------------------------------\n";

            if (rank % impl::summary_modulus == 0)
                impl::print_summary(output, impl::indent());

            if (rank == 0)
                output << "=================================================================\n";


            if (nprocs == 1)
                return;

            std::string data = tmp.str();
            int mylength = static_cast<int>(data.size()) + 1;
            std::vector<int> all_lengths(nprocs);
            MPI_Gather(&mylength, 1, MPI_INT, &all_lengths[0], 1, MPI_INT, 0, impl::comm);
            if (rank == 0)
            {
                std::vector<int> recv_offsets(nprocs);
                for (int cc=1; cc < nprocs; cc++)
                {
                    recv_offsets[cc] = recv_offsets[cc-1] + all_lengths[cc-1];
                }
                char* recv_buffer = new char[recv_offsets[nprocs-1] + all_lengths[nprocs-1]];
                MPI_Gatherv(const_cast<char*>(data.c_str()), mylength, MPI_CHAR,
                    recv_buffer, &all_lengths[0], &recv_offsets[0], MPI_CHAR, 0, impl::comm);

                for (int cc=1; cc < nprocs; cc++)
                {
                    if (cc % impl::summary_modulus == 0)
                    {
                        output << "\n"
                            << "=================================================================\n"
                            << "  Time/Memory log (rank: " << cc << ") \n"
                            << "  -------------------------------------------------------------\n";
                        output << (recv_buffer + recv_offsets[cc]);
                        output << "=================================================================\n";
                    }
                }

                delete []recv_buffer;
            }
            else
            {
                MPI_Gatherv(const_cast<char*>(data.c_str()), mylength, MPI_CHAR,
                    NULL, NULL, NULL, MPI_CHAR, 0, impl::comm);
            }
        }

    }

    // ----------------------------------------------------------------------------
    void set_communicator(MPI_Comm comm)
    {
        if (impl::comm != MPI_COMM_NULL)
            MPI_Comm_free(&impl::comm);

        // install an error handler
        MPI_Errhandler meh;
        MPI_Comm_create_errhandler(mpi_error_handler, &meh);
        MPI_Comm_set_errhandler(comm, meh);

        MPI_Comm_dup(comm, &impl::comm);
    }

    // ----------------------------------------------------------------------------
    void set_summary_modulus(int modulus)
    {
        impl::summary_modulus = modulus;
    }

    // ----------------------------------------------------------------------------
    void set_timer_log_file(const char *file)
    {
        impl::timer_log_file = file;
    }

    // ----------------------------------------------------------------------------
    void set_mem_prof_log_file(const char *file)
    {
        impl::mem_prof.set_filename(file);
    }

    // ----------------------------------------------------------------------------
    void set_mem_prof_interval(int interval)
    {
        impl::mem_prof.set_interval(interval);
    }

    // ----------------------------------------------------------------------------
    void initialize()
    {
        // always use isolated comm space
        if (impl::comm == MPI_COMM_NULL)
            teca_timer::set_communicator(MPI_COMM_WORLD);

        impl::mem_prof.set_communicator(impl::comm);

        // look for overrides in the environment
        char *tmp = nullptr;
        if ((tmp = getenv("TIMER_ENABLE")))
        {
            impl::logging_enabled = atoi(tmp);
            impl::summarize = false;
        }

        if ((tmp = getenv("TIMER_ENABLE_SUMMARY")))
        {
            impl::logging_enabled = atoi(tmp);
            impl::summarize = impl::logging_enabled;
        }

        if ((tmp = getenv("TIMER_SUMMARY_MODULUS")))
        {
            impl::summary_modulus = atoi(tmp);
        }

        if ((tmp = getenv("TIMER_LOG_FILE")))
        {
            impl::timer_log_file = tmp;
        }

        if ((tmp = getenv("MEMPROF_LOG_FILE")))
        {
            impl::mem_prof.set_filename(tmp);
        }

        if ((tmp = getenv("MEMPROF_INTERVAL")))
        {
            impl::mem_prof.set_interval(atof(tmp));
        }

        if (impl::logging_enabled && !impl::summarize)
        {
            impl::mem_prof.initialize();
        }

        #if defined(TECA_HAS_VTK)
        // enable diagnostic info about crashes
        vtksys::SystemInformation::SetStackTraceOnError(1);
        #endif

        // report what options are in use
        int rank = 0;
        MPI_Comm_rank(impl::comm, &rank);

        if ((rank == 0) && impl::logging_enabled)
            std::cerr << "Timer configured with logging "
                << (impl::logging_enabled ? "enabled" : "disabled")
                << ", summarize events " << (impl::summarize ? "on" : "off")
                << ", summary modulus " << impl::summary_modulus << ", " << std::endl
                << "timer log file \"" << impl::timer_log_file
                << "\", memory profiler log file \"" << impl::mem_prof.get_filename()
                << "\", sampling interval " << impl::mem_prof.get_interval()
                << " seconds" << std::endl;
    }

    // ----------------------------------------------------------------------------
    void finalize()
    {
        if (impl::logging_enabled)
        {
            // output timer log
            if (impl::summarize)
            {
                // pretty print to the termninal
                impl::print_summary(std::cerr);
            }
            else
            {
                int rank = 0;
                int n_ranks = 1;

                MPI_Comm_rank(impl::comm, &rank);
                MPI_Comm_size(impl::comm, &n_ranks);

                // serialize the logged events in CSV format
                std::ostringstream oss;
                oss.precision(std::numeric_limits<double>::digits10 + 2);
                oss.setf(std::ios::scientific, std::ios::floatfield);

                if (rank == 0)
                    oss << "# rank, name, start time, end time, delta" << std::endl;

                std::list<impl::event>::iterator iter = impl::global_events.begin();
                std::list<impl::event>::iterator end = impl::global_events.end();

                for (; iter != end; ++iter)
                    iter->to_stream(oss);

                // compute the file offset
                long n_bytes = oss.str().size();

                std::vector<long> gsizes(n_ranks);
                gsizes[rank] = n_bytes;

                MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                    gsizes.data(), 1, MPI_LONG, impl::comm);

                long offset = 0;
                for (int i = 0; i < rank; ++i)
                    offset += gsizes[i];

                long file_size = 0;
                for (int i = 0; i < n_ranks; ++i)
                    file_size += gsizes[i];

                // write the buffer
                MPI_File fh;
                MPI_File_open(impl::comm, impl::timer_log_file.c_str(),
                    MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

                MPI_File_set_view(fh, offset, MPI_BYTE, MPI_BYTE,
                    "native", MPI_INFO_NULL);

                MPI_File_write(fh, oss.str().c_str(), n_bytes,
                    MPI_BYTE, MPI_STATUS_IGNORE);

                MPI_File_set_size(fh, file_size);

                MPI_File_close(&fh);
            }

            // free up resources
            impl::global_events.clear();
            impl::mark.clear();

            // output the memory use profile and clean up resources
            if (impl::logging_enabled && !impl::summarize)
            {
                impl::mem_prof.finalize();
            }
        }

        // free up other resources
        MPI_Comm_free(&impl::comm);
    }

    //-----------------------------------------------------------------------------
    bool enabled()
    {
        return impl::logging_enabled;
    }

    //-----------------------------------------------------------------------------
    void enable(bool short_format)
    {
        impl::logging_enabled = true;
        impl::summarize = short_format;
    }

    //-----------------------------------------------------------------------------
    void disable()
    {
        impl::logging_enabled = false;
        impl::summarize = false;
    }

    //-----------------------------------------------------------------------------
    void mark_start_event(const char* eventname)
    {
        if (impl::logging_enabled)
        {
            #ifndef NDEBUG
            if (!eventname)
            {
                std::cerr << "null eventname detected. events must be named." << std::endl;
                abort();
            }
            #endif

            impl::event evt;
            evt.name = eventname;
            evt.time[impl::event::START] = impl::get_system_time();
            evt.count = 1;

            impl::mark.push_back(evt);
        }
    }

    //-----------------------------------------------------------------------------
    void mark_end_event(const char* eventname)
    {
        if (impl::logging_enabled)
        {
            impl::event evt = impl::mark.back();

            #ifdef NDEBUG
            (void)eventname;
            #else
            if (!eventname)
            {
                std::cerr << "null eventname detected. events must be named." << std::endl;
                abort();
            }

            if (strcmp(eventname, evt.name.c_str()) != 0)
            {
                std::cerr << "Mismatched mark_start_event/mark_end_event. Expecting: '"
                    << evt.name.c_str() << "' Got: '" << eventname << "'" << std::endl;
                abort();
            }
            #endif

            evt.time[impl::event::END] = impl::get_system_time();
            evt.time[impl::event::DELTA] = evt.time[impl::event::END] - evt.time[impl::event::START];

            impl::mark.pop_back();

            // handle event nesting
            if (impl::mark.empty())
            {
                impl::global_events.push_back(evt);
            }
            else
            {
                impl::mark.back().sub_events.push_back(evt);
            }
        }
    }

    //-----------------------------------------------------------------------------
    void mark_start_time_step(int timestep, double time)
    {
        impl::active_time_step = timestep;
        impl::active_time = time;

        std::ostringstream mk;
        mk << "timestep: " << impl::active_time_step << " time: " << impl::active_time;
        mark_start_event(mk.str().c_str());
    }

    //-----------------------------------------------------------------------------
    void mark_end_time_step()
    {
        std::ostringstream mk;
        mk << "timestep: " << impl::active_time_step << " time: " << impl::active_time;
        mark_end_event(mk.str().c_str());

        std::list<impl::event> &active_event_list =
            impl::mark.empty() ? impl::global_events : impl::mark.back().sub_events;

        // merge with previous timestep.
        if (impl::summarize && (active_event_list.size() >= 2))
        {
            std::list<impl::event>::reverse_iterator iter = active_event_list.rbegin();
            impl::event& cur = *iter;
            ++iter;

            impl::event& prev = *iter;
            if (strncmp(prev.name.c_str(), "timestep:", 9) == 0)
            {
                prev.add_to_summary(cur);

                std::ostringstream summary_label;
                summary_label << "timestep: (summary over " << prev.count << " timesteps)";
                prev.name = summary_label.str();
                active_event_list.pop_back();
            }
        }

        impl::active_time_step = -1;
        impl::active_time = 0.0;
    }

}
