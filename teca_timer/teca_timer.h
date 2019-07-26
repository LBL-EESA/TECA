#ifndef teca_timer_h
#define teca_timer_h

#include <iostream>
#include <mpi.h>


namespace teca_timer
{
	/// Initialize logging from environment variables, and/or the timer
	/// API below. This is a collective call with respect to the timer's
	/// communicator.
	///
	/// If found in the environment the following variable override the
	/// the current settings
	//
	///     TIMER_ENABLE          : integer turns on or off logging
	///     TIMER_ENABLE_SUMMARY  : integer truns on off logging in summary format
	///     TIMER_SUMMARY_MODULUS : print rank data when rank % modulus == 0
	///     TIMER_LOG_FILE        : path to write timer log to
	///     MEMPROF_LOG_FILE      : path to write memory profiler log to
	///     MEMPROF_INTERVAL      : number of seconds between memory recordings
	///
	void initialize();

	/// Finalize the log. this is where logs are written and cleanup occurs.
	/// All processes in the communicator must call, and it must be called
	/// prior to MPI_Finalize.
	void finalize();

	/// Sets the communicator for MPI calls. This must be called prior to
	/// initialization.
	/// default value: MPI_COMM_NULL
	void set_communicator(MPI_Comm comm);

	/// Sets the path to write the timer log to
	/// overriden by TIMER_LOG_FILE environment variable
	/// default value; Timer.csv
	void set_timer_log_file(const std::string &file_name);

	/// Sets the path to write the timer log to
	/// overriden by MEMPROF_LOG_FILE environment variable
	/// default value: MemProfLog.csv
	void set_mem_prof_log_file(const std::string &file_name);

	/// Sets the number of seconds in between memory use recordings
	/// overriden by MEMPROF_INTERVAL environment variable.
	void set_mem_prof_interval(int interval);

	/// Enable/Disable logging. Overriden by TIMER_ENABLE and
	/// TIMER_ENABLE_SUMMARY environment variables. In the
	/// default format a CSV file is generated capturing each ranks
	/// timer events. In the summary format a pretty and breif output
	/// is sent to the stderr stream.
	/// default value: disabled
	void enable(bool summaryFmt = false);
	void disable();

	/// return true if loggin is enabled.
	bool enabled();

	/// Sets the timer's summary log modulus. Output incudes data from
	/// MPI ranks where rank % modulus == 0 Overriden by
	/// TIMER_SUMMARY_MODULUS environment variable
	/// default value; 1000000000
	void set_summary_modulus(int modulus);

	/// @brief Log start of an event.
	///
	/// This marks the beginning of a event that must be logged.
	/// The @arg eventname must match when calling mark_end_event() to
	/// mark the end of the event.
	void mark_start_event(const char* eventname);

	/// @brief Log end of a log-able event.
	///
	/// This marks the end of a event that must be logged.
	/// The @arg eventname must match when calling mark_end_event() to
	/// mark the end of the event.
	void mark_start_event(const char* eventname);
	void mark_end_event(const char* eventname);

	/// @brief Mark the beginning of a timestep.
	///
	/// This marks the beginning of a timestep. All MarkStartEvent and
	/// MarkEndEvent after this until MarkEndTimeStep are assumed to be
	/// happening for a specific timestep and will be combined with subsequent
	/// timesteps.
	void mark_start_time_step(int timestep, double time);

	/// @brief Marks the end of the current timestep.
	void mark_end_time_step();

	/// @brief Print log to the output stream.
	///
	/// Note this triggers collective operations and hence must be called on all
	/// ranks. The amount of processes outputting can be reduced by using
	/// the moduloOuput which only outputs for (rank % moduloOutput) == 0.
	/// The default value for this is 1.
	void print_log(std::ostream& stream);

	/// MarkEvent -- A helper class that times it's life.
	/// A timer event is created that starts at the object's construction
	/// and ends at its destruction. The pointer to the event name must
	/// be valid throughout the objects life.
	class teca_mark_event
	{
		public:
	  		teca_mark_event(const char* name) : eventname(name) { mark_start_event(name); }
	  		~teca_mark_event() { mark_end_event(this->eventname); }
		private:
	  		const char *eventname;
	};
}

#endif
