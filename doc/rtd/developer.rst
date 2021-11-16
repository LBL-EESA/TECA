Development
===========

Online Source Code Documentation
--------------------------------
TECA's C++ sources are documented via Doxygen at the `TECA Doxygen site <doxygen/index.html>`_.

Class Indices
-------------

.. tip::

    The following tables contain a listing of some commonly used TECA classes. The
    `TECA Doxygen site <doxygen/index.html>`_ is a more complete reference.

.. include:: _build/rst/generated_rtd_alg.rst
.. include:: _build/rst/generated_rtd_io.rst
.. include:: _build/rst/generated_rtd_core.rst
.. include:: _build/rst/generated_rtd_data.rst

Testing
-------
TECA comes with an extensive regression test suite which can be used to validate
your build. The tests can be executed from the build directory with the ctest command.

.. code-block:: bash

   ctest --output-on-failure

Note that `PYTHONPATH`, `LD_LIBRARY_PATH` and `DYLD_LIBRARY_PATH` will need to
be set to include the build's lib directory and `PATH` will need to be set to
include ".".

Timing and Profiling
--------------------
TECA contains built in profiling mechanism which captures the run time
of each stage of a pipeline's execution and a sampling memory profiler.

The profiler records the times of user defined events and sample memory
at a user specified interval. The resulting data is written in parallel to
a CSV file in rank order. Times are stored in one file and memory use samples
in another. Each memory use sample includes the time it was taken, so that
memory use can be mapped back to corresponding events.

.. warning::
   In some cases TECA's built in profiling can negatively impact run time
   performance as the number of threads is increased. For that reason one should
   not use it in performance studies. However, it is well suited to debugging and
   diagnosing scaling issues and understanding control flow.

Compilation
~~~~~~~~~~~
The profiler is not built by default and must be compiled in by adding
`-DTECA_ENABLE_PROFILER=ON` to the CMake command line. Be sure to build in
release mode with `-DCMAKE_BUILD_TYPE=Release` and  also add `-DNDEBUG` to the
`CMAKE_CXX_FLAGS_RELEASE`. Once compiled the built in profilier may be enabled
at run time via environment variables described below or directly using its
API.

Runtime controls
~~~~~~~~~~~~~~~~
The profiler is activated by the following environment variables. Environmental
variables are parsed in `teca_profiler::initialize`. This should be
automatic in most cases as it's called from `teca_mpi_manager` which is used
by parallel TECA applications and tests.

+---------------------+---------------------------------------------------+
| Variable            | Description                                       |
+---------------------+---------------------------------------------------+
| PROFILER_ENABLE     | a binary mask that enables logging.               |
|                     | 0x01 -- event profiling enabled.                  |
|                     | 0x02 -- memory profiling enabled.                 |
+---------------------+---------------------------------------------------+
| PROFILER_LOG_FILE   | path to write timer log to                        |
+---------------------+---------------------------------------------------+
| MEMPROF_LOG_FILE    | path to write memory profiler log to              |
+---------------------+---------------------------------------------------+
| MEMPROF_INTERVAL    | float number of seconds between memory recordings |
+---------------------+---------------------------------------------------+

Visualization
~~~~~~~~~~~~~
The command line application `teca_profile_explorer` can be used to analyze the
log files. The application requires a timer profile file and a list of MPI
ranks to analyze be passed on the command line. Optionally a memory profile
file can be passed as well. For instance, the following command was used to
generate figure :numref:`prof_vis_10t_1r`.

.. code-block:: bash

   ./bin/teca_profile_explorer -e bin/test/test_bayesian_ar_detect \
      -m bin/test/test_bayesian_ar_detect_mem -r 0

When run the `teca_profile_explorer` creast an  interactive window displaying a
Gantt chart for each MPI rank.  The chart is organized with a row for each
thread. Threads with more events are displayed higher up. For each thread, and
every logged event, a colored rectangle is rendered. There can be 10's - 100's
of unique events per thread thus it is impractical to display a legend.
However, clicking on an event rectangle in the plot will result in all the data
associated with the event being printed in the terminal. If a memory profile is
passed on the command line the memory profile is normalized to the height of
the plot and shown on top of the event profile. The maximum memory use is added
to the title of the plot. Example output is shown in :numref:`prof_vis_10t_1r`.

.. _prof_vis_10t_1r:

.. figure:: ./images/tpc_rank_profile_data_0.png
   :width: 100 %
   :height: 2.5 in

   Visualization of TECA's run time profiler for the test_bayesian_ar_detect
   regression test, run with 1 MPI rank and 10 threads.


Creating PyPi Packages
----------------------
The typical sequence for pushing and testing to PyPi is as follows. Be sure to
add an rc number to the version in setup.py when testing since these are unique
and cannot be reused.

.. code-block:: bash

    python3 setup.py build_ext
    python3 setup.py install
    python3 setup.py sdist
    python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
    pip3 install --index-url https://test.pypi.org/simple/ teca


Python Coding Standard
----------------------
Match the C++ coding standard as closely as possible. Deviate from the C++
coding standard in ways described by PEP8. Use pycodestyle to check the source
code before committing.

C++ Coding Standard
-------------------
TECA has adopted the following code standard in an effort to simplify
maintenance and reduce bugs. The TECA source code ideally will look the same no
matter who wrote it. The following partial description is incomplete. When in
doubt refer to the existing code and follow the conventions there in.

Tabs, spaces, and indentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* use no tab `\t` chars
* leave no trailing white space at the end of lines and files
* indent 4 spaces
* use a single space between operators. For example:

    .. code-block:: c++

        a < b

* wrap lines at or before 80 characters.
* when wrapping assignments the = goes on the preceding line. For example:

    .. code-block:: c++

        unsigned long long a_very_lon_name =
            a_long_function_name_that_returns_a_value(foo, bar, baz);

* when wrapping conditionals logical opoerators go on the preceding line.  For
  example:

    .. code-block:: c++

        if (first_long_condition || second_long_condition ||
            third_long_condition)
        {
            // more code here
        }

Braces
~~~~~~
* use standard C++ bracing, braces go on the previous indentation level. For
  example :

    .. code-block:: c++

        if (a < b)
        {
            foo(a,b);
        }

* use braces when conditional branch takes more than one line, including when
  wrapped.
* use braces on all conditional branches in the same series when any one of the
  branches takes more than one line.

Variable, funciton, and class names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* use lower case with _ separators. This is often called "snake case"

Macros
~~~~~~
* to be definied in all caps, and do not require a semicolon. For
  example :

    .. code-block:: c++

        DO_SOMETHING(blah, blah)

* write macros so that they may be safely used in single line conditional
  branches.


Templates
~~~~~~~~~~~~~~~~~~~~~~~~
* prefer all captials for template typenames
* use TEMPLATE_DISPATCH macros for dispatching to templated data structures

Warnings
~~~~~~~~
* treat warnings as errors, developers should use -Wall or
  the compiler equivalent ( -Weverything, /W4, etc)

Conditionals
~~~~~~~~~~~~
* conditionals may omit braces, in that case the code should be on the
  following line (ie no one liners). For example :

    .. code-block:: c++

        if (a < b)
            foo(a,b);

Headers
~~~~~~~
* don't include using directives
* use include guards of the format. For example :

    .. code-block:: c++

        #ifndef file_name_h
        #define file_name_h
        // header file code here
        #endif

Classes
~~~~~~~
* use this pointer to access member variables, unless it's a private member
  variable prepended with `m_`
* use the `this` pointer to call member functions and access member variables
* use PIMPL idiom
* use `std::shared_ptr` in place of C style pointers for data structures thaat
  are costly to copy.
* use TECA's convenience macros where possible
* use const qualifiers when ever possible

Reporting errors
~~~~~~~~~~~~~~~~
* from functions return a non-zero value to indicate an error. return zero when
  the function succeeds
* use the TECA\_ERROR macro to report internals errors where it useful to
  include the call stack as context. The call stack is almost always useful
  when inside internal implementation and/or utility functions.. For example :

    .. code-block:: c++

        TECA_ERROR("message " << var)

* use the TECA\_FATAL\_ERROR macro from the `teca_algorithm` overrides.
  This macro invokes the error handler which by default aborts execution hence
  no contextual information about the call stack will be reported.. For example :

    .. code-block:: c++

        TECA_FATAL_ERROR("message " << var)

Exceptions and exception safety
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* the code is NOT designed for exception safety.
* use execptions only when absolutely necessary and the program
  needs to terminate.

Thread safety and MPI collectives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* design algorithms for thread per-pipeline invocation. this means avoiding the
  use of OpenMP for loop level parallelism.
* algorithms need to be thread safe except for the first invokation of
  `get_output_metadata`.
* MPI colectives are safe to use in the first invokation of `get_output_metadata`.

Testing
~~~~~~~
* New code will not be accepted without an accompanying regression test
* New code will not be accepted when existing tests fail
* Regression tests should prefer analytic solutions when practical
* Test data should be as small as possible.
