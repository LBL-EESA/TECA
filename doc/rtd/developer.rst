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


Data Model
----------
TECA's data model is an essential part of its pipeline based design. The data
model enables normalized transfer of data in between pipeline stages. All data
that flows through TECA's pipeline is stored in arrays.  Datasets are high
level collections of arrays that provide context for structured access.
Metadata is stored in dictionary like data structure that associates names with
values or arrays.

Data sets
~~~~~~~~~
Datasets are collections of arrays that have specific meaning when grouped
together.

Metadata
~~~~~~~~
Metadata is dictionary like structure that maps names to values or arrays.


Arrays
~~~~~~
All data in TECA is stored in arrays. Both data sets and metadata are
collections of arrays.
The core data structure in TECA is the `teca_variant_array`_. This is
a polymorphic base class that defines core API for dealing with array
based data. It also serves as a type erasure that is the basis for
collections of arrays of different data types. Type specific derived classes are templated
instantiations of the `teca_variant_array_impl`_ class.

A design goal of TECA is to achieve high performance to that end TECA does not
typically restrict or prescribe the type of data to be processed. Instead the
native data type is used. For instance if data loaded from disk has a single
precision floating point data type, this data type is maintained as the data
moves through a pipeline and this data type is used for the results of
calculations. However, in some situations a high or low precision is useful and/or necessary.
In those circumstances TECA may change the data type as needed.
For these reasons when writing data processing code one must not hard code
for a specific type and instead write generic code that can handle any of the
supported types.

Additionally, TECA's load balancing system dynamically executes codes on both the CPU and
GPU. Thus one must not make assumptions about the location of data input to a
pipeline stage. Instead one must write generic code that can process data on
either the CPU or GPU.

The following subsections describe some of the details of how one achieves
generic type and device agnostic pipeline stage implementations.

Determining the type of an array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once an array is retrieved from a dataset its type must be determined and it
must be cast into the appropriate type before its elements may be accessed. The
approach we take is to try the polymorphic cast to each supported type, when
the cast succeeds we know the type, and then array elements may be accessed.
The approach is illustrated in the following code defining a function that adds
two arrays and returns the result.

.. code-block:: c++
   :linenos:
   :caption: Manual type selection.

    p_teca_variant_array add_cpu(const const_p_teca_variant_array &a1,
                                 const const_p_teca_variant_array &a2)
    {
        size_t n_elem = a1->size();

        // allocate the output
        p_teca_variant_array ao = a1->new_instance(n_elem);

        if (dynamic_cast<teca_double_array*>(a1.get())
        {
            // cast output and get pointer to its elements
            ...
            // cast inputs and get pointers to their elements
            ...

            // do the calculation
            for (size_t i = 0; i < n_elem; ++i)
                pao[i] = pa1[i] + pa2[i];

            // return the result
            return ao;
        }
        else if (dynamic_cast<teca_float_array*>(a1.get())
        {
            // cast output and get pointer to its elements
            ...
            // cast inputs and get pointers to their elements
            ...

            // do the calculation
            for (size_t i = 0; i < n_elem; ++i)
                pao[i] = pa1[i] + pa2[i];

            // return the result
            return ao;
        }
        else if (dynamic_cast<teca_long_array*>(a1.get())
        {
            // cast output and get pointer to its elements
            ...
            // cast inputs and get pointers to their elements
            ...

            // do the calculation
            for (size_t i = 0; i < n_elem; ++i)
                pao[i] = pa1[i] + pa2[i];

            // return the result
            return ao;
        }
        // additional cases for all supported types
        ...

        TECA_ERROR("Failed to add the arrays unsupported type")
        return nullptr;
    }


We've left some of the details out for now to keep the snippet short. We'll
fill these details in shortly. Once the inputs' types are known we down cast,
access the elements, and perform the calculation.

Notice that the code is nearly identical for each different type. Because TECA
supports all ISO C++ floating point and integer types as well as a number of
higher level types such as std::string, this approach would be quite cumbersome
to use in practice if it could not be automated. TECA provides the
`VARIANT_ARRAY_DISPATCH` macro to automate type selection and apply a snippet
of code for each supported type. Here is the example modified to use
the dispatch macros.


.. code-block:: c++
   :linenos:
   :caption: Automating type selection with the VARIANT_ARRAY_DISPATCH macro.

    p_teca_variant_array add_cpu(const const_p_teca_variant_array &a1,
                                 const const_p_teca_variant_array &a2)
    {
        size_t n_elem = a1->size();

        // allocate the output
        p_teca_variant_array ao = a1->new_instance(n_elem);

        // select array type and apply code to each supported type
        VARIANT_ARRAY_DISPATCH(a1.get(),

            // cast output and get pointer to its elements
            ...
            // cast inputs and get pointers to their elements
            ...

            // do the calculation
            for (size_t i = 0; i < n_elem; ++i)
                pao[i] = pa1[i] + pa2[i];

            // return the result
            return ao;
            )

        TECA_ERROR("Failed to add the arrays unsupported type")
        return nullptr;
    }


Notice, that the dispatch macros have substantially simplified the type
selection process. The macro expands into a sequence of conditions one for
each supported type with the user provided code applied to each type.

When using the dispatch macros, in order to perform the casts and declare
pointers we need to know the array's type.  This is done through a series of
type aliases that are defined inside each expansion of the macro. The following
aliases give us the detected element type as well as a number of other
useful compound types derived from it.

+-----------+---------------------------------------------------+
| **Alias** | **Description**                                   |
+-----------+---------------------------------------------------+
| NT        | The array's element type                          |
+-----------+---------------------------------------------------+
| TT        | The variant array type                            |
+-----------+---------------------------------------------------+
| CTT       | The const form of the variant array type          |
+-----------+---------------------------------------------------+
| PT        | The type of variant array pointer                 |
+-----------+---------------------------------------------------+
| CPT       | The const form type of variant array pointer      |
+-----------+---------------------------------------------------+
| SP        | The element shared pointer type                   |
+-----------+---------------------------------------------------+
| CSP       | The const form of the element shared pointer type |
+-----------+---------------------------------------------------+

Read only access to an array's elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once the type of the array is known, the array will be cast to that type, its
elements will be accessed, and the calculations made. The variant array
implements so called `accessibility` methods that are used to access an array's
elements when the location of the data, either CPU or GPU, is unknown. This is
the case when dealing with the inputs to a pipeline stage.  The accessibility
methods declare where, either CPU or GPU, we intend to perform the calculations
and internally will move data if it is not already present in the declared
location. The accessibility methods are
`teca_variant_array_impl::get_host_accessible` and
`teca_variant_array_impl::get_cuda_accessible`.

When data is moved by the accessibility methods, temporary buffers are
employed. The use of temporaries has two important consequences when writing
data processing code. First, the accessibility functions return smart pointers.
The returned smart pointer must be held for as long as we need access to the
array's elements. Once the smart pointer is released, the array's elements are
no longer safe to access, and doing so may result in a nasty crash. Second,
data accessed through the accessibility methods should not be modified. This is
because when a temporary has been used the changes will be lost. Getting write
access to an array's elements is discussed below.

The `teca_variant_array_util` defines convenience accessibility functions that
simplify use in the most common scenarios.  These functions do the following:
perform a cast to a typed variant array, call array's the accessibility method,
and get a read only pointer to the array's elements for use in calculations.


.. code-block:: c++
   :linenos:
   :caption: Read only access to input arrays through the accessibility functions.

    p_teca_variant_array add_cpu(const const_p_teca_variant_array &a1,
                                 const const_p_teca_variant_array &a2)
    {
        size_t n_elem = a1->size();

        // allocate the output
        p_teca_variant_array ao = a1->new_instance(n_elem);

        // select array type and apply code to each supported type
        VARIANT_ARRAY_DISPATCH(a1.get(),

            // cast output and get pointer to its elements
            ...

            // cast inputs and get pointers to their elements
            assert_type<TT>(a2);
            auto [spa1, pa1] = get_host_accessible<TT>(a1);
            auto [spa2, pa1] = get_host_accessible<TT>(a2);

            // do the calculation
            for (size_t i = 0; i < n_elem; ++i)
                pao[i] = pa1[i] + pa2[i];

            // return the result
            return ao;
            )

        TECA_ERROR("Failed to add the arrays unsupported type")
        return nullptr;
    }

Here we've added calls to `get_host_accessible` function which ensures the data
as accessible on the CPU and returns a smart pointer managing the life span of
any temporaries that were needed, as well as a naked pointer which can be used
to access the array's elements.  Because we are assuming that both of the input
arrays have the same type we've also called the convenience method,
`assert_type`,  to verify that the type of the second array is indeed the same
as the first.  It is often the case that we will need to deal with multiple
input types. This is described below.

Write access to an array's elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The results of our calculations are returned in arrays we allocate.  Because we
have allocated the array ourselves, and have declared where it will be
allocated (CPU or GPU), there is no need for data movement or temporary
buffers. Instead we can access the pointer to the array's memory directly
using the `teca_variant_array_impl::data` method.

Direct access should be preferred when ever we are certain of an
array's location. This is because while the smart pointers returned by the
accessibility methods used to manage the life span of temporaries are relatively
cheap, they have overheads including the smart pointer's constructor
calls to malloc, which synchronizes threads due to use of locks in many
malloc implementations. Note: Google's tcmalloc library can provide increased
performance for multithreaded codes such as TECA and is recommended.


.. code-block:: c++
   :linenos:
   :caption: Accessing outputs with write access.

    p_teca_variant_array add_cpu(const const_p_teca_variant_array &a1,
                                 const const_p_teca_variant_array &a2)
    {
        size_t n_elem = a1->size();

        // allocate the output
        p_teca_variant_array ao = a1->new_instance(n_elem);

        // select array type and apply code to each supported type
        VARIANT_ARRAY_DISPATCH(a1.get(),

            // cast output and get pointer to its elements
            auto [pao] = data<TT>(ao);

            // cast inputs and get pointers to their elements
            assert_type<CTT>(a2);
            auto [spa1, pa1] = get_host_accessible<CTT>(a1);
            auto [spa2, pa1] = get_host_accessible<CTT>(a2);

            // do the calculation
            for (size_t i = 0; i < n_elem; ++i)
                pao[i] = pa1[i] + pa2[i];

            // return the result
            return ao;
            )

        TECA_ERROR("Failed to add the arrays unsupported type")
        return nullptr;
    }

We've added the `data` calls to get writable pointer to the output array's
elements. With this the code is finally complete. The above example serves as a
template for new data processing implementations.

Accessing data for use on the GPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TECA's load balancers automatically assign each thread a CPU core or a CPU core
and a GPU. When assigned a GPU it is important that calculations be made using
the assigned device. Here we show the changes needed to calculate on the GPU.
Generally speaking the steps needed to access data on the GPU are to:

# Activate the assigned device.
# Declare that we will access input data on the GPU. Data will be moved for us if needed.
# Allocate the output data on the assigned device and get a raw pointer to it.
# launch the kernel that performs the calculation.

Here is the example modified for calculation on the GPU.


.. code-block:: c++
   :linenos:
   :caption: Accessing data on the GPU.

    p_teca_variant_array add_cuda(const const_p_teca_variant_array &a1,
                                  const const_p_teca_variant_array &a2)
    {
        size_t n_elem = a1->size();

        // allocate the output
        p_teca_variant_array ao = a1->new_instance(n_elem, allocator::cuda_async);

        // select array type and apply code to each supported type
        VARIANT_ARRAY_DISPATCH(a1.get(),

            // cast output and get pointer to its elements
            auto [pao] = data<TT>(ao);

            // cast inputs and get pointers to their elements
            assert_type<CTT>(a2);
            auto [spa1, pa1] = get_cuda_accessible<CTT>(a1);
            auto [spa2, pa1] = get_cuda_accessible<CTT>(a2);

            // launch the kernel to do the calculation
            ...

            // return the result
            return ao;
            )

        TECA_ERROR("Failed to add the arrays unsupported type")
        return nullptr;
    }


The main changes here are specifying a GPU allocator, calling the GPU
accessibility methods, and launching a CUDA kernel.

Dealing with multiple input array types
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In many cases a calculation will need to access multiple arrays, all of which
could potentially have a different data type. One common scenario is needing
to access mesh coordinates as well as node centered data.
The `NESTED_VARIANT_ARRAY_DISPATCH` macros can be used in these situations.
The nested dispatch macros use token pasting so that the aliases are given
unique names such that the macros can be nested. The following code snippet taken
from the `teca_vorticity` algorithm illustrates.


.. code-block:: c++
   :linenos:
   :caption: Accessing data with multiple types.

    // allocate the output array
    p_teca_variant_array vort = comp_0->new_instance(comp_0->size());

    // compute vorticity
    NESTED_VARIANT_ARRAY_DISPATCH_FP(
        lon.get(), _COORD,

        assert_type<TT_COORD>(lat);

        auto [sp_lon, p_lon,
              sp_lat, p_lat] = get_host_accessible<CTT_COORD>(lon, lat);

        NESTED_VARIANT_ARRAY_DISPATCH_FP(
            comp_0.get(), _DATA,

            assert_type<TT_DATA>(comp_1);

            auto [sp_comp_0, p_comp_0,
                  sp_comp_1, p_comp_1] = get_host_accessible<CTT_DATA>(comp_0, comp_1);

            auto [p_vort] = data<TT_DATA>(vort);

            ::vorticity(p_vort, p_lon, p_lat,
                p_comp_0, p_comp_1, lon->size(), lat->size());
            )
        )


In the nested dispatch macros an additional parameter specifying an identifier
that is pasted onto the usual aliases must be provided. The identifier `_COORD` is passed to
the first dispatch macro where we select the data type of
mesh coordinate arrays. The identifier `_DATA` is is passed to the second dispatch macro where
we select the data type of the wind variable.


Environment Variables
---------------------
A number of environment variables can be used to modify the runtime behavior of
the system. Generally these must be set prior to the application starting. In
some cases these are useful in Python scripts as well.

+-------------------------+---------------------------------------------------+
| Variable                | Description                                       |
+-------------------------+---------------------------------------------------+
| TECA_THREADS_PER_DEVICE | The number of CPU threads serving data to each    |
|                         | GPU in the system. The default value is 8.        |
+-------------------------+---------------------------------------------------+
| TECA_RANKS_PER_DEVICE   | The number of MPI ranks per node allowed to use   |
|                         | each GPU. The default is 1 MPI rank per GPU. TECA |
|                         | will use multiple threads within a rank to        |
|                         | service assigned GPUs. This environment variable  |
|                         | may be useful when threading cannot be used.      |
+-------------------------+---------------------------------------------------+
| TECA_INITIALIZE_MPI     | If set to FALSE, or 0, MPI initialization is      |
|                         | skipped. This can be used to run in serial on     |
|                         | Cray login nodes.                                 |
+-------------------------+---------------------------------------------------+
| TECA_DO_TEST            | If set to 0 or FALSE regression tests will update |
|                         | the associated baseline images.                   |
+-------------------------+---------------------------------------------------+


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
`CMAKE_CXX_FLAGS_RELEASE`. Once compiled the built in profiler may be enabled
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

* when wrapping conditionals logical operators go on the preceding line.  For
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

Variable, function, and class names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* use lower case with _ separators. This is often called "snake case"

Macros
~~~~~~
* to be defined in all caps, and do not require a semicolon. For
  example :

    .. code-block:: c++

        DO_SOMETHING(blah, blah)

* write macros so that they may be safely used in single line conditional
  branches.


Templates
~~~~~~~~~~~~~~~~~~~~~~~~
* prefer all capitals for template typenames
* use TEMPLATE_DISPATCH macros for dispatching to templated data structures

Warnings
~~~~~~~~
* treat warnings as errors, developers should use -Wall or
  the compiler equivalent ( -Weverything, /W4, etc)

Conditionals
~~~~~~~~~~~~
* conditionals may omit braces, in that case the code should be on the
  following line (i.e. no one liners). For example :

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
* use `std::shared_ptr` in place of C style pointers for data structures that
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
* throw exceptions only when absolutely necessary and the program
  needs to terminate.

Thread safety and MPI collectives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* design algorithms for thread per-pipeline invocation. this means avoiding the
  use of OpenMP for loop level parallelism.
* algorithms need to be thread safe except for the first invocation of
  `get_output_metadata`.
* MPI collectives are safe to use in the first invocation of `get_output_metadata`.

Testing
~~~~~~~
* New code will not be accepted without an accompanying regression test
* New code will not be accepted when existing tests fail
* Regression tests should prefer analytic solutions when practical
* Test data should be as small as possible.
