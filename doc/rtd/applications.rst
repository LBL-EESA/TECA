Command Line Applications
=========================
TECA's command line applications deliver the highest perfromance, while
providing a good deal of flexibility for common use cases.  This section
describes how to run TECA's command line applications. If you need more
flexibility or functionality not packaged in a command line application consider
using our Python scripting capabilities.

Running the command line applications requires the environment to be correctly
configured. One should either install to a default location, or load the
appropriate modulefile, for example when using installs on NERSC. Experience
has proven that it is best to start with a clean environment (i.e. no other
modules loaded) and to load the teca modulefiles as late as possible.  For
instance when running in a batch script on the Cray load the teca module in the
batch script rather than in your shell.

+----------------------------------------+----------------------------------------------+
| **Application**                        | **Description**                              |
+----------------------------------------+----------------------------------------------+
| :ref:`teca_metadata_probe`             | A diagnostic tool used for run planning      |
+----------------------------------------+----------------------------------------------+
| :ref:`teca_integrated_vapor_transport` | Computes IVT (integrated vapor transport)    |
+----------------------------------------+----------------------------------------------+
| :ref:`teca_bayesian_ar_detect`         | AR detection with uncertainty quantification |
+----------------------------------------+----------------------------------------------+
| :ref:`teca_deeplab_ar_detect`          || A machine learning based AR detector        |
+----------------------------------------+----------------------------------------------+
| :ref:`teca_temporal_reduction`         || Computes reductions (min, max, average)     |
|                                        || over the time dimension                     |
+----------------------------------------+----------------------------------------------+
| :ref:`teca_tc_detect`                  | TC detection using GFDL algorithm            |
+----------------------------------------+----------------------------------------------+
| :ref:`teca_tc_trajectory`              | Computes TC tracks from a set of candidates  |
+----------------------------------------+----------------------------------------------+
| :ref:`teca_tc_wind_radii`              | Computes storm size from a set of TC tracks  |
+----------------------------------------+----------------------------------------------+
| :ref:`teca_tc_stats`                   || Computes descriptive statistics from a set  |
|                                        || of TC tracks                                |
+----------------------------------------+----------------------------------------------+
| :ref:`teca_tc_trajectory_scalars`      || Plots tracks on a map along with GFDL       |
|                                        || detectors parameters                        |
+----------------------------------------+----------------------------------------------+
| :ref:`teca_tc_wind_radii_stats`        || Computes descriptive statistics relating to |
|                                        || TC storm size                               |
+----------------------------------------+----------------------------------------------+
| :ref:`teca_convert_table`              || Convert tabular data from one file format   |
|                                        || to another                                  |
+----------------------------------------+----------------------------------------------+
| :ref:`teca_event_filter`               || Select TC tracks using run time provided    |
|                                        || expressions                                 |
+----------------------------------------+----------------------------------------------+

Considerations When Running at NERSC
-------------------------------------
Compute vs Login Nodes
~~~~~~~~~~~~~~~~~~~~~~
The Login nodes are the ones you land on when ssh'ing in while compute nodes
are obtained via slurm commands.  MPI parallel programs cannot be run on the login
nodes, even in serial. When one tries to run a parallel application on a login
node the program will abort in MPI_Init.  Because many of the TECA command line
applications use MPI, one should run them from the compute nodes. For large
runs this should be accomplished by submitting a batch job. For experimentation
and debugging in the shell use the interactive queue.

File Systems
~~~~~~~~~~~~
NERSC provides the following file systems, knowing their properties is a key
part of successfully making runs at NERSC.

Home ($HOME)
    The home file system is a conventional networked file system. It provides
    the worst performance and should not be used with TECA at all.

Scratch ($SCRATCH)
    The Lustre scratch file system provides the best performance and should be
    used for both TECA installs and the data that will be processed.

Project/Community ($CFS)
    The CFS file system provides reasonable performance and could be used for
    both TECA installs and data. Note that before launching runs processing
    data stored on CFS `export HDF5_USE_FILE_LOCKING=FALSE`. Not disabling file
    locking on CFS will result in the cryptic NetCDF error `NetCDF: HDF error`

When making runs at NERSC using TECA one should use the scratch file system for
both builds and installs if at all possible. When the data to be processed
resides on CFS file system, disable HDF5 file locking.

Python based code on KNL Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Importing numpy can abort on KNL nodes, a warning that 272 cores is beyond the
acceptable limit of 256 is displayed before the code exits.  One may work
around this by setting `export OMP_NUM_THREADS=1`. However, that this could
degrade the performance of other codes using OpenMP for parallelism.

KNL vs Haswell Nodes
~~~~~~~~~~~~~~~~~~~~
Some key differences in architectures should be taking into account when
planning runs.  Haswell CPU's have higher clock speeds and greater overall
computational throughput and TECA will run faster there. The Haswell nodes have
2 CPU's per node and hence 2 NUMA domains. For this reason one should tell
slurm to bind MPI ranks to cores. This ensures memory accesses occur within the
local NUMA domain. TECA's threading infrastructure does this by default.  While
the KNL nodes are slower, they have a lower charge factor and jobs over 1024
nodes in the regular queue have a 50% discount. Additionally queue wait times
for KNL nodes are substantially lower than for Haswell nodes. For those reasons
KNL is a great option.

Sizing and Layout of Runs
~~~~~~~~~~~~~~~~~~~~~~~~~
The number of time steps is key for planning a parallel run in TECA
applications that parallelize over time steps. One can use the run time of the
app on a single time step in conjunction with number of time steps per MPI rank
in the job to estimate the run time at a given concurrency level. One then
selects the concurrency level based on the run time and, queue wait times, etc.
See :ref:`teca_metadata_probe` for information on determining the number of
time steps and available arrays.

TECA will automatically make use of threads for on node parallelism if there
are more phyiscal cores available than there are MPI ranks in use. In order to
take advantage of this one needs to spread the MPI ranks out on the set of
compute nodes in use such that there are fewer MPI ranks than physical CPU
cores. This is done through the combination of *-n* and *-N* srun options.
Little n tells the total number of MPI ranks to use, and big N tells how many
nodes in total to spread them across.  :math:`(number of nodes) * (physical
cores per node) > (total number of MPI ranks)` Haswell nodes have 32 physical
cores per node while KNL nodes have 68 physical cores per node.

Common Command Line Options
---------------------------
The command line applications have been designed so that the specification input
datasets, output datasets, and common execution control options are the same
where ever possible. Details of the common options and controls are presented
in this section in more detail.

Getting Help
~~~~~~~~~~~~
All command line applications support the following options for displaying
application specific command line option documentation in the shell.

--help
    Display the basic options help. Basic options are the ones that will be
    most commonly used with the application. In most cases using just the basic
    options will suffice.

--advanced_help
    Display the advanced options help. All of the settable properties of the
    individual pipeline stages used in the application can be accessed via the
    advanced options. Each stage is given a name that is used to prefix its
    options. In that way multiple stages of the same type can be differentiated.
    Through this mechanism all available control parameters are exposed to the
    user.

--full_help
    Display both the basic and the advanced options help.

Specifying Input NetCDF Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For applications that process mesh based data in NetCDF format there are two
command line options for specifying input data to process.

--input_file arg
    a *teca_multi_cf_reader* configuration file identifying the set of NetCDF CF2
    files to process.  When present data is read using the
    teca_multi_cf_reader. Use one of either *--input_file* or *--input_regex*.

--input_regex arg
    a *teca_cf_reader* regex identifying the set of NetCDF CF2 files to process.
    When present data is read using the teca_cf_reader. Use one of either
    *--input_file* or *--input_regex*.

Note that both of these options make use of regular expressions to identify a
set of files to process. Regular expressions provide a compact mechanism for
specifying a large set of files. Furthermore they can be used to subset the
files based on the contents of the file names. Subsetting in this way enables
one to process time ranges.

Regular Expressions
^^^^^^^^^^^^^^^^^^^^
Regular expressions are used by the *teca_wrf_reader*, *teca_cf_reader* and
*teca_multi_cf_reader* to identify and select all or a subset of a NetCDF
dataset to read and process.

Regular expressions look like the more familiar shell glob, but are
much more powerful and the special characters have different meanings.
Here are the basics:

+---------------+------------------------------------------------------------------+
| **Character** | **Description**                                                  |
+---------------+------------------------------------------------------------------+
| .             | wild card, matches any character                                 |
+---------------+------------------------------------------------------------------+
| \*            | repeat the previous character zero or more times                 |
+---------------+------------------------------------------------------------------+
| []            || match any character in the brackets. For example [0-9] matches  |
|               || a single digit number zero through nine. [A-Z] matches a single |
|               || capital letter, while [xzy] matches a single x,y, or z          |
+---------------+------------------------------------------------------------------+
| ^             || If used as the first character in a [] list, it negates the     |
|               || list. Otherwise, this is an anchor matching the beginning of    |
|               || the string.                                                     |
+---------------+------------------------------------------------------------------+
| \\            || Escapes the next character. This is critical for specifying     |
|               || file names, the . separating the file from the extension needs  |
|               || to be escaped.                                                  |
+---------------+------------------------------------------------------------------+
| $             | matches the end of the string.                                   |
+---------------+------------------------------------------------------------------+

Because shell globs uses some of the same control characters, when a regex
is issued in a shell the characters must be quoted or escaped to keep the
shell from expanding them. Single quotes are the most effective since they
prevent the shell from processing the quoted text. Alternatively backslashes
may be used to escape characters individually.

Another difference between shell globbing and regular expressions is that
regular expressions can partially match. Use of anchors ^ and $
can help, even so care is required to avoid picking up files inadvertently.

An example of an input regex which includes all .nc files is: '.\*\\.nc\$'. If
instead one wanted to grab only files from 2004-2005 then
'.\*\\.200[45].*\\.nc\$' would do the trick. For the best performance, specify
the smallest set of files needed to achieve the desired result. Each of the
files will be opened in order to scan the time axis.

TECA makes use of POSIX Basic Regular Expressions, more information can be found
`here <https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap09.html#tag_09>`_.

MCF Reader Configuration Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When data from the same dataset resides in different directories MCF
configuration files are used to configure the readers.

The configuration file consists of *name = value* pairs and *flags* organized
in sections. Sections are declared using brackets []. There is an optional
opening global section that comes first followed by one or more `[cf_reader]`
sections.

The following example from the HighResMIP dataset MCF configures the reader to
read the variables *hus*, *ua* and *va* each from a different subdirectory.

.. _HighResMIPMCF:

.. code-block:: ini

    # TECA multi cf reader config
    # Fri Nov  6 09:24:17 PST 2020

    data_root = /global/cfs/cdirs/m3522/cmip6/CMIP6_hrmcol/HighResMIP/CMIP6/HighResMIP/ECMWF/ECMWF-IFS-HR/highresSST-present/r1i1p1f1/6hrPlevPt
    regex = 6hrPlevPt_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_gr_199[0-9].*\.nc$

    [cf_reader]
    variables = hus
    regex = %data_root%/hus/gr/v20170915/hus_%regex%
    provides_time
    provides_geometry

    [cf_reader]
    variables = ua
    regex = %data_root%/ua/gr/v20170915/ua_%regex%

    [cf_reader]
    variables = va
    regex = %data_root%/va/gr/v20170915/va_%regex%


The global section may contain `data_root` and `regex` name-value pairs.
Occurrences of the strings `%data_root%` and `%regex%` in the `regex` name-value
pairs found in following `[cf_reader]` sections are replaced with the values of
the globals.  The following global section key words are supported:

+-----------------------+-----------------------------------------------------+
| key word              | description                                         |
+=======================+=====================================================+
| data_root             | | Can be used to hold the common part of the path   |
|                       | | to data files. Optional. Occurrences of           |
|                       | | %data_root% found in the regex specification will |
|                       | | be replaced with this value.                      |
+-----------------------+-----------------------------------------------------+
| regex                 | | Can be used to hold the common part of the        |
|                       | | regular expression. Optional. Occurrences of      |
|                       | | %data_root% found in the regex specification will |
|                       | | be replaced with this value.                      |
+-----------------------+-----------------------------------------------------+

Each `[cf_reader]` section consists of a `name`(optional), a `regex`,
a list of `variables`, a `provides_time` flag(optional) and a
`provides_geometry` flag(optional). At least one section must contain a
`provides_time` and `provides geometry` flag.  The following are `[cf_reader]`
section key words:

+-----------------------+-----------------------------------------------------+
| key word              | description                                         |
+=======================+=====================================================+
| name                  | | An optional name for the reader. This is used to  |
|                       | | set advanced command line options.                |
+-----------------------+-----------------------------------------------------+
| regex                 | A regular expression identifying a set of files.    |
+-----------------------+-----------------------------------------------------+
| variables             | | A comma separated list of variables to serve from |
|                       | | this reader. Variables not in the list are        |
|                       | | ignored.                                          |
+-----------------------+-----------------------------------------------------+
| provides_time         | | A flag, the presence of which indicates that this |
|                       | | reader will provide the time axis.                |
+-----------------------+-----------------------------------------------------+
| provides_geometry     | | A flag the presence of which indicates that this  |
|                       | | reader will provide the mesh definition.          |
+-----------------------+-----------------------------------------------------+


Overriding the Time Axis
~~~~~~~~~~~~~~~~~~~~~~~~
In cases where it is necessary to override the time axis or manually specify
values, the following teca_cf_reader options can be used.

--cf_reader::t_axis_variable arg
  The name of variable that has time axis coordinates (time). Set to an empty
  string to enable override methods (--filename_time_template, --t_values) or
  to disable time coordinates completely

--cf_reader::t_calendar arg
  An optional calendar override. May be one of: standard, Julian,
  proplectic_Julian, Gregorian, proplectic_Gregorian, Gregorian_Y0,
  proplectic_Gregorian_Y0, noleap, no_leap, 365_day, 360_day. When the
  override is provided it takes precedence over the value found in the file.
  Otherwise the calendar is expected to be encoded in the data files using
  CF2 conventions.

--cf_reader::t_units arg
  An optional CF2 time units specification override declaring the units of
  the time axis and a reference date and time from which the time values are
  relative to. If this is provided it takes precedence over the value found
  in the file. Otherwise the time units are expected to be encoded in the
  files using the CF2 conventions

--cf_reader::filename_time_template arg
  An optional std::get_time template string for decoding time from the input
  file names. If no calendar is specified the standard calendar is used. If
  no units are specified then "days since %Y-%m-%d 00:00:00" where Y,m,d are
  determined from the filename of the first file. Set t_axis_variable to an
  empty string to use.

--cf_reader::t_values arg
  An optional explicit list of double precision values to use as the time
  axis. If provided these take precedence over the values found in the files.
  Otherwise the variable pointed to by the t_axis_variable provides the time
  values. Set t_axis_variable to an empty string to use.

The Overrides, *--filename_time_template* and *t_values* are activated by
setting *--t_axis_variable* to an empty string "". When decoding the time
axis from file names, a template must be provided that matches the filenames.
For instance a template for the files: my_file_20170516_00.nc,
my_file_20170516_03.nc, ... might be my_file_%Y%m%d_%H.nc.


Subsetting in the Time Dimension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A simple way of subsetting the time dimension is by using a suitable regex
when specifying the input dataset. This section describes options that will
subset withing the data identified by the regular expression.

The following two command line options let one subset by time step. This is the
most efficient way to subset the time dimension. However, Use of these
options requires one to know the mapping between steps and times. In some cases
this is easy to calculate. For instance when there is a file per day of data.

--first_step arg
    The first time step to process

--last_step arg
    The last time step to process

When it is not easy to determine the mapping between time steps and time the
following command line options use TECA's calendaring capabilities to select a
subset of data occurring between two dates provided in a human readable form.

--start_date arg
    The first time to process in 'Y-M-D h:m:s' format

--end_date arg
    The last time to process in 'Y-M-D h:m:s' format

The accepted format is a CF style human readable date spec such as YYYY-MM-DD
hh:mm:ss.  Because of the space in between day and hour spec quotes must be
used. For example "2005-01-01 00:00:00". Specifying start and end dates are
optional. If only *--start_date* is given then the steps from that date on are
included, while if only *--end-date* is given steps up to an including that
date are included.  if neither *--start-date* nor *--end-date* are given then
all of the time steps in all of the files specified are processed.


Specifying Mesh Dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~
TECA identifies mesh coordinate axes using the names *lon* and *lat*. One can
override these and make another variable provide the coordinate axes.

Unless an application is intrinsically 3D (eg. a vertical integral) by default
the mesh is assumed to be 2D. To tell the reader to generate a 3D mesh
set the z-axis variable name.

--x_axis_variable arg
    name of x coordinate variable (lon)

--y_axis_variable arg
    name of y coordinate variable (lat)

--z_axis_variable arg
    name of z coordinate variable (). When processing 3D set this to the
    variable containing the vertical coordinates. When empty the data will be
    treated as 2D.

Writing Results To Disk
~~~~~~~~~~~~~~~~~~~~~~~

Mesh Based Data in NetCDF CF2 Format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following options control file name and layout when writing mesh based data
in NetCDF CF2 format.

--output_file arg
    file pattern for output netcdf files (%t% is the time index)

--steps_per_file arg
    number of time steps per output file

--cf_writer::date_format arg
    A strftime format used when encoding dates into the output
    file names (%F-%HZ)")


Tabular Data
^^^^^^^^^^^^
Data such as TC tracks is stored in tabular format. The writer will select the format
based on the output file extension. This can be one of: *.csv*, *.bin*, or *.nc*.

.. _teca_metadata_probe:

teca_metadata_probe
------------------------------------
The metadata probe is a command line application that presents how TECA sees
input dataset to the user in a textual format. The primary use of the metadata
probe is: planning runs by getting the number of time steps selected by a date
range or regular expression; validating regular expression or MCF (multi-cf)
configuration files; determining which variables are in the files and what are
their shapes and dimensions.

Inputs
~~~~~~
A 2 or 3D time dependent mesh in NetCDF CF2 format

Outputs
~~~~~~~
#. The number of time steps found in the files selected by the regex and/or start and end date
#. The calendar and simulated time range selected by the regex and/or start/end date
#. The mesh dimensionality. The default is 2D , for 3D data use the `--z_axis_variable` command line option.
#. A list of the available arrays and their dimensions and shapes.


Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~

--input_file arg
    a teca_multi_cf_reader configuration file identifying the set of NetCDF CF2
    files to process.  When present data is read using the
    teca_multi_cf_reader. Use one of either *--input_file* or *--input_regex*.

--input_regex arg
    a teca_cf_reader regex identifying the set of NetCDF CF2 files to process.
    When present data is read using the teca_cf_reader. Use one of either
    *--input_file* or *--input_regex*.

--x_axis_variable arg
    name of x coordinate variable (lon)

--y_axis_variable arg
    name of y coordinate variable (lat)

--z_axis_variable arg
    name of z coordinate variable (). When processing 3D set this to the
    variable containing vertical coordinates. When empty the data will be
    treated as 2D.

--start_date arg
    The first time to process in 'Y-M-D h:m:s' format. Note: There must be a
    space between the date and time specification.

--end_date arg
    The last time to process in 'Y-M-D h:m:s' format

--help
    display the basic options help

--advanced_help
    display the advanced options help

--full_help
    display all options help


Examples
~~~~~~~~

.. _mdp_highresmip:

HighResMIP data organized with each variable in its own folder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In this example the metadata_probe examines data from the HighResMIP collection.
The data is organized such that the data files for each variable reside in their
own directory. In this case we use the :ref:`MCF file<HighResMIPMCF>`.

.. code-block:: bash

    $ salloc -N 17 -C knl -q interactive -t 01:00:00

    $ module swap PrgEnv-intel PrgEnv-gnu
    $ module use /global/cscratch1/sd/loring/teca_testing/deps/cf_reader_performance/modulefiles/
    $ module load teca

    $ export HDF5_USE_FILE_LOCKING=FALSE

    $ time srun -N 17 -n 1024 teca_metadata_probe --z_axis_variable plev \
        --input_file HighResMIP_ECMWF_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_6hrPlevPt.mcf

    WARNING: [0:46912496725888] [/global/cscratch1/sd/loring/teca_testing/TECA_superbuild/build-cf-reader/TECA-prefix/src/TECA/io/teca_cf_reader.cxx:623 TECA-BARD-v1.0.1-222-ge294c25]
    WARNING: File 528 "hus_6hrPlevPt_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_gr_199401010000-199401311800.nc" units "days since 1994-1-1 00:00:00" differs from base units "days since 1950-1-1 00:00:00" a conversion will be made.
    WARNING: [0:46912496725888] [/global/cscratch1/sd/loring/teca_testing/TECA_superbuild/build-cf-reader/TECA-prefix/src/TECA/io/teca_cf_reader.cxx:623 TECA-BARD-v1.0.1-222-ge294c25]
    WARNING: File 529 "hus_6hrPlevPt_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_gr_199402010000-199402281800.nc" units "days since 1994-1-1 00:00:00" differs from base units "days since 1950-1-1 00:00:00" a conversion will be made.
    #
    # many simlar warning messages omitted...
    #
    WARNING: [0:46912496725888] [/global/cscratch1/sd/loring/teca_testing/TECA_superbuild/build-cf-reader/TECA-prefix/src/TECA/io/teca_cf_reader.cxx:623 TECA-BARD-v1.0.1-222-ge294c25]
    WARNING: File 779 "va_6hrPlevPt_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_gr_201412010000-201412311800.nc" units "days since 1994-1-1 00:00:00" differs from base units "days since 1950-1-1 00:00:00" a conversion will be made.

    A total of 94964 steps available. Using the gregorian calendar. Times are specified
    in units of days since 1950-1-1 00:00:00. The available times range from 1950-1-1
    0:0:0 (0) to 2014-12-31 18:0:0 (23740.8).

    Mesh dimension: 3D
    Mesh coordinates: lon, lat, plev

    7 data arrays available

      Id   Name    Type         Dimensions                Shape
    -----------------------------------------------------------------------
      1    hus     NC_FLOAT     [time, plev, lat, lon]    [94964, 7, 361, 720]
      2    lat     NC_DOUBLE    [lat]                     [361]
      3    lon     NC_DOUBLE    [lon]                     [720]
      4    plev    NC_DOUBLE    [plev]                    [7]
      5    time    NC_DOUBLE    [time]                    [94964]
      6    ua      NC_FLOAT     [time, plev, lat, lon]    [94964, 7, 361, 720]
      7    va      NC_FLOAT     [time, plev, lat, lon]    [94964, 7, 361, 720]


    real	1m23.011s
    user	0m0.451s
    sys	0m0.469s

There are 94964 time steps in this 3D dataset. The maximum MPI concurrency for
this dataset is 94964 MPI ranks. Using fewer MPI ranks will result in some or
all ranks processing multiple time steps. A number of warnings were printed as
the probe ran because the reader detected that the calendar and/or time units
were inconsistent in some of the files. In this case the reader made a
conversion such that all of the data is presented to the down stream stages
in the same calendar and units.

.. _mdp_artmip:

ARTMIP data organized by year with the time axis encoded in the file name
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This example shows how to configure the reader for extracting the time axis
from the file names.  In this example dataset was organized such that the data
from each simulated year was stored in its own folder. Each time step was stored
in a file, no time information was stored in the file itself. Instead, the date
and time was encoded in the file name.

.. code-block:: bash

    $ salloc -N 10 -C knl -q interactive -t 01:00:00

    $ module swap PrgEnv-intel PrgEnv-gnu
    $ module use `pwd`/deps/cf_reader_performance/modulefiles
    $ module load teca

    $ year=1980
    $ data_dir=/global/project/projectdirs/m1517/cascade/external_datasets/ARTMIP/MERRA_2D/${year}
    $ regex='ARTMIP_MERRA_2D_.*\.nc'

    $ time srun -n 680 -N 10 teca_metadata_probe \
        --input_regex "${data_dir}/${regex}" --cf_reader::t_axis_variable '' \
        --cf_reader::filename_time_template ARTMIP_MERRA_2D_%Y%m%d_%H.nc

    STATUS: [0:46912496725888] [/global/cscratch1/sd/loring/teca_testing/TECA_superbuild/build-cf-reader/TECA-prefix/src/TECA/io/teca_cf_reader.cxx:823 TECA-BARD-v1.0.1-222-ge294c25]
    STATUS: The time axis will be infered from file names using the user provided template "ARTMIP_MERRA_2D_%Y%m%d_%H.nc" with the "standard" calendar in units "days since 1980-01-01 00:00:00"

    A total of 2928 steps available in 2928 files. Using the standard calendar.
    Times are specified in units of days since 1980-01-01 00:00:00. The available
    times range from 1980-1-1 0:0:0 (0) to 1980-12-31 21:0:0 (365.875).

    Mesh dimension: 2D
    Mesh coordinates: lon, lat

    7 data arrays available

      Id   Name    Type         Dimensions    Shape
    -------------------------------------------------
      1    IVT     NC_FLOAT     [lat, lon]    [361, 576]
      2    IWV     NC_FLOAT     [lat, lon]    [361, 576]
      3    PS      NC_FLOAT     [lat, lon]    [361, 576]
      4    lat     NC_DOUBLE    [lat]         [361]
      5    lon     NC_DOUBLE    [lon]         [576]
      7    uIVT    NC_FLOAT     [lat, lon]    [361, 576]
      8    vIVT    NC_FLOAT     [lat, lon]    [361, 576]

    real	0m13.980s
    user	0m0.307s
    sys	0m0.240s

The output shows that there were 2928 time steps in this year. The maximum level
of concurrency one could exploit in processing this dataset is 2928 MPI ranks.
Running with fewer than 2928 MPI ranks will result in some or all ranks processing
multiple time steps.

.. _mdp_cam5:

A large CAM5 dataset
^^^^^^^^^^^^^^^^^^^^
In the following example the metadata probe is used
to determine the number of time steps in a large CAM5 dataset spread over many files.

.. code-block:: bash

    $data_dir=/global/cscratch1/sd/mwehner/machine_learning_climate_data/All-Hist/CAM5-1-0.25degree_All-Hist_est1_v3_run1/h2
    $srun -N 17 -n 1024 ./bin/teca_metadata_probe --input_regex=${data_dir}/'.*\.nc$'

    A total of 58400 steps available in 7300 files. Using the noleap calendar.
    Times are specified in units of days since 1995-02-01 00:00:00. The available
    times range from 1996-1-1 0:0:0 (334) to 2015-12-31 21:0:0 (7633.88).

    Mesh dimension: 2D
    Mesh coordinates: lon, lat

    45 data arrays available

      Id    Name            Type         Dimensions          Shape
    -----------------------------------------------------------------------

      1     PRECT           NC_FLOAT     [time, lat, lon]    [58400, 768, 1152]
      2     PS              NC_FLOAT     [time, lat, lon]    [58400, 768, 1152]
      3     PSL             NC_FLOAT     [time, lat, lon]    [58400, 768, 1152]
      4     QREFHT          NC_FLOAT     [time, lat, lon]    [58400, 768, 1152]
      5     T200            NC_FLOAT     [time, lat, lon]    [58400, 768, 1152]
      6     T500            NC_FLOAT     [time, lat, lon]    [58400, 768, 1152]
      7     TMQ             NC_FLOAT     [time, lat, lon]    [58400, 768, 1152]
      8     TREFHT          NC_FLOAT     [time, lat, lon]    [58400, 768, 1152]
      9     TS              NC_FLOAT     [time, lat, lon]    [58400, 768, 1152]
      10    U850            NC_FLOAT     [time, lat, lon]    [58400, 768, 1152]
      11    UBOT            NC_FLOAT     [time, lat, lon]    [58400, 768, 1152]
      12    V850            NC_FLOAT     [time, lat, lon]    [58400, 768, 1152]
      13    VBOT            NC_FLOAT     [time, lat, lon]    [58400, 768, 1152]
      14    Z1000           NC_FLOAT     [time, lat, lon]    [58400, 768, 1152]
      15    Z200            NC_FLOAT     [time, lat, lon]    [58400, 768, 1152]
      16    ZBOT            NC_FLOAT     [time, lat, lon]    [58400, 768, 1152]
      17    ch4vmr          NC_DOUBLE    [time]              [58400]
      18    co2vmr          NC_DOUBLE    [time]              [58400]
      19    date            NC_INT       [time]              [58400]
      20    date_written    NC_BYTE      [time, chars]       [58400, 8]
      21    datesec         NC_INT       [time]              [58400]
      22    f11vmr          NC_DOUBLE    [time]              [58400]
      23    f12vmr          NC_DOUBLE    [time]              [58400]
      24    gw              NC_DOUBLE    [lat]               [768]
      25    hyai            NC_DOUBLE    [ilev]              [31]
      26    hyam            NC_DOUBLE    [lev]               [30]
      27    hybi            NC_DOUBLE    [ilev]              [31]
      28    hybm            NC_DOUBLE    [lev]               [30]
      29    ilev            NC_DOUBLE    [ilev]              [31]
      30    lat             NC_DOUBLE    [lat]               [768]
      31    lev             NC_DOUBLE    [lev]               [30]
      32    lon             NC_DOUBLE    [lon]               [1152]
      33    n2ovmr          NC_DOUBLE    [time]              [58400]
      34    ndcur           NC_INT       [time]              [58400]
      35    nlon            NC_INT       [lat]               [768]
      36    nscur           NC_INT       [time]              [58400]
      37    nsteph          NC_INT       [time]              [58400]
      38    slat            NC_DOUBLE    [slat]              [767]
      39    slon            NC_DOUBLE    [slon]              [1152]
      40    sol_tsi         NC_DOUBLE    [time]              [58400]
      41    time            NC_DOUBLE    [time]              [58400]
      42    time_bnds       NC_DOUBLE    [time, nbnd]        [58400, 2]
      43    time_written    NC_BYTE      [time, chars]       [58400, 8]
      44    w_stag          NC_DOUBLE    [slat]              [767]
      45    wnummax         NC_INT       [lat]               [768]

In this example the dataset is quite large comprised of 7300 files. Each file has
456MB of data for a total aggregate size of over 3TB. In this case it is necessary
to run the metadata probe using MPI in order for the probe to complete in a
reasonable amount of time. A serial run of the  probe on this dataset took over
71 minutes while the parallel run shown above took about 47 seconds. Note that because this
dataset has a large number of files it is an extreme case, for datasets with on the order
of a few hundred files a serial or small MPI parallel run should work well.

.. _teca_bayesian_ar_detect:

teca_bayesian_ar_detect
--------------------------------------
The Bayesian AR detection (BARD) application is an MPI+threads parallel
code that applies an uncertainty inference on a range of input fields, mainly
Integrated Vapor Transport (IVT) and Integrated Water Vapor (IWV).
We use a Bayesian framework to sample from the set of AR detector parameters
that yield AR counts similar to the expert database of AR counts; this yields
a set of plausible AR detectors from which we can assess quantitative uncertainty.
TECA-BARD is described in :cite:`teca_bard`.


.. figure:: images/cascade_ar_globe_fig.png
    :width: 70%
    :align: center

    Pseudocoloring of precipitable water with superposed 5%, 50%, and 100%
    contours of AR probability. Green x shows ground truth obtained via
    manual expert identification.

Inputs
~~~~~~
A 3D time dependent mesh in NetCDF CF2 format with:

1. horizontal wind velocity vector
2. specific humidity

Or a 2D time dependent mesh with:

1. IVT magnitude

Outputs
~~~~~~~
A 2D mesh with:

1. AR probability
2. A segmentation of AR probability

If IVT was computed from horizontal wind and specific humidity

1. IVT vector
2. IVT magnitude

Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~

--input_file arg
  a teca_multi_cf_reader configuration file identifying the set of NetCDF CF2 files to process.
  When present data is read using the teca_multi_cf_reader. Use one of either --input_file or
  --input_regex.

--input_regex arg
  a teca_cf_reader regex identifying the set of NetCDF CF2 files to process. When present data
  is read using the teca_cf_reader. Use one of either --input_file or --input_regex.

--ivt arg
  name of variable with the magnitude of integrated vapor transport (IVT)

--compute_ivt_magnitude
  when this flag is present magnitude of vector IVT is calculated. use --ivt_u and --ivt_v to set
  the name of the IVT vector components if needed.

--ivt_u arg
  name of variable with longitudinal component of the integrated vapor transport vector. (IVT_U)

--ivt_v arg
  name of variable with latitudinal component of the integrated vapor transport vector. (IVT_V)

--write_ivt_magnitude
  when this flag is present IVT magnitude is written to disk with the AR detector results

--compute_ivt
  when this flag is present IVT vector is calculated from specific humidity, and wind vector
  components. use --specific_humidity --wind_u and --wind_v to set the name of the specific
  humidity and wind vector components, and --ivt_u and --ivt_v to control the names of the
  results, if needed.

--specific_humidity arg
  name of variable with the 3D specific humidity field. If present IVT vector will be computed
  from 3D wind  and specific humidity fields.

--wind_u arg (=U)
  name of variable with the 3D longitudinal component of the wind vector. If present IVT vector
  will be computed from 3D wind  and specific humidity fields.

--wind_v arg (=V)
  name of variable with the 3D latitudinal component of the wind vector. If present IVT vector
  will be computed from 3D wind  and specific humidity fields.

--write_ivt
  when this flag is present IVT vector is written to disk with the result

--x_axis_variable arg
  name of x coordinate variable (lon)

--y_axis_variable arg
  name of y coordinate variable (lat)

--z_axis_variable arg
  name of z coordinate variable (plev)

--binary_ar_threshold arg (=0.667)
  probability threshold for segmenting ar_probability to produce ar_binary_tag

--output_file arg (=CASCADE_BARD_%t%.nc)
  A path and file name pattern for the output NetCDF files. %t% is replaced with a human readable
  date and time corresponding to the time of the first time step in the file. Use
  --cf_writer::date_format to change the formatting

--first_step arg
  first time step to process

--last_step arg
  last time step to process

--steps_per_file arg
  number of time steps per output file

--start_date arg
  The first time to process in 'Y-M-D h:m:s' format. Note: There must be a space between the date
  and time specification

--end_date arg
  The last time to process in 'Y-M-D h:m:s' format

--n_threads arg
  Sets the thread pool size on each MPI rank. When the default value of -1 is used TECA will
  coordinate the thread pools across ranks such each thread is bound to a unique physical core.

--periodic_in_x arg (=1)
  Flags whether the x dimension (typically longitude) is periodic.

--verbose
  enable extra terminal output

--help
  displays documentation for application specific command line options

--advanced_help
  displays documentation for algorithm specific command line options

--full_help
  displays both basic and advanced documentation together

Examples
~~~~~~~~

HighResMIP data organized with each variable in its own folder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This example illustrates detecting ARs(atmospheric rivers) in a HighResMIP
dataset using TECA's BARD(Bayesian AR detector) detector.

.. code-block:: bash

    #!/bin/bash
    #SBATCH -C knl
    #SBATCH -N 1484
    #SBATCH -q regular
    #SBATCH -t 00:30:00
    #SBATCH -A m1517
    #SBATCH -J 2_CASCADE_BARD_AR_detect

    module swap PrgEnv-intel PrgEnv-gnu
    module use /global/cscratch1/sd/loring/teca_testing/deps/cf_reader_performance/modulefiles/
    module load teca/cf_reader_performance

    # configure HDF5 file locking if on Cori (CFS)community file system
    export HDF5_USE_FILE_LOCKING=FALSE

    # make a directory for the output files
    out_dir=HighResMIP_ECMWF_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_6hrPlevPt/CASCADE_BARD_all
    mkdir -p ${out_dir}

    # do the ar detections. change -N and -n to match the rus size.
    time srun -N 1484 -n 23744 teca_bayesian_ar_detect \
        --input_file ./HighResMIP_ECMWF_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_6hrPlevPt.mcf \
        --specific_humidity hus --wind_u ua --wind_v va --ivt_u ivt_u --ivt_v ivt_v --ivt ivt \
        --compute_ivt --write_ivt --write_ivt_magnitude --steps_per_file 128 \
        --output_file ${out_dir}/CASCADE_BARD_AR_%t%.nc

This HighResMIP dataset spans the year 1950 to 2014 with 7 pressure levels at a
1 degree spatial and 6 hourly time resolution. There are 94964 simulated time
steps stored in 780 files which require 290 GB disk space per scalar field.
The data is organized such that the data files for each variable reside in their
own directory. This :ref:`MCF file<HighResMIPMCF>` was used to configure the
readers.

In this example IVT is calculated on the fly from horizontal wind vector and
specific humidity, thus 870 GB was processed. If IVT magnitude is available on
disk, one may omit the *--compute_ivt* flag to use it directly.

The dataset was processed using 100912 cores on 1484 KNL nodes on NERSC's Cray
supercomputer Cori. The run computed the IVT vector, its magnitude, the
probability of an AR and a segmentation of the AR probability. The run
completed in 4m 1s and generated a total of 392 GB of data.

In determining the number of MPI ranks to use in this run, the number of time
steps in the dataset was first determined using the :ref:`teca_metadata_probe`
as shown in the above :ref:`example<mdp_highresmip>`.  Note that CASCADE BARD
relies on threading for performance and spreading the MPI ranks out such that
each has a number of threads is advised. Here each MPI rank was given 4
physical cores for exclusive use.


ARTMIP data organized by year with the time axis encoded in the file name
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following example documents SLURM script that was used to generate output
used by :cite:`teca_bard`.  This run used 1520 nodes, and simultaneously ran
1,024 AR detectors on the 37 years of the MERRA-2 reanalysis in approximately 2
minutes on the Cori KNL supercomputer at NERSC.

.. code-block:: bash

    #!/bin/bash
    #SBATCH -J bard_merra2
    #SBATCH -N 1520
    #SBATCH -C knl
    #SBATCH -q regular
    #SBATCH -t 00:20:00

    WORKDIR=$SCRATCH/teca_bard_merra2_artmip
    mkdir -p ${WORKDIR}
    cd $WORKDIR

    module load teca/develop_haswell

    # coinfigure HDF5 for the CFS file system
    HDF5_USE_FILE_LOCKING=FALSE

    for year in `seq 1980 2017`
    do
        echo "Starting ${year}"
        srun -n 680 -c 16 -N 40 --cpu_bind=cores teca_bayesian_ar_detect \
            --input_regex "/global/project/projectdirs/m1517/cascade/external_datasets/ARTMIP/MERRA_2D/${year}/ARTMIP_MERRA_2D_.*\.nc" \
            --cf_reader::t_axis_variable "" \
            --cf_reader::filename_time_template  "ARTMIP_MERRA_2D_%Y%m%d_%H.nc" \
            --steps_per_file 3000 \
            --cf_writer::date_format "%Y" \
            --output_file MERRA2.ar_tag.teca_bard_v1.0.3hourly.%t%.nc4 &> bard_${year}_${SLURM_JOB_ID}.log &
    done

    wait
    echo "All done."

This example shows how to configure the reader for extracting the time axis
from the file names.  In this example dataset was organized such that the data
from each simulated year was stored in its own folder. Each time step was stored
in a file, no time information was stored in the file itself. Instead, the date
and time was encoded in the file name.

In the above script, *srun* launches the detector once for each year on a unique
set of 40 nodes using 680 MPI ranks. The BARD makes use of threads internally
and it can be beneficial to spread the MPI ranks out giving each rank exclusive
access to a number of physical cores. In this example each rank has
approximately 4 cores.

In determining the number of ranks to use per run, the number of steps per year was
taken into account. See the :ref:`teca_metadata_probe` ARTMIP :ref:`example<mdp_artmip>`.


.. _teca_integrated_vapor_transport:

teca_integrated_vapor_transport
-------------------------------------------------
The integrated vapor transport(IVT) command line application computes IVT from
input wind vector and specific humidity.

Inputs
~~~~~~
A 3D time dependent mesh in NetCDF CF2 format with:

1. horizontal wind velocity vector
2. specific humidity

Outputs
~~~~~~~
A 2D mesh with one or more of:

1. IVT vector
2. IVT magnitude


Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~

--input_file arg
    a teca_multi_cf_reader configuration file identifying the set of NetCDF CF2 files to process.
    When present data is read using the teca_multi_cf_reader. Use one of either --input_file or
    --input_regex.

--input_regex arg
    a teca_cf_reader regex identifying the set of NetCDF CF2 files to process. When present data is
    read using the teca_cf_reader. Use one of either --input_file or --input_regex.

--specific_humidity arg
    name of variable with the 3D specific humidity field. (Q)

--wind_u arg (=U)
    name of variable with the 3D longitudinal component of the wind vector. (U)

--wind_v arg (=V)
    name of variable with the 3D latitudinal component of the wind vector. If present IVT vector
    will be computed from 3D wind  and specific humidity fields.

--ivt_u arg
    name to use for the longitudinal component of the integrated vapor transport vector. If present
    the magnitude of IVT will be computed from vector components. (IVT_U)

--ivt_v arg
    name to use for the latitudinal component of the integrated vapor transport vector. If present
    the magnitude of IVT will be computed from vector components. (IVT_V)

--ivt arg
    name of variable with the magnitude of integrated vapor transport (IVT)

--write_ivt_magnitude arg (=0)
    when this is set to 1 magnitude of vector IVT is calculated. use --ivt_u and --ivt_v to set the
    name of the IVT vector components and --ivt to set the name of the result if needed. (0)

--write_ivt arg (=1)
    when this is set to 1 IVT vector is written to disk with the result. use --ivt_u and --ivt_v to
    set the name of the IVT vector components of the result if needed. (1)

--output_file arg (=IVT_%t%.nc)
    A path and file name pattern for the output NetCDF files. %t% is replaced with a human readable
    date and time corresponding to the time of the first time step in the file. Use
    --cf_writer::date_format to change the formatting

--steps_per_file arg
    number of time steps per output file

--x_axis_variable arg
    name of x coordinate variable (lon)

--y_axis_variable arg
    name of y coordinate variable (lat)

--z_axis_variable arg
    name of z coordinate variable (plev)

--periodic_in_x arg (=1)
    Flags whether the x dimension (typically longitude) is periodic.

--first_step arg
    first time step to process

--last_step arg
    last time step to process

--start_date arg
    The first time to process in 'Y-M-D h:m:s' format. Note: There must be a space between the date
    and time specification

--end_date arg
    The last time to process in 'Y-M-D h:m:s' format

--n_threads arg
    Sets the thread pool size on each MPI rank. When the default value of -1 is used TECA will
    coordinate the thread pools across ranks such each thread is bound to a unique physical core.

--verbose
    enable extra terminal output

--help
    displays documentation for application specific command line options

--advanced_help
    displays documentation for algorithm specific command line options

--full_help
    displays both basic and advanced documentation together


Examples
~~~~~~~~

HighResMIP
^^^^^^^^^^
This example illustrates computing IVT(integrated vapor transport) from a
HighResMIP dataset using TECA.

.. code-block:: bash

    #!/bin/bash
    #SBATCH -C knl
    #SBATCH -N 500
    #SBATCH -q debug
    #SBATCH -t 00:30:00
    #SBATCH -A m1517

    module swap PrgEnv-intel PrgEnv-gnu

    # bring a TECA install into your environment
    module use /global/cscratch1/sd/loring/teca_testing/deps/develop/modulefiles/
    module load teca/develop

    # coinfigure HDF5 for the CFS file system
    export HDF5_USE_FILE_LOCKING=FALSE

    # make a directory for the output files
    mkdir -p HighResMIP_ECMWF_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_6hrPlevPt/ivt

    # do the IVT calcllation. change -N and -n to match the run size.
    time srun -N 500 -n 1000 teca_integrated_vapor_transport                                                \
        --input_file ./HighResMIP_ECMWF_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_6hrPlevPt.mcf              \
        --specific_humidity hus --wind_u ua --wind_v va --ivt_u ivt_u --ivt_v ivt_v --ivt ivt               \
        --write_ivt 1 --write_ivt_magnitude 1                                                               \
        --output_file ./HighResMIP_ECMWF_ECMWF-IFS-HR_highresSST-present_r1i1p1f1_6hrPlevPt/ivt/ivt_%t%.nc  \
        --steps_per_file 32 --n_threads -1 --verbose

This HighResMIP dataset spans the year 1950 to 2014 with 7 pressure levels at a
1 degree spatial and 6 hourly time resolution. There are 94964 simulated time
steps stored in 780 files which require 290 GB disk space per scalar field.
The IVT calculation makes use of horizontal wind vector and specific humidity,
thus 870 GB was processed.

The dataset was processed using
100912 cores on 1484 KNL nodes on NERSC's Cray supercomputer Cori. The run
computed the IVT vector and its magnitude. The run completed in 2m 49s and
generated a total of 276 GB of data.

The HighResMIP data is organized such that each
variable is stored in its own directory.  This :ref:`MCF file<HighResMIPMCF>`
was used to configure the readers.


.. _teca_tc_detect:

teca_tc_detect
------------------------------
The cyclone detector is an MPI+threads parallel map-reduce based application
that identifies tropical cyclone tracks in NetCDF-CF2 climate data. The
application is comprised of a number of stages that are run in succession
producing tables containing cyclone tracks. The tracks then can be visualized
or further analyzed using the TECA TC statistics application, TECA's Python
bindings, or the TECA ParaView plugin.

The detection algorithm is based on the open source GFDL code
described in :cite:`gfdl_tc` with improvements to the original code to handle
modern higher spatio-temporal resolution datasets and adjustments to default
thresholds based on observational data published in :cite:`tc_props`.

.. _tracks:

.. figure:: ./images/tracks_1824.png

    Cyclone tracks plotted with 850 mb wind speed and integrated moisture.


.. _candidates:

.. figure:: ./images/candidates_and_tracks.png

    Cyclone candidates and tracks. Not all candidates end up in tracks.

Inputs
~~~~~~
A Cartesian mesh stored in a collection of NetCDF CF2 files. The detector requires on
the following fields.

1. Sea level pressure
2. Surface wind vector
3. 850 mb wind vector
4. 500 mb temperature
5. 200 mb temperature
6. 1000 mb height
7. 200 mb height

Outputs
~~~~~~~
1. Cyclone andidate table
2. Cyclone track table

Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~

--input_file arg
    a teca_multi_cf_reader configuration file identifying the set of NetCDF CF2 files to process.
    When present data is read using the teca_multi_cf_reader. Use one of either --input_file or
    --input_regex.

--input_regex arg
    a teca_cf_reader regex identifying the set of NetCDF CF2 files to process. When present data is
    read using the teca_cf_reader. Use one of either --input_file or --input_regex.

--candidate_file arg
    file path to write the storm candidates to (candidates.bin)

--850mb_wind_u arg
    name of variable with 850 mb wind x-component (U850)

--850mb_wind_v arg
    name of variable with 850 mb wind x-component (V850)

--surface_wind_u arg
    name of variable with surface wind x-component (UBOT)

--surface_wind_v arg
    name of variable with surface wind y-component (VBOT)

--sea_level_pressure arg
    name of variable with sea level pressure (PSL)

--500mb_temp arg
    name of variable with 500mb temperature for warm core calc (T500)

--200mb_temp arg
    name of variable with 200mb temperature for warm core calc (T200)

--1000mb_height arg
    name of variable with 1000mb height for thickness calc (Z1000)

--200mb_height arg
    name of variable with 200mb height for thickness calc (Z200)

--storm_core_radius arg
    maximum number of degrees latitude separation between vorticity max and pressure min defining a
    storm (2.0)

--min_vorticity arg
    minimum vorticty to be considered a tropical storm (1.6e-4)

--vorticity_window arg
    size of the search window in degrees. storms core must have a local vorticity max centered on
    this window (7.74446)

--pressure_delta arg
    maximum pressure change within specified radius (400.0)

--pressure_delta_radius arg
    radius in degrees over which max pressure change is computed (5.0)

--core_temp_delta arg
    maximum core temperature change over the specified radius (0.8)

--core_temp_radius arg
    radius in degrees over which max core temperature change is computed (5.0)

--thickness_delta arg
    maximum thickness change over the specified radius (50.0)

--thickness_radius arg
    radius in degrees over with max thickness change is computed (4.0)

--lowest_lat arg
    lowest latitude in degrees to search for storms (80)

--highest_lat arg
    highest latitude in degrees to search for storms (80)

--max_daily_distance arg
    max distance in km that a storm can travel in one day (1600)

--min_wind_speed arg
    minimum peak wind speed to be considered a tropical storm (17.0)

--min_wind_duration arg
    number of, not necessarily consecutive, days min wind speed sustained (2.0)

--track_file arg
    file path to write storm tracks to

--first_step arg
    first time step to process

--last_step arg
    last time step to process

--start_date arg
    The first time to process in 'Y-M-D h:m:s' format. Note: There must be a space between the date
    and time specification

--end_date arg
    The last time to process in 'Y-M-D h:m:s' format

--n_threads arg
    Sets the thread pool size on each MPI rank. When the default value of -1 is used TECA will
    coordinate the thread pools across ranks such each thread is bound to a unique physical core.

--help
    displays documentation for application specific command line options

--advanced_help
    displays documentation for algorithm specific command line options

--full_help
    displays both basic and advanced documentation together


Examples
~~~~~~~~~

.. _tc_cam5:

A large CAM5 dataset
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    #!/bin/bash
    #SBATCH -N 913
    #SBATCH -C knl
    #SBATCH -q regular
    #SBATCH -t 01:00:00
    #SBATCH -A m1517
    #SBATCH -J teca_tc_detect

    module swap PrgEnv-intel PrgEnv-gnu
    module use /global/cscratch1/sd/loring/teca_testing/deps/cf_reader_performance/modulefiles/
    module load teca/cf_reader_performance

    data_dir=/global/cscratch1/sd/mwehner/machine_learning_climate_data/All-Hist/CAM5-1-0.25degree_All-Hist_est1_v3_run1/h2

    time srun -N 913 -n 58400 \
        teca_tc_detect --input_regex ${data_dir}/'.*\.nc$'  \
            --candidate_file CAM5-1-025degree_All-Hist_est1_v3_run1_h2_candidates.bin \
            --track_file CAM5-1-025degree_All-Hist_est1_v3_run1_h2_tracks.bin

This example shows the detection of TC's in a large (3TB, 7300 file) CAM5
dataset using 58400 cores on NERSC Cori. The run completed in 35 minutes 4
seconds on the KNL nodes. As shown in the above :ref:`example<mdp_cam5>`,
:ref:`teca_metadata_probe` was used to determine the number of MPI ranks.


.. _teca_tc_trajectory:

teca_tc_trajectory
------------------------------
The teca_tc_trajevctory application generates cyclone tracks from a set of
cyclone candidates. A number of detector parameters are applied here that
influence the assembly of tracks from candidates. The parameters are more
completely described in :cite:`gfdl_tc`.  The teca_tc_trajectory application is
not needed to obtain TC tracks because the same code runs as part of the
:ref:`teca_tc_detect` application. It's primary use is for re-running tracking
stages of the algorithm with different parameters on the same set of
candidates.

Inputs
~~~~~~
A table of TC candidates.

Outputs
~~~~~~~
A table of TC tracks.

Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~
The most commonly used command line arguments to the trajectory stage are:

--help
    prints documentation for the most common options.

--full_help
    prints documentation for all options. See --help notes.

--candidate_file
    a file name specifying where to read the storm candidates from.

--track_file
    a file name specifying where to write the detected storm tracks. If not
    specified the tracks are written to a file named tracks.bin in the current
    working directory. One sets the output format via the extension. Supported
    formats include csv, nc, and bin.

Examples
~~~~~~~~~
An example of running the trajectory stage is:

.. code-block:: bash

    teca_tc_trajectory \
        --candidate_file candidates_1990s.bin       \
        --track_file tracks_1990s.bin

the file *tracks_1990s.bin* will contain the list of storm tracks.

.. _teca_tc_wind_radii:

teca_tc_wind_radii
----------------------------------
The wind radii application computes an estimation of a TC's  physical size
using the algorithm described in :cite:`wind_rad_1` and :cite:`wind_rad_2`.
For each point on each track a radial profile is computed over a number of
angular intervals. The radial profiles are used to compute distance from the
storm center to the first downward crossing of given wind speeds. The default
wind speeds are the Saffir-Simpson transitions. Additionally distance to the
peak wind speed and peak wind speed are recorded.  A new table is produced
containing the data.

Tracks are processed in parallel.  The TC trajectory scalars application, TC
stats application and ParaView plugin can be used to further analyze the data.
This application has been used to generate data to train machine learning
algorithms.


.. figure:: images/vbot_tmq_wr_prof_and_plot_0093.png
    :width: 100%
    :align: center

    A category 5 storm about to make landfall over Japan and
    the strom's radial wind profile at the same time instant.
    Rings in the image on the right correspond to blue lines
    in the radial profile on the right. Red lines in the profile
    show the Saffir-Simpson thresholds.

Inputs
~~~~~~
1. A table of TC tracks as generated by the :ref:`teca_tc_detect` application
   or other application with at least time step, track id, and lat, lon locations.
2. The original mesh based data from which tracks were computed with at least
   wind velocity vector.

Output
~~~~~~
A table of TC tracks augmented with storm size columns, with a column for each
threshold and an additional column for the radius at the peak wind speed.

Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~

--track_file arg
    A file containing cyclone tracks (tracks.bin)

--input_file arg
    a teca_multi_cf_reader configuration file identifying the set of NetCDF CF2 files to process.
    When present data is read using the teca_multi_cf_reader. Use one of either --input_file or
    --input_regex.

--input_regex arg
    a teca_cf_reader regex identifying the set of NetCDF CF2 files to process. When present data is
    read using the teca_cf_reader. Use one of either --input_file or --input_regex.

--wind_files arg
    a synonym for --input_regex.

--track_file_out arg
    file path to write cyclone tracks with size (tracks_size.bin)

--wind_u_var arg
    name of variable with wind x-component (UBOT)

--wind_v_var arg
    name of variable with wind y-component (VBOT)

--track_mask arg
    An expression to filter tracks by ()

--number_of_bins arg
    number of bins in the radial wind decomposition (32)

--profile_type arg
    radial wind profile type. max or avg (avg)

--search_radius arg
    size of search window in degrees (6)

--first_track arg
    first track to process

--last_track arg
    last track to process

--n_threads arg
    Sets the thread pool size on each MPI rank. When the default value of -1 is used TECA will
    coordinate the thread pools across ranks such each thread is bound to a unique physical core.

--help
    displays documentation for application specific command line options

--advanced_help
    displays documentation for algorithm specific command line options

--full_help
    displays both basic and advanced documentation together


Examples
~~~~~~~~

A large CAM5 dataset
^^^^^^^^^^^^^^^^^^^^
.. code-block:: bash

    #!/bin/bash

    #SBATCH -A m1517
    #SBATCH -C knl
    #SBATCH -t 00:30:00
    #SBATCH -q debug
    #SBATCH -N 22

    module swap PrgEnv-intel PrgEnv-gnu

    module use  /global/cscratch1/sd/loring/teca_testing/deps/cf_reader_performance/modulefiles/
    module load teca/cf_reader_performance

    # if on KNL. avoid an error about too many cores in OpenBLAS (used by numpy)
    export OMP_NUM_THREADS=1

    data_dir=/global/cscratch1/sd/mwehner/machine_learning_climate_data/All-Hist/CAM5-1-0.25degree_All-Hist_est1_v3_run1/h2

    # run the wind radii calculation
    time srun -N ${SLURM_NNODES} -n 1448 \
        teca_tc_wind_radii --input_regex ${data_dir}/'^CAM5.*\.nc$' \
            --track_file ${data_dir}/../TECA2/tracks_CAM5-1-2_025degree_All-Hist_est1_v3_run1.bin \
            --track_file_out ./wind_tracks_CAM5-1-2_025degree_All-Hist_est1_v3_run1.bin

This script shows computing the radial wind profiles for the 1448 tracks that
were detected in the run shown in the :ref:`teca_tc_detect`, :ref:`example<tc_cam5>` above.

.. _teca_tc_stats:

teca_tc_stats
-------------------------------------------
The statistics stage can be used to compute a variety of statistics on detected
cyclones. It generates a number of plots and tables and it can be ran on the
login nodes. The most common options are the input file and output prefix.

Inputs
~~~~~~
A table of TC tracks generated by the :ref:`teca_tc_detect` application.

Outputs
~~~~~~~

+--------------------+--------------------------------------------------------------+
| **Class table**    || A table containing a row for each track, with columns       |
|                    || containing location of the first point in the track; min    |
|                    || and max of detector parameters; track length and duration;  |
|                    || a column indicating the basin of origin; ACE and PDI        |
|                    || columns, and Saffir-Simspon categorization                  |
+--------------------+--------------------------------------------------------------+
| **Summary table**  | Annual counts of storms by Saffir-Simpson category.          |
+--------------------+--------------------------------------------------------------+
| **Plots**          || Plots of the tabular data by year and region.               |
|                    ||                                                             |
|                    || Categorical Distribution                                    |
|                    ||     Produces a histogram containing counts of each class    |
|                    ||     of storm on the Saffir-Simpson scale. See figure        |
|                    ||     :numref:`ass`.                                          |
|                    ||                                                             |
|                    || Categorical Monthly Breakdown                               |
|                    ||     Produces histogram for each year that shows             |
|                    ||     the breakdown by month and Saffir-Simpson category.     |
|                    ||     See figure :numref:`mon`.                               |
|                    ||                                                             |
|                    || Categorical Regional Breakdown                              |
|                    ||     Produces a histogram for each year that shows           |
|                    ||     breakdown by region and Saffir-Simpson category. See    |
|                    ||     figure :numref:`reg`.                                   |
|                    ||                                                             |
|                    || Categorical Regional Trend                                  |
|                    ||     Produces a histogram for each geographic region that    |
|                    ||     shows trend of storm count and Saffir-Simpson category  |
|                    ||     over time. See figure :numref:`trend`                   |
|                    ||                                                             |
|                    || Parameter Distributions                                     |
|                    ||     Produces box and whisker plots for each year for a      |
|                    ||     number of detector parameters. See figure               |
|                    ||     :numref:`dist`.                                         |
|                    ||                                                             |
+--------------------+--------------------------------------------------------------+

Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~
The command line arguments to the stats stage are:

positional
^^^^^^^^^^

tracks_file
    A required positional argument pointing to the file containing TC storm tracks.

output_prefix
    Required positional argument declaring the prefix that is prepended to all output files.

optional
^^^^^^^^

--help
    prints documentation for the command line options.

-d, --dpi
    Sets the resolution of the output images.

-i, --interactive
    Causes the figures to open immediately in a pop-up window.

-a, --ind_axes
    Normalize y axes in the subplots allowing for easier inter-plot comparison.



Analyses produced by the stats stage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Stats output 1

    * - .. _dist:

        .. figure:: ./images/1990s_cam5_amip1_hero_stats/stats_1990s_3hr_distribution_1.png

            Parameter Dist.

      - .. _ass:

        .. figure:: ./images/1990s_cam5_amip1_hero_stats/stats_1990s_3hr_annual_saphire_simpson_distribution_1.png

            Categorical Dist.

    * - .. _mon:

        .. figure:: ./images/1990s_cam5_amip1_hero_stats/stats_1990s_3hr_monthly_breakdown_1.png

            Monthly Breakdown

      - .. _reg:

        .. figure:: ./images/1990s_cam5_amip1_hero_stats/stats_1990s_3hr_regional_break_down_1.png

            Regional Breakdown

    * - .. _trend:

        .. figure:: ./images/1990s_cam5_amip1_hero_stats/stats_1990s_3hr_regional_trend_1.png

            Regional trend.

      -

.. _basins:

.. figure:: ./images/basin_poly_validation.png

    Basin Definitions and Cyclogenesis Plot


Example
~~~~~~~
An example of running the stats stage is:

.. code-block:: bash

   teca_tc_stats tracks_1990s.bin stats/stats_1990s


.. _teca_tc_trajectory_scalars:

teca_tc_trajectory_scalars
--------------------------
The trajectory scalars application can be used to plot detection parameters for
each storm in time. For each track, a geographic plot of the track along with the
detector parameters is generated.  The application can be run in parallel with
a maximum concurrency equal to the number of tracks.

.. _traj_scalar:

.. figure:: ./images/track_scalars_1990s_3hr_mdd_4800_000280.png
    :width: 5 in

    The trajectory scalars application plots cyclone properties over time.

Inputs
~~~~~~
A table of TC tracks generated by :ref:`teca_tc_detect` optionally including
storm size data generated by :ref:`teca_tc_wind_radii`.

Outputs
~~~~~~~
An image per track showing the path of the track on a map. If storm size data
is present that is plotted as bands around the track. A plot showing each
detector parameter time evolution is included.


Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~

positional
^^^^^^^^^^
tracks_file
    A required positional argument pointing to the file containing TC storm tracks.

output_prefix
    A required positional argument declaring the prefix that is prepended to all output files.

optional
^^^^^^^^
-h, --help
    prints documentation for the command line options.

-d, --dpi
    Sets the resolution of the output images.

-i, --interactive
    Causes the figures to open immediately in a pop-up window.

--first_track
    Id of the first track to process

--last_track
    Id of the last track to process

--texture
    An image containing a map of the Earth to plot the tracks on.


Example
~~~~~~~

.. code-block:: bash

    mpiexec -np 10 ./bin/teca_tc_trajectory_scalars   \
        --texture ../../TECA_data/earthmap4k.png      \
        tracks_1990s_3hr_mdd_4800.bin                 \
        traj_scalars_1990s_3hr_mdd_4800


.. _teca_tc_wind_radii_stats:

teca_tc_wind_radii_stats
----------------------------------------------
The wind radii stats application can be used to plot the wind radii
distributions. If one takes the category 1 radius to be the size of
the cyclone then category 1 radii distribution could be used to compare
storm size between two or more runs.

.. _wind_rad_stats :

.. figure:: ./images/wind_radii_stats.png
    :width: 5 in

    The wind radii stats application plots distribution of wind radii.

Inputs
~~~~~~
A table containing TC tracks with storm size data generated by the :ref:`teca_tc_wind_radii` application.

Outputs
~~~~~~~
Two plots showing a distrubtions of the radii.

Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~

positional
^^^^^^^^^^

tracks_file
    A required positional argument pointing to the file containing TC storm tracks.

output_prefix
    Required positional argument declaring the prefix that is prepended to all output files.

optional
^^^^^^^^

--help
    prints documentation for the command line options.

-d, --dpi
    Sets the resolution of the output images.

-i, --interactive
    Causes the figures to open immediately in a pop-up window.

--wind_column
    Name of the column to load instantaneous max wind speeds from.

Example
~~~~~~~

.. code-block:: bash

    teca_tc_wind_radii_stats \
        wind_radii_1990s_3hr_mdd_4800_ed.bin wind_radii_stats_ed/

.. _teca_event_filter:

teca_event_filter
----------------------------------------
The event filter application lets one remove rows from an input table that do
not fall within specified geographic and/or temporal bounds. This gives one the
capability to zoom into a specific storm, time period, or geographic region for
detailed analysis.

Inputs
~~~~~~
A table containing TC tracks.


Outputs
~~~~~~~
A table with TC tracks matching the bounds query.

Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~

in_file
    A required positional argument pointing to the input file.

out_file
    A required positional argument pointing where the output should be written.

-h, --help
    prints documentation for the command line options.

--time_column
    name of column containing time axis

--start_time
    filter out events occurring before this time

--end_time
    filter out events occurring after this time

--step_column
    name of column containing time steps

--step_interval
    filter out time steps modulo this interval

--x_coordinate_column
    name of column containing event x coordinates

--y_coordinate_column
    name of column containing event y coordinates

--region_x_coords
    x coordinates defining region to filter

--region_y_coords
    y coordinates defining region to filter

--region_sizes
    sizes of each of the regions


Example
~~~~~~~
.. code-block:: bash

    teca_event_filter --start_time=1750 --end_time=1850                       \
        --region_x_coords 260 320 320 260 --region_y_coords 10 10 50 50       \
        --region_sizes 4 --x_coordinate_column lon --y_coordinate_column lat  \
        candidates_1990s_3hr.bin filtered.bin

.. _teca_temporal_reduction:

teca_temporal_reduction
-----------------------
The temporal reduction application is an MPI+threads parallel code that
applies a reduction operator on the time axis of a NetCDF CF2 dataset.
The reduction can be applied over a number of specific intervals, for instance
daily, and monthly intervals. Minimum, maximum, and average operators are
supported. A rule of thumb for when running in parallel one should size the
job such that there is a rank per interval. For instance, to apply a
reduction that computes a monthly time series from a 6 hourly time series
spanning 10 years one would use 120 ranks. In this case the reduction
interval is monthly and one arrives at 120 MPI ranks by multiplying:
12 months/year * 10 years. The procedure for selecting the number of MPI
ranks for other intervals and datasets with different time spans is similar.

Inputs
~~~~~~
A mesh with one temporal resolution.

Outputs
~~~~~~~
A mesh with a reduced temporal resolution.


Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~
The most common command line options are:


--input_regex arg
    regex matching the desired set of input NetCDF CF2
    files

--interval arg
    interval to reduce the time axis to. One of daily, or monthly (monthly)

--operator arg
   reduction operator to use. One of minimum, maximum, or
   average (average)

--point_arrays
    A list of point centered arrays to process.

--fill_value
    A value that identifies missing or invalid data.  Specifying the fill value
    on the command line overrides array specific fill values stored in the
    file. (None)

--ignore_fill_value
    Boolean flag that enables missing or invalid value handling. When enabled
    NetCDF CF conventions are used to determine fill value. Alternativley one
    can explicitly provide a fill value on the command line via the
    --fill_value argument. (True).

--output_file
    file pattern for writing output netcdf files. %t% will
    be replaced by a date/time string or time step. See
    the teca_cf_writer for more information.

--steps_per_file
    number of time steps to write to each output file.  (12)

--x_axis_variable
    name of the variable to use for x-coordinates (lon)

--y_axis_variable
    name of the variable to use for y-coordinates (lat)

--z_axis_variable
    name of the variable to use for z-coordinates ()

--t_axis_variable
    name of the variable to use for t-coordinates (time)

--n_threads
    Number of threads to use when stremaing the reduction (2)

--verbose
    enable verbose mode. (1)

For 3D data one will need to use --z_axis_variable to specify the name of
the array to use as vertical coordinates.

Example
~~~~~~~
The following SLURM batch script was used to calculate daily averages from a 3
hourly input dataset. The first 10 years of the 2000's was selected using a
regular expression.

.. code-block:: bash

    #!/bin/bash
    #SBATCH --account=m1517
    #SBATCH --qos=debug
    #SBATCH --time=00:30:00
    #SBATCH --nodes=10
    #SBATCH --tasks-per-node=32
    #SBATCH --constraint=haswell

    export PYTHONUNBUFFERED=1

    module swap PrgEnv-intel PrgEnv-gnu

    module use /global/common/software/m1517/users/taobrien/modulefiles
    module load teca/develop_haswell

    #export HDF5_USE_FILE_LOCKING=FALSE
    #data_dir=/global/cfs/cdirs/m3522/cmip6/CMIP6_hrmcol/HighResMIP/CMIP6/HighResMIP/ECMWF/ECMWF-IFS-LR/highresSST-present/r1i1p1f1/6hrPlevPt/psl/gr/v20170915/

    data_dir=.
    regex=psl_6hrPlevPt_ECMWF-IFS-LR_highresSST-present_r1i1p1f1_gr_198'.*\.nc$'

    time teca_metadata_probe --input_regex ${data_dir}/${regex}

    time srun -n 320 teca_temporal_reduction \
        --input_regex ${data_dir}/${regex} \
        --interval daily --operator average \
        --arrays psl --steps_per_file 366 \
        --output_file psl_daily_avg_loc_%t%.nc

.. _teca_deeplab_ar_detect:

teca_deeplab_ar_detect
-----------------------
The Deeplabv3+ AR detection application identifies AR's using a machine
learning based approach. It is currently under development and released for
experimental use and as a proof of concept illustrating the use of machine
learning in TECA. For the time being, users are strongly encouraged to make use
of the :ref:`teca_bayesian_ar_detect` application.

Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~

--help
    show this help message and exit

--input_file INPUT_FILE
    a teca_multi_cf_reader configuration file identifying the set of NetCDF CF2 files to process.
    When present data is read using the teca_multi_cf_reader. Use one of either --input_file or
    --input_regex. (default: None)

--input_regex INPUT_REGEX
    a teca_cf_reader regex identifying the set of NetCDF CF2 files to process. When present data is
    read using the teca_cf_reader. Use one of either --input_file or --input_regex. (default: None)

--output_file OUTPUT_FILE
    A path and file name pattern for the output NetCDF files. %t% is replaced with a human readable
    date and time corresponding to the time of the first time step in the file. Use --date_format to
    change the formatting (default: ar_detect_%t%.nc)

--ivt IVT
    name of variable with integrated vapor transport (IVT) magnitude (default: IVT)

--n_threads N_THREADS
    Sets the thread pool size on each MPI rank. When the default value of -1 is used TECA will
    coordinate the thread pools across ranks such each thread is bound to a unique physical core.
    (default: -1)

--binary_ar_threshold BINARY_AR_THRESHOLD
    probability threshold for segmentingar_probability to produce ar_binary_tag (default:
    0.6666666666666666)

--pytorch_deeplab_model PYTORCH_DEEPLAB_MODEL
    the pretrained deeplabv3plus model file (default: None)

--t_axis_variable T_AXIS_VARIABLE
    time dimension name (default: None)

--t_calendar T_CALENDAR
    time calendar (default: None)

--t_units T_UNITS
    time unit (default: None)

--filename_time_template FILENAME_TIME_TEMPLATE
    filename time template (default: None)

--date_format DATE_FORMAT
    A strftime format used when encoding dates into the output file names (%F-%HZ). %t% in the file
    name is replaced with date/time of the first time step in the file using this format specifier.
    (default: None)

--first_step FIRST_STEP
    first time step to process (default: None)

--last_step LAST_STEP
    last time step to process (default: None)

--steps_per_file STEPS_PER_FILE
    number of time steps per output file (default: None)

--start_date START_DATE
    first time to process in YYYY-MM-DD hh:mm:ss format (default: None)

--end_date END_DATE
    end time to process in YYYY-MM-DD hh:mm:ss format (default: None)

Examples
~~~~~~~~~
.. code-block:: bash

    teca_deeplabv3p_ar_detect                                        \
        --input_regex "${data_dir}/ARTMIP_MERRA_2D_2017-05.*\.nc$"   \
        --pytorch_deeplab_model ${model_dir}/cascade_deeplab_IVT.pt

Because torch is multithreaded it's recommended to spread MPI ranks out such
that each rank has exclusive access to a number of cores. The application manages
torch such that load-balancing occurs automatically.


.. _teca_convert_table:

teca_convert_table
-------------------
The teca_convert_table application can be used to convert tabular data in
between known file formats. Its most often used to convert tracks generated
with :ref:`teca_tc_detect` from TECA's binary file format into CSV or NetCDF
format that can be easily ingested by other tools. By default the input and output file
formats are determined from the file extension.


Inputs
~~~~~~
The file name of a table to convert, followed by the file name to write the
result to. By default the extension of the file names are used to determine the
file formats to use. The recognized file formats are: *.bin*, *.csv*, and
*.nc*.

Outputs
~~~~~~~
The converted table.

Command Line Options
~~~~~~~~~~~~~~~~~~~~~

positional
^^^^^^^^^^
in_file
    the path to the table to convert

out_file
    the path to write the converted table to.

optional
^^^^^^^^^^
--help
    show this help message and exit

--in_format
    The file format to assume on the input. One of: auto, bin, csv. (auto)

--out_format
    The file formate to use in the output. One of: auto, bin, csv, netcdf. (auto)

--select
    a logical expression on table columns. Rows where this evaluates to true are
    passed to the output. All other rows are skipped.

Examples
~~~~~~~~

.. code-block:: bash


   teca_convert_table tracks.bin tracks.nc

This example converts a table stored in TECA's binary format to NetCDF. To covnert
to CSV instead one would change the extension from *.nc* to *.csv*.

