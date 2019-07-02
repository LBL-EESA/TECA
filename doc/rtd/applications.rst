TECA Applications
=================
TECA's command line applications deliver the highest perfromance, while
providing a good deal of flexibility for common use cases.  This section
describes how to run TECA's command line applciations. If you need more
felxibility or functionalty not packaged in a command line application consider
using our Python scripting capabilities.


.. _candidates_and_tracks:

.. figure:: ./images/candidates_and_tracks.png

    Cyclone tracks plotted with 850 mb wind speed and integrated moisture.


Tropical Cyclone Detector
--------------------------

The cyclone detector is an MPI+threads parallel map-reduce based application
that identifies tropical cyclone tracks in NetCDF-CF2 climate data. The
application is comprised of a number of stages that are run in succession
producing tables containing cyclone tracks. The tracks then can be visualized
or further analyzed using the TECA TC statistics application, TECA's Python
bindings, or the TECA ParaView plugin.

Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~
The most common command line options are:

--help
    prints documentation for the most common options. MPI programs, such as
    teca\_tc\_detect aren't allowed to run on the login noes at NERSC. For
    this reason to use \textit{--help} you'll need to obtain a compute node
    via \textit{salloc} first.

--full\_help
    prints documentation for all options. See \textit{--help} notes.

--input\_regex
    this is how you tell TECA what files are in the dataset. We use the grep
    style regex, which must be quoted with single ticks to protect it from the
    shell. Regex meta characters present in the file name must be escaped with
    a \textbackslash. An example of an input regex which includes all .nc files
    is: '.*\textbackslash.nc\$'. If instead one wanted to grab only files from
    2004-2005 then '.*\textbackslash.200[45 .*\textbackslash.nc\$' would do the
    trick. For the best performance, specify the smallest set of files needed
    to achieve the desired result. Each of the files will be opened in order to
    scan the time axis.

--start\_date
    an optional way to further specify the time range to process. The accepted
    format is a CF style human readable date spec such as YYYY-MM-DD hh:mm:ss.
    Because of the space in between day and hour spec quotes must be used. For
    example "2005-01-01 00:00:00". Specifying a start date is optional,  if
    none is given then all of the time steps in all of the files specified in
    the \textit{--input\_regex} are processed.

--end\_date
    see \textit{--start\_date}. this is has a similar purpose in restricting
    the range of time steps processed.

--candidate\_file
    a file name specifying where to write the storm candidates to. If not
    specified result will be written to candidates.bin in the current working
    directory. One sets the output format via the extension. Supported formats
    include csv, xlsx, and bin.

--track\_file
    a file name specifying where to write the detected storm tracks. If not
    specified the tracks are written to a file named tracks.bin in the current
    working directory. See \textit{--candidate\_file} for information about the
    supported formats.


Example
~~~~~~~
Once on Edison load the TECA module

.. code-block:: bash

    module load teca

note that there are multiple versions installed, just use the latest and
greatest as they become available.

Processing an entire dataset is straight forward once you know how many cores
you want to run on. You will launch teca\_tc\_detect, the tropical cyclone
application, from a SLURM batch script. A batch script is provided below.

TECA can process any size dataset on any number of compute cores. However, the
fastest results are attained when there is 1 time step per core. In order to
set this up one must determine how many time steps there are and write the
SLURM batch script accordingly. The teca\_metadata\_probe command line
application can be used for this purpose. When executed with the same
\textit{--input\_regex} and optionally the \textit{--start\_date} and or
\textit{--end\_date} options that will be used in the cyclone detection run it
will print out the information needed to configure a 1 to 1 (time steps to
cores) run. The metadata probe is a serial application and can be run on the
login nodes.

.. code-block:: bash

    teca_metadata_probe --input_regex '.*\.199[0-9].*\.nc$'
    
    # A total of 29200 steps available in 3650 files. Using the noleap calendar.
    # Times are specified in units of days since 1979-01-01 00:00:00. The available
    # times range from 1990-1-1 3:0:0 (4015.12) to 2000-1-1 0:0:0 (7665).

With the number of time steps in hand one can set up the SLURM batch script for
the run. The following batch script, named \textit{1990s.sh}, processes the
entire decade of the 1990's. The teca\_metadata\_probe was used to determine
that there are 29200 time steps. The srun command is used to launch the cyclone
detector on 29200 cores.

.. code-block:: bash

    #!/bin/bash -l
    
    #SBATCH -p regular
    #SBATCH -N 1217
    #SBATCH -t 00:30:00
    
    data_dir=/scratch2/scratchdirs/prabhat/TCHero/data
    files_regex=cam5_1_amip_run2'\.cam2\.h2\.199[0-9].*.nc$'
    
    srun -n 29200 teca_tc_detect                    \
        --input_regex ${data_dir}/${files_regex}    \
        --candidate_file candidates_1990s.bin       \
        --track_file tracks_1990s.bin

Finally, the batch script must be submitted to the batch system requesting the
appropriate number of nodes. In this case the command is:

.. code-block:: bash

    sbatch ./1990s.sh

For the $\frac{1}{4}$ degree resolution dataset when processing latitudes
between -90 to 90 the detector runs in approx 15 min. Detector run time could
be reduced by subsetting in latitude (see \textit{--lowest\_lat},
\textit{--highest\_lat} options). Note that as the number of files in the
dataset increases the metadata phase takes more time. You can use
teca\_metadata\_probe to get a sense of how much more and extend the run time
accordingly.

Tropical Cyclone Trajectories
------------------------------
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

The trajectory stage runs after the map-reduce candidate detection stage and
generates cyclone storm tracks. The TC detector described above invokes the
trajectory stage automatically, however it can also be run independently on the
candidate stage output. The trajectory stage can be run from the login nodes.

Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~
The most commonly used command line arguments to the trajectory stage are:

--help
    prints documentation for the most common options.

--full\_help
    prints documentation for all options. See \textit{--help} notes.

--candidate\_file
    a file name specifying where to read the storm candidates from. 

--track\_file
    a file name specifying where to write the detected storm tracks. If not
    specified the tracks are written to a file named tracks.bin in the current
    working directory. One sets the output format via the extension. Supported
    formats include csv, xlsx, and bin.



Example
~~~~~~~
An example of running the trajectory stage is:

.. code-block:: bash

    teca_tc_trajectory \
        --candidate_file candidates_1990s.bin       \
        --track_file tracks_1990s.bin

the file \textit{tracks\_1990s.bin} will contain the list of storm tracks.

TC Wind Radii
--------------------------
The wind radii application can be used to compute wind radii from track data in
parallel. For each point on each track a radial profile is computed over a
number of angular intervals. The radial profiles are used to compute distance
from the storm center to the first downward crossing of given wind speeds. The
default wind speeds are the3 Saffir-Simpson transitions. Additionally distance
to the peak wind speed and peak wind speed are recorded.  A new table is
produced containing the data. The TC trajectory scalars application, TC stats
application and ParaView plugin can be used to further analyze the data.

Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~
The most commonly used command liine arguments are:

--track\_file
    file path to read the cyclone from (tracks.bin)

--wind\_files
    regex matching simulation files containing wind fields ()

--track\_file\_out
    file path to write cyclone tracks with size (tracks\_size.bin)

--wind\_u\_var
    name of variable with wind x-component (UBOT)

--wind\_v\_var
    name of variable with wind y-component (VBOT)

--track\_mask
    expression to filter tracks by ()

--n\_theta
    number of points in the wind profile in the theta direction (32)

--n\_r
    number cells in the wind profile in radial direction (32)

--profile\_type
    radial wind profile type. max or avg (avg)

--search\_radius
    size of search window in deg lat (6)

see --help and --full\_help for more information.

Example
~~~~~~~
The following examples shows computation of wind radii for a decades worth of
tracks using 128 cores on NERSC Cori.

.. code-block:: bash

    module load teca
    sbatch wind_radii_1990s.sh

where the contents of \textit{wind\_radii\_1990s.sh} are as follows

.. code-block:: bash

    #!/bin/bash -l
    #SBATCH -p debug
    #SBATCH -N 4
    #SBATCH -t 00:30:00
    #SBATCH -C haswell
    
    data_dir=/global/cscratch1/sd/mwehner/cylones_ensemble/cam5_1_amip_run2/ncfiles
    files_regex=${data_dir}/cam5_1_amip_run2'\.cam2\.h2\.199[0-9].*\.nc$'
    track_file=tracks_1990s_3hr_mdd_4800.bin
    track_file_out=wind_radii_1990s_3hr_mdd_4800_co.bin
    
    srun -n 4 --ntasks-per-node=1 \
        teca_tc_wind_radii --n_threads 32 --first_track 0 \
        --last_track -1 --wind_files ${files_regex} --track_file ${track_file} \
        --track_file_out ${track_file_out}

Tropical Cyclone Statistics
---------------------------
The statistics stage can be used to compute a variety of statistics on detected
cyclones. It generates a number of plots and tables and it can be ran on the
login nodes. The most common options are the input file and output prefix. 

Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~
The command line arguments to the stats stage are:

tracks\_file
    A required positional argument pointing to the file containing TC storm tracks.

output\_prefix
    Required positional argument declaring the prefix that is prepended to all output files.

--help
    prints documentation for the command line options.

-d, --dpi
    Sets the resolution of the output images.

-i, --interactive
    Causes the figures to open immediately in a pop-up window.

-a, --ind\_axes
    Normalize y axes in the subplots allowing for easier inter-plot comparison.


Analysis
~~~~~~~~
The following analysis are performed by the stats stage:

Classification Table
    Produces a table containing cyclogenisis information, Saphir-Simpson
    category, and the min/max of a number of detection parameters.

Categorical Distribution
    Produces a histogram containing counts of each class of storm on the
    Saphir-Simpson scale. See figure :numref:`ass`.

Categorical Monthly Breakdown
    Produces histogram for each year that shows the breakdown by month and
    Saphir-Simpson category. See figure :numref:`mon`.

Categorical Regional Breakdown
    Produces a histogram for each year that shows breakdown by region and
    Saphir-Simpson category. See figure :numref:`reg`.

Categorical Regional Trend
    Produces a histogram for each geographic region that shows trend of storm
    count and Saphir-Simpson category over time. See figure :numref:`trend`

Parameter Distributions
    Produces box and whisker plots for each year for a number of detector
    parameters. See figure :numref:`dist`.


Example
~~~~~~~
An example of running the stats stage is:

.. code-block:: bash

   teca_tc_stats tracks_1990s.bin stats/stats_1990s


TC Trajectory Scalars
--------------------------

.. _traj_scalar:

.. figure:: ./images/track_scalars_1990s_3hr_mdd_4800_000280.png
    :width: 3 in

    The trajectory scalars application plots cyclone properties over time.

The trajectory scalars application can be used to plot detection parameters for
each storm in time. The application can be run in parallel.

Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~

tracks\_file
    A required positional argument pointing to the file containing TC storm tracks.

output\_prefix
    A required positional argument declaring the prefix that is prepended to all output files.

-h, --help
    prints documentation for the command line options.

-d, --dpi
    Sets the resolution of the output images.

-i, --interactive
    Causes the figures to open immediately in a pop-up window.

--first\_track
    Id of the first track to process

--last\_track
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

TC Wind Radii Stats
-------------------
The wind radii stats application can be used to generate summary statistics
describing the wind radii distributions.

.. _wind_rad_stats :

.. figure:: ./images/wind_radii_stats.png
    :width: 3 in

    The wind radii stats application plots distribution of wind radii.

Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~

tracks\_file
    A required positional argument pointing to the file containing TC storm tracks.

output\_prefix
    Required positional argument declaring the prefix that is prepended to all output files.

--help
    prints documentation for the command line options.

-d, --dpi
    Sets the resolution of the output images.

-i, --interactive
    Causes the figures to open immediately in a pop-up window.

--wind\_column
    Name of the column to load instantaneous max wind speeds from.

Example
~~~~~~~

.. code-block:: bash

    teca_tc_wind_radii_stats \
        wind_radii_1990s_3hr_mdd_4800_ed.bin wind_radii_stats_ed/

Event Filter
------------
The event filter application lets one remove rows from an input table that do
not fall within specified geographic and/or temporal bounds. This gives one the
capability to zoom into a specific storm, time period, or geographic region for
detailed analysis.

Command Line Arguments
~~~~~~~~~~~~~~~~~~~~~~

in\_file
    A required positional argument pointing to the input file.

out\_file
    A required positional argument pointing where the output should be written.

-h, --help
    prints documentation for the command line options.

--time\_column
    name of column containing time axis

--start\_time
    filter out events occurring before this time

--end\_time
    filter out events occurring after this time

--step\_column
    name of column containing time steps

--step\_interval
    filter out time steps modulo this interval

--x\_coordinate\_column
    name of column containing event x coordinates

--y\_coordinate\_column
    name of column containing event y coordinates

--region\_x\_coords
    x coordinates defining region to filter

--region\_y\_coords
    y coordinates defining region to filter

--region\_sizes
    sizes of each of the regions


Example
~~~~~~~

.. code-block:: bash

    teca_event_filter --start_time=1750 --end_time=1850                       \
        --region_x_coords 260 320 320 260 --region_y_coords 10 10 50 50       \
        --region_sizes 4 --x_coordinate_column lon --y_coordinate_column lat  \
        candidates_1990s_3hr.bin filtered.bin

