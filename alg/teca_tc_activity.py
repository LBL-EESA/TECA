import sys
import teca_py
import numpy as np

class teca_tc_activity:
    """
    Computes summary statistics, histograms on sorted, classified,
    TC trajectory output.
    """
    @staticmethod
    def New():
        return teca_tc_activity()

    def __init__(self):
        self.basename = 'activity'
        self.dpi = 100
        self.interactive = False
        self.rel_axes = True
        self.color_map = None

        self.impl = teca_py.teca_programmable_algorithm.New()
        self.impl.set_number_of_input_connections(1)
        self.impl.set_number_of_output_ports(1)
        self.impl.set_execute_callback(self.get_tc_activity_execute(self))

    def __str__(self):
        return 'basename=%s, dpi=%d, interactive=%s, rel_axes=%s'%( \
            self.basename, self.dpi, str(self.interactive), str(self.rel_axes))

    def set_basename(self, basename):
        """
        All output files are prefixed by the basename. default 'stats'
        """
        self.basename = basename

    def set_dpi(self, dpi):
        """
        set the DPI resolution for image output. default 100
        """
        self.dpi = dpi

    def set_interactive(self, interactive):
        """
        plots are rendered to a an on screen window when enabled.
        when disabled plots are written directly to disk. default False
        """
        self.interactive = interactive

    def set_rel_axes(self, rel_axes):
        """
        When enabled y-axes in subplots are scaled to reflect the max
        value across all the plots making it easy to compare between plots.
        when disabled matplotlib's default scaling is used. default True
        """
        self.rel_axes = rel_axes

    def set_color_map(self, color_map):
        """
        set colormap to color plots by
        """
        self.color_map = color_map

    def set_input_connection(self, obj):
        """
        set the input
        """
        self.impl.set_input_connection(obj)

    def get_output_port(self):
        """
        get the output
        """
        return self.impl.get_output_port()

    def update(self):
        """
        execute the pipeline from this algorithm up.
        """
        self.impl.update()

    @staticmethod
    def get_tc_activity_execute(state):
        """
        return a teca_algorithm::execute function. a closure
        is used to gain state.
        """
        def execute(port, data_in, req):
            """
            expects the output of the teca_tc_classify algorithm
            generates a handful of histograms, summary statistics,
            and plots. returns summary table with counts of annual
            storms and their categories.
            """
            global plt
            global plt_mp
            global plt_tick

            import matplotlib.pyplot as plt
            import matplotlib.patches as plt_mp
            import matplotlib.ticker as plt_tick

            # store matplotlib state we modify
            legend_frame_on_orig = plt.rcParams['legend.frameon']

            # tweak matplotlib slightly
            plt.rcParams['figure.max_open_warning'] = 0
            plt.rcParams['legend.frameon'] = 1

            # get the input table
            in_table = teca_py.as_teca_table(data_in[0])
            if in_table is None:
                # TODO if this is part of a parallel pipeline then
                # only rank 0 should report an error.
                sys.stderr.write('ERROR: empty input, or not a table\n')
                return teca_table.New()

            time_units = in_table.get_time_units()

            # get the columns of raw data
            year = in_table.get_column('year').as_array()
            region_id = in_table.get_column('region_id').as_array()
            region_name = in_table.get_column('region_name')
            region_long_name = in_table.get_column('region_long_name')
            start_y = in_table.get_column('start_y').as_array()

            ACE = in_table.get_column('ACE').as_array()
            PDI = in_table.get_column('PDI').as_array()

            # organize the data by year month etc...
            regional_ACE = []
            regional_PDI = []

            # get unique for use as indices etc
            uyear = sorted(set(year))
            n_year = len(uyear)
            ureg = sorted(set(zip(region_id, region_name, region_long_name)))


            teca_tc_activity.accum_by_year_and_region(uyear, \
                ureg, year, region_id, start_y, ACE, regional_ACE)

            teca_tc_activity.accum_by_year_and_region(uyear, \
                ureg, year, region_id, start_y, PDI, regional_PDI)

            # now plot the organized data in various ways
            if state.color_map is None:
                state.color_map = plt.cm.jet

            teca_tc_activity.plot_individual(state, uyear, \
                 ureg, regional_ACE,'ACE', '$10^4 kn^2$')

            teca_tc_activity.plot_individual(state, uyear, \
                ureg, regional_PDI, 'PDI', '$m^3 s^{-2}$')

            teca_tc_activity.plot_cumulative(state, uyear, \
                ureg, regional_ACE, 'ACE', '$10^4 kn^2$')

            teca_tc_activity.plot_cumulative(state, uyear, \
                ureg, regional_PDI, 'PDI', '$m^3 s^{-2}$')

            if (state.interactive):
                plt.show()

            # restore matplot lib global state
            plt.rcParams['legend.frameon'] = legend_frame_on_orig

            # send data downstream
            return in_table
        return execute

    @staticmethod
    def two_digit_year_fmt(x, pos):
        q = int(x)
        q = q - q/100*100
        return '%02d'%q

    @staticmethod
    def accum_by_year_and_region(uyear,ureg,year,region_id,start_y,var,var_out):
        for yy in uyear:
            yids = np.where(year==yy)
            # break these down by year
            yvar = var[yids]
            # break down by year and region
            rr = region_id[yids]
            max_reg = np.max(rr)
            tot = 0
            for r,n,l in ureg:
                rids = np.where(rr==r)
                rvar = yvar[rids]
                var_out.append(np.sum(rvar))
            # south
            shids = np.where(start_y[yids] < 0.0)
            rvar = yvar[shids]
            var_out.append(np.sum(rvar))
            # add northern, southern hemisphere and global regions
            # north
            nhids = np.where(start_y[yids] >= 0.0)
            rvar = yvar[nhids]
            var_out.append(np.sum(rvar))
            # global
            var_out.append(np.sum(yvar))

    @staticmethod
    def plot_individual(state, uyear, ureg, var, var_name, units):
        n_reg = len(ureg) + 3 # add 2 for n & s hemi, 1 for global
        n_year = len(uyear)

        # now plot the organized data in various ways
        n_cols = 3
        wid = 2.5*n_cols

        # plot regions over time
        reg_t_fig = plt.figure()

        rnms = zip(*ureg)[2]
        rnms += ('Southern','Northern','Global')

        n_plots = n_reg + 1
        n_left = n_plots%n_cols
        n_rows = n_plots/n_cols + (1 if n_left else 0)
        wid = 2.5*n_cols
        ht = 2.0*n_rows
        reg_t_fig.set_size_inches(wid, ht)

        max_y_reg = -1
        max_y_hem = -1
        q = 0
        while q < n_reg:
            if q < n_reg-3:
                max_y_reg = max(max_y_reg, np.max(var[q::n_reg]))
            elif q < n_reg-1:
                max_y_hem = max(max_y_hem, np.max(var[q::n_reg]))
            q += 1

        fill_col = [state.color_map(i) for i in np.linspace(0, 1, n_reg)]

        q = 0
        while q < n_reg:
            plt.subplot(n_rows, n_cols, q+1)
            ax = plt.gca()
            ax.grid(zorder=0)
            ax.xaxis.set_major_formatter(plt_tick.FuncFormatter( \
                teca_tc_activity.two_digit_year_fmt))

            plt.plot(uyear, var[q::n_reg],'-',color=fill_col[q],linewidth=2)
            ax.set_xticks(uyear[:] if n_year < 10 else uyear[::2])
            ax.set_xlim([uyear[0], uyear[-1]])

            if state.rel_axes and q < n_reg - 1:
                ax.set_ylim([0, 1.05*(max_y_reg if q < n_reg - 3 else max_y_hem)])
            if (q%n_cols == 0):
                plt.ylabel(units, fontweight='normal', fontsize=10)
            if (q >= (n_reg - n_cols)):
                plt.xlabel('Year', fontweight='normal', fontsize=10)
            plt.title('%s'%(rnms[q]), fontweight='bold', fontsize=11)
            plt.grid(True)

            q += 1

        plt.suptitle('%s Individual Region'%(var_name), fontweight='bold')
        plt.subplots_adjust(wspace=0.35, hspace=0.6, top=0.92)

        plt.savefig('%s_%s_individual_%d.png'%( \
            state.basename, var_name, state.dpi), dpi=state.dpi)

        return

    def plot_cumulative(state, uyear, ureg, var, var_name, units):

        n_reg = len(ureg) + 3 # add 2 for n & s hemi, 1 for global
        n_year = len(uyear)

        rnms = zip(*ureg)[2]
        rnms += ('Southern','Northern','Global')

        fill_col = [state.color_map(i) for i in np.linspace(0, 1, n_reg)]

        # stacked plot by region
        nhsh_stack_fig = plt.figure()
        ax = plt.gca()
        ax.xaxis.set_major_formatter(plt_tick.FuncFormatter( \
            teca_tc_activity.two_digit_year_fmt))

        base = np.zeros(n_year)
        q = 0
        while q < n_reg-3:
            vals = var[q::n_reg]
            bot = base
            top = bot + vals
            ax.fill_between(uyear, bot, top, facecolor=fill_col[q], alpha=0.75, zorder=3)
            ax.plot(uyear, top, color=fill_col[q], linewidth=2, label=rnms[q], zorder=3)
            base = top
            q += 1

        ax.set_xticks(uyear[:] if n_year < 15 else uyear[::2])
        ax.set_xlim([uyear[0], uyear[-1]])
        plt.grid(True, zorder=0)

        ylim = ax.get_ylim()
        ax.set_ylim([0, ylim[1]])

        leg=plt.legend(loc=2, bbox_to_anchor=(1.0, 1.01))
        plt.subplots_adjust(right=0.78)

        plt.ylabel(units)
        plt.xlabel('Year')
        plt.title('%s by Region'%(var_name), fontweight='bold')

        plt.savefig('%s_%s_regions_%d.png'%( \
            state.basename, var_name, state.dpi), dpi=state.dpi)

        # stacked plot of northern, southern hemispheres
        nhsh_stack_fig = plt.figure()
        ax = plt.gca()
        ax.xaxis.set_major_formatter(plt_tick.FuncFormatter( \
            teca_tc_activity.two_digit_year_fmt))

        base = np.zeros(n_year)
        q = n_reg-3
        while q < n_reg-1:
            vals = var[q::n_reg]
            bot = base
            top = bot + vals
            ax.fill_between(uyear, bot, top, facecolor=fill_col[q], alpha=0.75, zorder=3)
            ax.plot(uyear, top, color=fill_col[q], linewidth=2, label=rnms[q], zorder=3)
            base = top
            q += 1

        ax.set_xticks(uyear[:] if n_year < 15 else uyear[::2])
        ax.set_xlim([uyear[0], uyear[-1]])
        plt.grid(True, zorder=0)

        ylim = ax.get_ylim()
        ax.set_ylim([0, ylim[1]])

        leg=plt.legend(loc=2, bbox_to_anchor=(1.0, 1.01))
        plt.subplots_adjust(right=0.78)

        plt.xlabel('Year')
        plt.ylabel(units)
        plt.title('%s by Hemisphere'%(var_name), fontweight='bold')

        plt.savefig('%s_%s_hemispheres_%d.png'%( \
            state.basename, var_name, state.dpi), dpi=state.dpi)

        return

#    def write_table(state, uyear, ureg, var, var_name, units, table_out):
#
#        n_reg = len(ureg) + 3 # add 2 for n & s hemi, 1 for global
#        n_year = len(uyear)
#
#        rnms = zip(*ureg)[2]
#        rnms += ('Southern','Northern','Global')
#
#
#        tab = teca_table.New()
#        tab.declare_columns(['Year'] + rnms,
#            ['l'] + ['d']*n_reg)
#
#        for val in var:
#            if 
#        q = 0
#        while q < n_reg:
#            tmp = var[q::n_reg]
#
#            plt.plot(uyear, ,'-',color=fill_col[q],linewidth=2)
