import sys
import teca_py
import numpy as np

class teca_tc_stats:
    """
    Computes summary statistics, histograms on sorted, classified,
    TC trajectory output.
    """
    @staticmethod
    def New():
        return teca_tc_stats()

    def __init__(self):
        self.basename = 'stats'
        self.dpi = 100
        self.interactive = False
        self.rel_axes = True

        self.impl = teca_py.teca_programmable_algorithm.New()
        self.impl.set_number_of_input_connections(1)
        self.impl.set_number_of_output_ports(1)
        self.impl.set_execute_callback(self.get_tc_stats_execute(self))

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
    def get_tc_stats_execute(state):
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
            import matplotlib.pyplot as plt
            import matplotlib.patches as plt_mp

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
            month = in_table.get_column('month').as_array()
            duration = in_table.get_column('duration').as_array()
            length = in_table.get_column('length').as_array()/1000.0
            category = in_table.get_column('category').as_array()
            region_id = in_table.get_column('region_id').as_array()
            region_name = in_table.get_column('region_name')
            region_long_name = in_table.get_column('region_long_name')
            wind = in_table.get_column('max_surface_wind').as_array()
            press = in_table.get_column('min_sea_level_pressure').as_array()
            start_y = in_table.get_column('start_y').as_array()
            ACE = in_table.get_column('ACE').as_array()

            # organize the data by year month etc...
            annual_cat = []
            annual_count = []
            annual_wind = []
            annual_press = []
            annual_dur = []
            annual_len = []
            annual_ACE = []
            by_month = []
            by_region = []
            totals = []

            # get unique for use as indices etc
            uyear = sorted(set(year))
            n_year = len(uyear)

            ureg = sorted(set(zip(region_id, region_name, region_long_name)))
            n_reg = len(ureg) + 3 # add 2 for n & s hemi, 1 for global

            for yy in uyear:
                yids = np.where(year==yy)
                # break these down by year
                annual_count.append(len(yids[0]))
                annual_cat.append(category[yids])
                annual_wind.append(wind[yids])
                annual_press.append(press[yids])
                annual_dur.append(duration[yids])
                annual_len.append(length[yids])
                annual_ACE.append(ACE[yids])

                # global totals
                tmp = [annual_count[-1]]
                for c in np.arange(0,6,1):
                    cids = np.where(category[yids]==c)
                    tmp.append(len(cids[0]))
                totals.append(tmp)

                # break down by year, month, and category
                mm = month[yids]
                mnum = np.arange(1,13,1)
                monthly = []
                for m in mnum:
                    mids = np.where(mm==m)
                    mcats = category[yids][mids]
                    cats = []
                    for c in np.arange(0,6,1):
                        cids = np.where(mcats==c)
                        cats.append(len(cids[0]))
                    monthly.append(cats)
                by_month.append(monthly)
                # break down by year and region
                rr = region_id[yids]
                max_reg = np.max(rr)
                regional = []
                for r,n,l in ureg:
                    rids = np.where(rr==r)
                    rcats = category[yids][rids]
                    cats = []
                    for c in np.arange(0,6,1):
                        cids = np.where(rcats==c)
                        cats.append(len(cids[0]))
                    regional.append(cats)
                by_region.append(regional)
                # add north and south hemisphere regions
                hemi = []
                nhids = np.where(start_y[yids] >= 0.0)
                cats = category[yids][nhids]
                nhcats = []
                for c in np.arange(0,6,1):
                    cids = np.where(cats==c)
                    nhcats.append(len(cids[0]))
                by_region[-1].append(nhcats)
                shids = np.where(start_y[yids] < 0.0)
                cats = category[yids][shids]
                shcats = []
                for c in np.arange(0,6,1):
                    cids = np.where(cats==c)
                    shcats.append(len(cids[0]))
                by_region[-1].append(shcats)
                # global break down
                gcats = []
                cats = category[yids]
                for c in np.arange(0,6,1):
                    cids = np.where(cats==c)
                    gcats.append(len(cids[0]))
                by_region[-1].append(gcats)

            # dump annual totals
            summary = teca_py.teca_table.New()
            summary.declare_columns(['year', 'total', 'cat 0', \
                 'cat 1', 'cat 2', 'cat 3', 'cat 4', 'cat 5'], \
                 ['i', 'ul', 'i', 'i', 'i', 'i', 'i', 'i'])
            q = 0
            while q < n_year:
                summary << int(uyear[q]) << int(totals[q][0]) \
                    << int(totals[q][1]) << int(totals[q][2]) \
                    << int(totals[q][3]) << int(totals[q][4]) \
                    << int(totals[q][5]) << int(totals[q][6])
                q += 1
            f = open('%s_summary.csv'%(state.basename),'w')
            f.write(str(summary))
            f.close()

            # now plot the organized data in various ways
            n_cols = 3
            n_plots = n_year + 1
            n_left = n_plots%n_cols
            n_rows = n_plots/n_cols + (1 if n_left else 0)
            wid = 2.5*n_cols
            ht = 2.0*n_rows

            # use this color map for Saphir-Simpson scale
            red_cmap = ['#ffd2a3','#ffa749','#ff7c04', \
                '#ea4f00','#c92500','#a80300']

            red_cmap_pats = []
            q = 0
            while q < 6:
                red_cmap_pats.append( \
                    plt_mp.Patch(color=red_cmap[q], label='cat %d'%(q)))
                q += 1

            # plot annual saphir-simpson distribution
            page_no = 1
            cat_fig = plt.figure()
            cat_fig.set_size_inches(wid, ht)

            max_y = 0
            q = 0
            while q < n_year:
                max_y = max(max_y, len(np.where(annual_cat[q]==0)[0]))
                q += 1

            q = 0
            for yy in uyear:
                plt.subplot(n_rows, n_cols, q+1)
                ax = plt.gca()
                ax.grid(zorder=0)
                n,bins,pats = plt.hist(annual_cat[q], bins=np.arange(-0.5, 6.0, 1.0), \
                    facecolor='steelblue', alpha=0.95, edgecolor='black', \
                    linewidth=2, zorder=3)
                j = 0
                while j < 6:
                    pats[j].set_facecolor(red_cmap[j])
                    j += 1
                plt.xticks(np.arange(0,6,1))
                if state.rel_axes:
                    ax.set_ylim([0, max_y*1.05])
                if (q%n_cols == 0):
                    plt.ylabel('Count', fontweight='normal', fontsize=10)
                if (q >= (n_year - n_cols)):
                    plt.xlabel('Category', fontweight='normal', fontsize=10)
                plt.title('%d'%(yy), fontweight='bold', fontsize=11)
                plt.grid(True)

                q += 1

            plt.subplot(n_rows, n_cols, q+1)
            ax = plt.gca()
            ax.grid(zorder=0)
            l = plt.legend(handles=red_cmap_pats, loc=2, bbox_to_anchor=(0.0, 1.0))
            plt.axis('off')

            plt.suptitle('Annual Saphir-Simpson Distribution', fontweight='bold')
            plt.subplots_adjust(hspace=0.4, top=0.92)

            plt.savefig('%s_annual_saphire_simpson_distribution_%d.png'%( \
                state.basename, page_no), dpi=state.dpi)

            # break annual distributions down by month
            mos_fig = plt.figure()
            mos_fig.set_size_inches(wid, ht)

            max_y = 0
            q = 0
            while q < n_year:
                p = 0
                while p < 12:
                    max_y = max(max_y, sum(by_month[q][p]))
                    p += 1
                q += 1

            q = 0
            for yy in uyear:
                plt.subplot(n_rows, n_cols, q+1)
                ax = plt.gca()
                ax.grid(zorder=0)
                # build up a stacked bar chart, each category is a layer
                # copy that cat for all months into a temp array then add
                # it to the plot at the right hight and color.
                mcts = by_month[q]
                bot = np.zeros((12))
                c = 0
                while c < 6:
                    tmp = []
                    p = 0
                    while p < 12:
                        tmp.append(mcts[p][c])
                        p += 1
                    plt.bar(np.arange(1,13,1)-0.375, tmp, width=0.75, bottom=bot, \
                        facecolor=red_cmap[c], edgecolor='k', linewidth=1, \
                        tick_label=['J','F','M','A','M','J','J','A','S','O','N','D'], \
                        zorder=3)
                    bot += tmp
                    c += 1

                plt.xticks(np.arange(1,13,1))
                if state.rel_axes:
                    ax.set_ylim([0, 1.05*max_y])
                if (q%n_cols == 0):
                    plt.ylabel('Count', fontweight='normal', fontsize=10)
                if (q >= (n_year - n_cols)):
                    plt.xlabel('Month', fontweight='normal', fontsize=10)
                plt.title('%d'%(yy), fontweight='bold', fontsize=11)
                plt.grid(True)

                q += 1

            plt.subplot(n_rows, n_cols, q+1)
            ax = plt.gca()
            ax.grid(zorder=0)
            l = plt.legend(handles=red_cmap_pats, loc=2, bbox_to_anchor=(0.0, 1.0))
            plt.axis('off')

            plt.suptitle('Monthly Breakdown', fontweight='bold')
            plt.subplots_adjust(hspace=0.4, top=0.92)

            plt.savefig('%s_monthly_breakdown_%d.png'%( \
                state.basename, page_no), dpi=state.dpi)

            # plot annual counts by region
            reg_fig = plt.figure()
            reg_fig.set_size_inches(wid, ht)

            rcds = list(zip(*ureg))[1]
            rcds += ('NH', 'SH', 'G')

            max_y = 0
            q = 0
            while q < n_year:
                j = 0
                while j < n_reg:
                    max_y = max(max_y, sum(by_region[q][j]))
                    j += 1
                q += 1

            q = 0
            for yy in uyear:
                plt.subplot(n_rows, n_cols, q+1)
                ax = plt.gca()
                ax.grid(zorder=0)
                # build up a stacked bar chart, each category is a layer
                # copy that cat for all months into a temp array then add
                # it to the plot at the right height and color.
                rcnts = by_region[q]
                bot = np.zeros((n_reg))
                c = 0
                while c < 6:
                    tmp = []
                    p = 0
                    while p < n_reg:
                        tmp.append(rcnts[p][c])
                        p += 1

                    plt.bar(np.arange(0,n_reg,1)-0.375, tmp, width=0.75, bottom=bot, \
                        facecolor=red_cmap[c], edgecolor='k', linewidth=1, \
                        tick_label=rcds, \
                        zorder=3)

                    bot += tmp
                    c += 1

                plt.xticks(np.arange(0,n_reg,1), rotation='vertical')
                if state.rel_axes:
                    ax.set_ylim([0, 1.05*max_y])
                if (q%n_cols == 0):
                    plt.ylabel('Count', fontweight='normal', fontsize=10)
                if (q >= (n_year - n_cols)):
                    plt.xlabel('Region', fontweight='normal', fontsize=10)
                plt.title('%d'%(yy), fontweight='bold', fontsize=11)
                plt.grid(True)

                q += 1

            # add the color map legend
            plt.subplot(n_rows, n_cols, q+1)
            ax = plt.gca()
            ax.grid(zorder=0)
            l = plt.legend(handles=red_cmap_pats, loc=2, bbox_to_anchor=(0.0, 1.0))
            plt.axis('off')

            plt.suptitle('Regional Breakdown', fontweight='bold')
            plt.subplots_adjust(wspace=0.3, hspace=0.6, top=0.92)

            plt.savefig('%s_regional_break_down_%d.png'%( \
                state.basename, page_no), dpi=state.dpi)

            # plot annual distributions
            dist_fig = plt.figure()

            wid = n_year*0.65
            dist_fig.set_size_inches(wid, 9.0)

            ax = plt.subplot(5,1,1)
            plt.boxplot(annual_wind, labels=uyear)
            plt.xlabel('Year')
            plt.ylabel('ms^-1')
            plt.title('Peak Instantaneous Wind', fontweight='bold')
            ax.get_yaxis().set_label_coords(-0.1,0.5)

            ax = plt.subplot(5,1,2)
            plt.boxplot(annual_press, labels=uyear)
            plt.xlabel('Year')
            plt.ylabel('Pa')
            plt.title('Min Instantaneous Pressure', fontweight='bold')
            ax.get_yaxis().set_label_coords(-0.1,0.5)

            ax = plt.subplot(5,1,3)
            plt.boxplot(annual_dur, labels=uyear)
            plt.xlabel('Year')
            plt.ylabel('%s'%(time_units.split()[0]))
            plt.title('Track Duration', fontweight='bold')
            ax.get_yaxis().set_label_coords(-0.1,0.5)

            ax = plt.subplot(5,1,4)
            plt.boxplot(annual_len, labels=uyear)
            plt.xlabel('Year')
            plt.ylabel('km')
            plt.title('Track Length', fontweight='bold')
            ax.get_yaxis().set_label_coords(-0.1,0.5)

            ax = plt.subplot(5,1,5)
            #plt.axhline(82,color='k',linestyle='--',alpha=0.25)
            plt.boxplot(annual_ACE, labels=uyear)
            plt.xlabel('Year')
            plt.ylabel('10^4 kn^2')
            plt.title('ACE', fontweight='bold')
            ax.get_yaxis().set_label_coords(-0.1,0.5)

            plt.suptitle('Distributions', fontweight='bold')
            plt.subplots_adjust(hspace=0.72, top=0.93)

            plt.savefig('%s_distribution_%d.png'%( \
                state.basename, page_no), dpi=state.dpi)

            # plot region over time
            reg_t_fig = plt.figure()

            rnms = list(zip(*ureg))[2]
            rnms += ('Northern', 'Southern', 'Global')

            tmp = np.array(uyear)
            tmp = tmp - tmp/100*100
            ynms = []
            for t in tmp:
                ynms.append('%02d'%t)

            n_plots = n_reg + 1
            n_left = n_plots%n_cols
            n_rows = n_plots/n_cols + (1 if n_left else 0)
            wid = 2.5*n_cols
            ht = 2.0*n_rows
            reg_t_fig.set_size_inches(wid, ht)

            reg_by_t = []
            p = 0
            while p < n_reg:
                reg = []
                q = 0
                while q < n_year:
                    reg.append(by_region[q][p])
                    q += 1
                reg_by_t.append(reg)
                p += 1

            max_y_reg = -1
            max_y_hem = -1
            q = 0
            while q < n_reg:
                dat = reg_by_t[q]
                p = 0
                while p < n_year:
                    if q < n_reg-3:
                        max_y_reg = max(max_y_reg, sum(dat[p]))
                    elif q < n_reg-1:
                        max_y_hem = max(max_y_hem, sum(dat[p]))
                    p += 1
                q += 1

            q = 0
            while q < n_reg:
                dat = reg_by_t[q]

                plt.subplot(n_rows, n_cols, q+1)
                ax = plt.gca()
                ax.grid(zorder=0)

                # build up a stacked bar chart, each category is a layer
                # copy that cat for all months into a temp array then add
                # it to the plot at the right height and color.
                bot = np.zeros((n_year))
                c = 0
                while c < 6:
                    tmp = []
                    p = 0
                    while p < n_year:
                        tmp.append(dat[p][c])
                        p += 1

                    plt.bar(np.arange(0,n_year,1)-0.375, tmp, width=0.75, bottom=bot, \
                        facecolor=red_cmap[c], edgecolor='k', linewidth=1, \
                        tick_label=ynms, \
                        zorder=3)

                    bot += tmp
                    c += 1

                plt.xticks(np.arange(0,n_year,1), rotation='vertical')
                if state.rel_axes and q < n_reg - 1:
                    ax.set_ylim([0, 1.05*(max_y_reg if q < n_reg - 3 else max_y_hem)])
                if (q%n_cols == 0):
                    plt.ylabel('Count', fontweight='normal', fontsize=10)
                if (q >= (n_reg - n_cols)):
                    plt.xlabel('Year', fontweight='normal', fontsize=10)
                plt.title('%s'%(rnms[q]), fontweight='bold', fontsize=11)
                plt.grid(True)

                q += 1

            plt.suptitle('Regional Trend', fontweight='bold')
            plt.subplots_adjust(wspace=0.3, hspace=0.6, top=0.92)

            # add the color map legend
            plt.subplot(n_rows, n_cols, q+1)
            ax = plt.gca()
            ax.grid(zorder=0)
            l = plt.legend(handles=red_cmap_pats, loc=2, bbox_to_anchor=(0.0, 1.0))
            plt.axis('off')

            plt.savefig('%s_regional_trend_%d.png'%( \
                state.basename, page_no), dpi=state.dpi)

            if (state.interactive):
                plt.show()

            # restore matplot lib global state
            plt.rcParams['legend.frameon'] = legend_frame_on_orig

            # send data downstream
            return summary
        return execute
