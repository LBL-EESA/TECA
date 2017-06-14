import sys
import teca_py
import numpy as np

class teca_tc_wind_radii_stats:
    """
    Computes statistics using track wind radii
    """
    @staticmethod
    def New():
        return teca_tc_wind_radii_stats()

    def __init__(self):
        self.basename = 'stats'
        self.dpi = 100
        self.interactive = False
        self.wind_column = 'surface_wind'
        self.output_prefix = ''

        self.impl = teca_py.teca_programmable_algorithm.New()
        self.impl.set_number_of_input_connections(1)
        self.impl.set_number_of_output_ports(1)
        self.impl.set_execute_callback(self.get_tc_wind_radii_stats_execute(self))

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

    def set_wind_column(self, wind_column):
        """
        set the name of the column to obtain wind speed from
        """
        self.wind_column = wind_column

    def set_output_prefix(self, output_prefix):
        """
        set the path to prepend to output files
        """
        self.output_prefix = output_prefix

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
    def get_tc_wind_radii_stats_execute(state):
        """
        return a teca_algorithm::execute function. a closure
        is used to gain state.
        """
        def execute(port, data_in, req):
            """
            expects a table with track data containing wind radii computed
            along each point of the track. produces statistical plots showing
            the global distribution of wind radii.
            """
            track_table = teca_py.as_teca_table(data_in[0])

            # plot stats
            import matplotlib.pyplot as plt
            import matplotlib.patches as plt_mp
            from matplotlib.colors import LogNorm

            red_cmap = ['#ffd2a3','#ffa749','#ff7c04', \
                '#ea4f00','#c92500','#a80300']

            km_per_deg_lat = 111
            km_s_per_m_hr = 3.6

            fig = plt.figure(figsize=(9.25,6.75),dpi=state.dpi)

            # scatter
            plt.subplot('331')
            plt.hold('True')

            if not track_table.has_column(state.wind_column):
                sys.stderr.write('ERROR: track table missing %s\n'%(state.wind_column))
                sys.exit(-1)


            year = track_table.get_column('year').as_array()
            month = track_table.get_column('month').as_array()
            day = track_table.get_column('day').as_array()

            ws = km_s_per_m_hr*track_table.get_column(state.wind_column).as_array()

            wr = []
            nwr = 0
            while track_table.has_column('wind_radius_%d'%(nwr)):
                wr.append(km_per_deg_lat*track_table.get_column('wind_radius_%d'%(nwr)).as_array())
                nwr += 1

            i = 0
            while i < nwr:
                wc = teca_py.teca_tc_saffir_simpson.get_upper_bound_kmph(i-1)
                wri = wr[i]
                ii = np.where(wri > 0.0)
                plt.scatter(wri[ii], ws[ii], c=red_cmap[i], alpha=0.25, marker='.', zorder=3+i)
                i += 1

            plt.ylabel('Wind speed (km/hr)', fontweight='normal', fontsize=10)
            plt.title('R0 - R5 vs Wind speed', fontweight='bold', fontsize=11)
            plt.grid(True)
            ax = plt.gca()
            ax.set_xlim([0.0, 6.0*km_per_deg_lat])

            # all
            plt.subplot('332')
            plt.hold(True)
            i = 0
            while i < nwr:
                wc = teca_py.teca_tc_saffir_simpson.get_upper_bound_kmph(i-1)
                wri = wr[i]
                n,bins,pats = plt.hist(wri[np.where(wri > 0.0)], 32, range=[0,6.0*km_per_deg_lat], \
                    facecolor=red_cmap[i], alpha=0.95, edgecolor='black', \
                    linewidth=2, zorder=3+i)
                i += 1
            plt.ylabel('Number', fontweight='normal', fontsize=10)
            plt.title('All R0 - R5', fontweight='bold', fontsize=11)
            plt.grid(True)
            ax = plt.gca()
            ax.set_xlim([0.0, 6.0*km_per_deg_lat])

            # r0 - r5
            i = 0
            while i < nwr:
                plt.subplot(333+i)
                wc = teca_py.teca_tc_saffir_simpson.get_upper_bound_kmph(i-1)
                wri = wr[i]
                wrii=wri[np.where(wri > 0.0)]
                n,bins,pats = plt.hist(wrii, 32, \
                    facecolor=red_cmap[i], alpha=1.00, edgecolor='black', \
                    linewidth=2, zorder=3)
                if ((i % 3) == 1):
                    plt.ylabel('Number', fontweight='normal', fontsize=10)
                if (i >= 3):
                    plt.xlabel('Radius (km)', fontweight='normal', fontsize=10)
                plt.title('R%d (%0.1f km/hr)'%(i,wc), fontweight='bold', fontsize=11)
                plt.grid(True)
                ax = plt.gca()
                ax.set_xlim([np.min(wrii), np.max(wrii)])
                i += 1

            # legend
            plt.subplot('339')
            red_cmap_pats = []
            q = 0
            while q < nwr:
                red_cmap_pats.append( \
                    plt_mp.Patch(color=red_cmap[q], label='R%d'%(q)))
                q += 1
            l = plt.legend(handles=red_cmap_pats, loc=2, bbox_to_anchor=(-0.1, 1.0), fancybox=True)
            plt.axis('off')


            plt.suptitle('Wind Radii %s/%d/%d - %s/%d/%d'%(month[0],day[0],year[0], \
                month[-1],day[-1],year[-1]), fontweight='bold', fontsize=12)
            plt.subplots_adjust(hspace=0.35, wspace=0.35, top=0.90)

            plt.savefig(state.output_prefix + 'wind_radii_stats.png')

            fig = plt.figure(figsize=(7.5,4.0),dpi=100)
            # peak radius
            pr = km_per_deg_lat*track_table.get_column('peak_radius').as_array()
            # peak radius is only valid if one of the other wind radii
            # exist
            kk = wr[0] > 1.0e-6
            q = 1
            while q < nwr:
                kk = np.logical_or(kk, wr[q] > 1.0e-6)
                q += 1
            pr = pr[kk]

            plt.subplot(121)
            n,bins,pats = plt.hist(pr[np.where(pr > 0.0)], 24, \
                facecolor='steelblue', alpha=0.95, edgecolor='black', \
                linewidth=2, zorder=3)
            plt.ylabel('Number', fontweight='normal', fontsize=10)
            plt.xlabel('Radius (km)', fontweight='normal', fontsize=10)
            plt.title('RP (radius at peak wind)', fontweight='bold', fontsize=11)
            plt.grid(True)
            ax = plt.gca()
            ax.set_xlim([0.0, np.max(pr)])

            # scatter
            plt.subplot('122')
            plt.hold('True')
            ii = np.where(pr > 0.0)
            cnts,xe,ye,im = plt.hist2d(pr[ii], ws[ii], bins=24, norm=LogNorm(), zorder=2)
            plt.ylabel('Wind speed (km/hr)', fontweight='normal', fontsize=10)
            plt.xlabel('Radius (km)', fontweight='normal', fontsize=10)
            plt.title('RP vs Wind speed', fontweight='bold', fontsize=11)
            plt.grid(True)
            ax = plt.gca()
            ax.set_xlim([0.0, np.max(pr)])

            fig.subplots_adjust(right=0.85)
            cbar_ax = fig.add_axes([0.88, 0.35, 0.05, 0.5])
            fig.colorbar(im, cax=cbar_ax)

            plt.suptitle('Wind Radii %s/%d/%d - %s/%d/%d'%(month[0],day[0],year[0], \
                month[-1],day[-1],year[-1]), fontweight='bold', fontsize=12)
            plt.subplots_adjust(hspace=0.3, wspace=0.3,  top=0.85)

            plt.savefig(state.output_prefix + 'peak_radius_stats.png')

            if state.interactive:
                plt.show()

            # send data downstream
            return track_table
        return execute
