import sys
import teca_py
import numpy as np

class teca_tc_trajectory_scalars:
    """
    Computes summary statistics, histograms on sorted, classified,
    TC trajectory output.
    """
    @staticmethod
    def New():
        return teca_tc_trajectory_scalars()

    def __init__(self):
        self.basename = 'tc_track'
        self.texture = ''
        self.dpi = 100
        self.interactive = False

        self.impl = teca_py.teca_programmable_algorithm.New()
        self.impl.set_number_of_input_connections(1)
        self.impl.set_number_of_output_ports(1)
        self.impl.set_execute_callback(self.get_execute_callback(self))

    def __str__(self):
        return 'basename=%s, dpi=%d, interactive=%s, rel_axes=%s'%( \
            self.basename, self.dpi, str(self.interactive), str(self.rel_axes))

    def set_basename(self, basename):
        """
        All output files are prefixed by the basename. default 'tc_track'
        """
        self.basename = basename

    def set_texture(self, texture):
        """
        All output files are prefixed by the textrure. default 'tc_track'
        """
        self.texture = texture

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
    def get_execute_callback(state):
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
            #sys.stderr.write('teca_tc_trajectory_scalars::execute\n')

            import matplotlib.pyplot as plt
            import matplotlib.patches as plt_mp
            import matplotlib.image as plt_img

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
                return teca_py.teca_table.New()

            time_units = in_table.get_time_units()

            time = in_table.get_column('time').as_array()
            step = in_table.get_column('step').as_array()
            track = in_table.get_column('track_id').as_array()

            lon = in_table.get_column('lon').as_array()
            lat = in_table.get_column('lat').as_array()

            year = in_table.get_column('year').as_array()
            month = in_table.get_column('month').as_array()
            day = in_table.get_column('day').as_array()
            hour = in_table.get_column('hour').as_array()
            minute = in_table.get_column('minute').as_array()

            wind = in_table.get_column('surface_wind').as_array()
            vort = in_table.get_column('850mb_vorticity').as_array()
            psl = in_table.get_column('sea_level_pressure').as_array()
            temp = in_table.get_column('core_temp').as_array()
            have_temp = in_table.get_column('have_core_temp').as_array()
            thick = in_table.get_column('thickness').as_array()
            have_thick = in_table.get_column('have_thickness').as_array()
            speed = in_table.get_column('storm_speed').as_array()

            utrack = sorted(set(track))
            nutracks = len(utrack)

            # load background image
            tex = plt_img.imread(state.texture) if state.texture else None

            for i in utrack:
                #sys.stderr.write('processing track %d\n'%(i))
                sys.stderr.write('.')

                fig = plt.figure()
                fig.set_size_inches(8,11)

                ii = np.where(track == i)[0]

                # construct the title
                q = ii[0]
                r = ii[-1]

                t0 = time[q]
                t1 = time[r]

                s0 = step[q]
                s1 = step[r]

                Y0 = year[q]
                Y1 = year[r]

                M0 = month[q]
                M1 = month[r]

                D0 = day[q]
                D1 = day[r]

                h0 = hour[q]
                h1 = hour[r]

                m0 = minute[q]
                m1 = minute[r]

                tt = time[ii] - t0

                plt.suptitle( \
                    'Track %d, steps %d - %d\n%d/%d/%d %d:%d:00 - %d/%d/%d %d:%d:00'%(\
                    i, s0, s1, Y0, M0, D0, h0, m0, Y1, M1, D1, h1, m1), \
                    fontweight='bold')

                # get the scalar values for this storm
                lon_i = lon[ii]
                lat_i = lat[ii]
                wind_i = wind[ii]
                psl_i = psl[ii]
                vort_i = vort[ii]
                thick_i = thick[ii]
                temp_i = temp[ii]
                speed_i = speed[ii]/24.0

                # we'll add marks where thickness and core temp are
                # satisifed in each plot
                thick_ids = np.where(have_thick[ii] == 1)
                temp_ids = np.where(have_temp[ii] == 1)

                # plot the scalars
                plt.subplot(421)
                # prepare the texture
                if tex is not None:
                    ext = [np.min(lon_i), np.max(lon_i), np.min(lat_i), np.max(lat_i)]
                    i0 = int(tex.shape[1]/360.0*ext[0])
                    i1 = int(tex.shape[1]/360.0*ext[1])
                    j0 = int(-((ext[3] + 90.0)/180.0 - 1.0)*tex.shape[0])
                    j1 = int(-((ext[2] + 90.0)/180.0 - 1.0)*tex.shape[0])
                    plt.imshow(tex[j0:j1, i0:i1], extent=ext, aspect='auto')
                    plt.plot(lon_i, lat_i, '-', linewidth=2, color='#ffff00')
                    plt.plot(lon_i[0], lat_i[0], 'x', markersize=7, markeredgewidth=2, color='#ffff00')
                else:
                    plt.plot(lon_i, lat_i, 'k-', linewidth=2)
                    plt.plot(lon_i[thick_ids], lat_i[thick_ids], 'rx', markeredgewidth=2, markersize=7, markerfacecolor='none',linewidth=2)
                    plt.plot(lon_i[temp_ids], lat_i[temp_ids], 'b+', markeredgewidth=2, markersize=7,  markerfacecolor='none', linewidth=2)
                    plt.plot(lon_i[0], lat_i[0], 'ko', markeredgewidth=2, markerfacecolor='none')
                plt.grid(True)
                plt.xlabel('deg lon')
                plt.ylabel('deg lat')
                plt.title('Track', fontweight='bold')

                plt.subplot(422)
                plt.plot(tt, psl_i, 'k-', linewidth=2)
                plt.plot(tt[thick_ids], psl_i[thick_ids], 'rx', markeredgewidth=2, markersize=7, markerfacecolor='none',linewidth=2)
                plt.plot(tt[temp_ids], psl_i[temp_ids], 'b+', markeredgewidth=2, markersize=7,  markerfacecolor='none', linewidth=2)
                plt.grid(True)
                plt.xlabel('time (days)')
                plt.ylabel('millibars')
                plt.title('Sea Level Pressure', fontweight='bold')

                plt.subplot(423)
                plt.plot(tt, speed_i, 'k-', linewidth=2)
                plt.plot(tt[thick_ids], speed_i[thick_ids], 'rx', markeredgewidth=2, markersize=7, markerfacecolor='none', linewidth=2)
                plt.plot(tt[temp_ids], speed_i[temp_ids], 'b+', markeredgewidth=2, markersize=7,  markerfacecolor='none', linewidth=2)
                plt.grid(True)
                plt.xlabel('time (days)')
                plt.ylabel('km d^-1')
                plt.title('Speed', fontweight='bold')

                plt.subplot(424)
                plt.plot(tt, wind_i, 'k-', linewidth=2)
                plt.plot(tt[thick_ids], wind_i[thick_ids], 'rx', markeredgewidth=2, markersize=7, markerfacecolor='none', linewidth=2)
                plt.plot(tt[temp_ids], wind_i[temp_ids], 'b+', markeredgewidth=2, markersize=7,  markerfacecolor='none', linewidth=2)
                plt.grid(True)
                plt.xlabel('time (days)')
                plt.ylabel('ms^-1')
                plt.title('Wind Speed', fontweight='bold')

                plt.subplot(425)
                plt.plot(tt, thick_i, 'k-', linewidth=2)
                plt.plot(tt[thick_ids], thick_i[thick_ids], 'rx', markeredgewidth=2, markersize=7, markerfacecolor='none', linewidth=2)
                plt.plot(tt[temp_ids], thick_i[temp_ids], 'b+', markeredgewidth=2, markersize=7,  markerfacecolor='none', linewidth=2)
                plt.grid(True)
                plt.xlabel('time (days)')
                plt.ylabel('meters')
                plt.title('Thickness', fontweight='bold')

                plt.subplot(426)
                plt.plot(tt, vort_i, 'k-', linewidth=2)
                plt.plot(tt[thick_ids], vort_i[thick_ids], 'rx', markeredgewidth=2, markersize=7, markerfacecolor='none', linewidth=2)
                plt.plot(tt[temp_ids], vort_i[temp_ids], 'b+', markeredgewidth=2, markersize=7,  markerfacecolor='none', linewidth=2)
                plt.grid(True)
                plt.xlabel('time (days)')
                plt.ylabel('s^-1')
                plt.title('Vorticity', fontweight='bold')

                plt.subplot(427)
                plt.plot(tt, temp_i, 'k-', linewidth=2)
                plt.plot(tt[thick_ids], temp_i[thick_ids], 'rx', markeredgewidth=2, markersize=7, markerfacecolor='none', linewidth=2)
                plt.plot(tt[temp_ids], temp_i[temp_ids], 'b+', markeredgewidth=2, markersize=7,  markerfacecolor='none', linewidth=2)
                plt.grid(True)
                plt.xlabel('time (days)')
                plt.ylabel('deg K')
                plt.title('Core Temperature', fontweight='bold')

                #plt.subplot(525)
                #plt.plot(tt, have_thick[ii], 'k-', linewidth=2)
                #plt.grid(True)
                #plt.xlabel('time (days)')
                #plt.ylabel('yes/no')
                #plt.title('have thickness')

                #plt.subplot(529)
                #plt.plot(tt, have_temp[ii], 'k-', linewidth=2)
                #plt.grid(True)
                #plt.xlabel('time (days)')
                #plt.ylabel('yes/no')
                #plt.title('have core temperature')

                plt.subplots_adjust(wspace=0.3, hspace=0.45, top=0.9)

                plt.savefig('%s_%06d.png'%(state.basename, i), dpi=state.dpi)

            if (state.interactive):
                plt.show()

            out_table = teca_py.teca_table.New()
            out_table.shallow_copy(in_table)
            return out_table
        return execute

