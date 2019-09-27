import sys
import teca_py
import numpy as np

class teca_tc_trajectory_scalars(teca_py.teca_python_algorithm):
    """
    Computes summary statistics, histograms on sorted, classified,
    TC trajectory output.
    """
    def __init__(self):
        self.basename = 'tc_track'
        self.tex_file = ''
        self.tex = None
        self.dpi = 100
        self.interactive = False
        self.axes_equal = True
        self.plot_peak_radius = False

    def __str__(self):
        return 'basename=%s, dpi=%d, interactive=%s, rel_axes=%s'%( \
            self.basename, self.dpi, str(self.interactive), str(self.rel_axes))

    def set_basename(self, basename):
        """
        All output files are prefixed by the basename. default 'tc_track'
        """
        self.basename = basename

    def set_texture(self, file_name):
        """
        All output files are prefixed by the textrure. default 'tc_track'
        """
        self.tex_file = file_name

    def set_dpi(self, dpi):
        """
        set the DPI resolution for image output. default 100
        """
        self.dpi = dpi

    def set_interactive(self, interactive):
        """
        plots are rendered to an on screen window when enabled.
        when disabled plots are written directly to disk. default False
        """
        self.interactive = interactive

    def set_axes_equal(self, axes_equal):
        """
        controls the scaling of track plots. when off the aspect ratio
        is modified to best suit the window size. default True.
        """
        self.axes_equal = axes_equal

    def set_plot_peak_radius(self, plot_peak_radius):
        """
        when set peak wind radius is included in the plots. default False
        """
        self.plot_peak_radius = plot_peak_radius

    def get_execute_callback(self):
        """
        return a teca_algorithm::execute function. a closure
        is used to gain self.
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
            import matplotlib.gridspec as plt_gridspec

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

            # use this color map for Saphir-Simpson scale
            red_cmap = ['#ffd2a3','#ffa749','#ff7c04', \
                '#ea4f00','#c92500','#a80300']

            km_per_deg_lat = 111

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

            wind_rad = []
            i = 0
            while i < 5:
                col_name = 'wind_radius_%d'%(i)
                if in_table.has_column(col_name):
                    wind_rad.append(in_table.get_column(col_name).as_array())
                i += 1
            peak_rad = in_table.get_column('peak_radius').as_array() \
                if in_table.has_column('peak_radius') else None

            # get the list of unique track ids, this is our loop index
            utrack = sorted(set(track))
            nutracks = len(utrack)

            # load background image
            if (self.tex is None) and self.tex_file:
                self.tex = plt_img.imread(self.tex_file)

            for i in utrack:
                #sys.stderr.write('processing track %d\n'%(i))
                sys.stderr.write('.')

                fig = plt.figure()
                fig.set_size_inches(10,9.75)

                ii = np.where(track == i)[0]

                # get the scalar values for this storm
                lon_i = lon[ii]
                lat_i = lat[ii]
                wind_i = wind[ii]
                psl_i = psl[ii]
                vort_i = vort[ii]
                thick_i = thick[ii]
                temp_i = temp[ii]
                speed_i = speed[ii]/24.0

                wind_rad_i = []
                for col in wind_rad:
                    wind_rad_i.append(col[ii])
                peak_rad_i = peak_rad[ii] if peak_rad is not None else None

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

                cat = teca_py.teca_tc_saffir_simpson.classify_mps(float(np.max(wind_i)))

                plt.suptitle( \
                    'Track %d, cat %d, steps %d - %d\n%d/%d/%d %d:%d:00 - %d/%d/%d %d:%d:00'%(\
                    i, cat, s0, s1, Y0, M0, D0, h0, m0, Y1, M1, D1, h1, m1), \
                    fontweight='bold')

                # plot the scalars
                gs = plt_gridspec.GridSpec(5, 4)

                plt.subplot2grid((5,4),(0,0),colspan=2,rowspan=2)
                # prepare the texture
                if self.tex is not None:
                    ext = [np.min(lon_i), np.max(lon_i), np.min(lat_i), np.max(lat_i)]
                    if self.axes_equal:
                        w = ext[1]-ext[0]
                        h = ext[3]-ext[2]
                        if w > h:
                            c = (ext[2] + ext[3])/2.0
                            w2 = w/2.0
                            ext[2] = c - w2
                            ext[3] = c + w2
                        else:
                            c = (ext[0] + ext[1])/2.0
                            h2 = h/2.0
                            ext[0] = c - h2
                            ext[1] = c + h2
                    border_size = 0.15
                    wrimax = 0 if peak_rad_i is None else \
                        max(0 if not self.plot_peak_radius else \
                            np.max(peak_rad_i), np.max(wind_rad_i[0]))
                    dlon = max(wrimax, (ext[1] - ext[0])*border_size)
                    dlat = max(wrimax, (ext[3] - ext[2])*border_size)
                    ext[0] = max(ext[0] - dlon, 0.0)
                    ext[1] = min(ext[1] + dlon, 360.0)
                    ext[2] = max(ext[2] - dlat, -90.0)
                    ext[3] = min(ext[3] + dlat, 90.0)
                    i0 = int(self.tex.shape[1]/360.0*ext[0])
                    i1 = int(self.tex.shape[1]/360.0*ext[1])
                    j0 = int(-((ext[3] + 90.0)/180.0 - 1.0)*self.tex.shape[0])
                    j1 = int(-((ext[2] + 90.0)/180.0 - 1.0)*self.tex.shape[0])
                    plt.imshow(self.tex[j0:j1, i0:i1], extent=ext, aspect='auto')

                edge_color = '#ffff00' if self.tex is not None else 'b'

                # plot the storm size
                if peak_rad_i is None:
                    plt.plot(lon_i, lat_i, '.', linewidth=2, color=edge_color)
                else:
                    # compute track unit normals
                    npts = len(ii)
                    norm_x = np.zeros(npts)
                    norm_y = np.zeros(npts)
                    npts -= 1
                    q = 1
                    while q < npts:
                        norm_x[q] = lat_i[q+1] - lat_i[q-1]
                        norm_y[q] = -(lon_i[q+1] - lon_i[q-1])
                        nmag = np.sqrt(norm_x[q]**2 + norm_y[q]**2)
                        norm_x[q] = norm_x[q]/nmag
                        norm_y[q] = norm_y[q]/nmag
                        q += 1
                    # normal at first and last point on the track
                    norm_x[0] = lat_i[1] - lat_i[0]
                    norm_y[0] = -(lon_i[1] - lon_i[0])
                    norm_x[0] = norm_x[0]/nmag
                    norm_y[0] = norm_y[0]/nmag
                    norm_x[npts] = lat_i[npts] - lat_i[npts-1]
                    norm_y[npts] = -(lon_i[npts] - lon_i[npts-1])
                    norm_x[npts] = norm_x[npts]/nmag
                    norm_y[npts] = norm_y[npts]/nmag
                    # for each wind radius, render a polygon of width 2*wind
                    # centered on the track. have to break it into continuous
                    # segments
                    nwri = len(wind_rad_i)
                    q = nwri - 1
                    while q >= 0:
                        self.plot_wind_rad(lon_i, lat_i, norm_x, norm_y, \
                            wind_rad_i[q], '-', edge_color if q==0 else red_cmap[q], \
                            2 if q==0 else 1, red_cmap[q], 0.98, q+4)
                        q -= 1
                    # plot the peak radius
                    if (self.plot_peak_radius):
                        # peak radius is only valid if one of the other wind radii
                        # exist, zero out other values
                        kk = wind_rad_i[0] > 1.0e-6
                        q = 1
                        while q < nwri:
                            kk = np.logical_or(kk, wind_rad_i[q] > 1.0e-6)
                            q += 1
                        peak_rad_i[np.logical_not(kk)] = 0.0
                        self.plot_wind_rad(lon_i, lat_i, norm_x, norm_y, \
                            peak_rad_i, '--', (0,0,0,0.25), 1, 'none', 1.00, nwri+4)
                    # mark track
                    marks = wind_rad_i[0] <= 1.0e-6
                    q = 1
                    while q < nwri:
                        marks = np.logical_and(marks, np.logical_not(wind_rad_i[q] > 1.0e-6))
                        q += 1
                    kk = np.where(marks)[0]

                    plt.plot(lon_i[kk], lat_i[kk], '.', linewidth=2, \
                        color=edge_color,zorder=10)

                    plt.plot(lon_i[kk], lat_i[kk], '.', linewidth=1, \
                        color='k', zorder=10, markersize=1)

                    marks = wind_rad_i[0] > 1.0e-6
                    q = 1
                    while q < nwri:
                        marks = np.logical_or(marks, wind_rad_i[q] > 1.0e-6)
                        q += 1
                    kk = np.where(marks)[0]

                    plt.plot(lon_i[kk], lat_i[kk], '.', linewidth=1, \
                        color='k', zorder=10, markersize=2, alpha=0.1)

                # mark track start and end
                plt.plot(lon_i[0], lat_i[0], 'o', markersize=6, markeredgewidth=2, \
                    color=edge_color, markerfacecolor='g',zorder=10)

                plt.plot(lon_i[-1], lat_i[-1], '^', markersize=6, markeredgewidth=2, \
                    color=edge_color, markerfacecolor='r',zorder=10)

                plt.grid(True)
                plt.xlabel('deg lon')
                plt.ylabel('deg lat')
                plt.title('Track', fontweight='bold')

                plt.subplot2grid((5,4),(0,2),colspan=2)
                plt.plot(tt, psl_i, 'b-', linewidth=2)
                plt.grid(True)
                plt.xlabel('time (days)')
                plt.ylabel('millibars')
                plt.title('Sea Level Pressure', fontweight='bold')
                plt.xlim([0, tt[-1]])

                plt.subplot2grid((5,4),(1,2),colspan=2)
                plt.plot(tt, wind_i, 'b-', linewidth=2)
                plt.grid(True)
                plt.xlabel('time (days)')
                plt.ylabel('ms^-1')
                plt.title('Surface Wind', fontweight='bold')
                plt.xlim([0, tt[-1]])

                plt.subplot2grid((5,4),(2,0),colspan=2)
                plt.plot(tt, speed_i, 'b-', linewidth=2)
                plt.grid(True)
                plt.xlabel('time (days)')
                plt.ylabel('km d^-1')
                plt.title('Propagation Speed', fontweight='bold')
                plt.xlim([0, tt[-1]])

                plt.subplot2grid((5,4),(2,2),colspan=2)
                plt.plot(tt, vort_i, 'b-', linewidth=2)
                plt.grid(True)
                plt.xlabel('time (days)')
                plt.ylabel('s^-1')
                plt.title('Vorticity', fontweight='bold')
                plt.xlim([0, tt[-1]])

                plt.subplot2grid((5,4),(3,0),colspan=2)
                plt.plot(tt, thick_i, 'b-', linewidth=2)
                plt.grid(True)
                plt.xlabel('time (days)')
                plt.ylabel('meters')
                plt.title('Thickness', fontweight='bold')
                plt.xlim([0, tt[-1]])

                plt.subplot2grid((5,4),(3,2),colspan=2)
                plt.plot(tt, temp_i, 'b-', linewidth=2)
                plt.grid(True)
                plt.xlabel('time (days)')
                plt.ylabel('deg K')
                plt.title('Core Temperature', fontweight='bold')
                plt.xlim([0, tt[-1]])

                if peak_rad_i is not None:
                    plt.subplot2grid((5,4),(4,0),colspan=2)
                    q = len(wind_rad_i) - 1
                    while q >= 0:
                        wr_i_q = km_per_deg_lat*wind_rad_i[q]
                        plt.fill_between(tt, 0, wr_i_q, color=red_cmap[q], alpha=0.9, zorder=q+3)
                        plt.plot(tt, wr_i_q, '-', linewidth=2, color=red_cmap[q], zorder=q+3)
                        q -= 1
                    if (self.plot_peak_radius):
                        plt.plot(tt, km_per_deg_lat*peak_rad_i, 'k--', linewidth=1, zorder=10)
                    plt.plot(tt, np.zeros(len(tt)), 'w-', linewidth=2, zorder=10)
                    plt.grid(True)
                    plt.xlabel('time (days)')
                    plt.ylabel('radius (km)')
                    plt.title('Storm Size', fontweight='bold')
                    plt.xlim([0, tt[-1]])
                    plt.ylim(ymin=0)

                    plt.subplot2grid((5,4),(4,2))
                    red_cmap_pats = []
                    q = 0
                    while q < 6:
                        red_cmap_pats.append( \
                            plt_mp.Patch(color=red_cmap[q], label='R%d'%(q)))
                        q += 1
                    if (self.plot_peak_radius):
                        red_cmap_pats.append(plt_mp.Patch(color='k', label='RP'))
                    l = plt.legend(handles=red_cmap_pats, loc=2, \
                            bbox_to_anchor=(-0.1, 1.0), borderaxespad=0.0, \
                            frameon=True, ncol=2)
                    plt.axis('off')

                plt.subplots_adjust(left=0.065, right=0.98, \
                    bottom=0.05, top=0.9, wspace=0.6, hspace=0.7)

                plt.savefig('%s_%06d.png'%(self.basename, i), dpi=self.dpi)
                if (not self.interactive):
                    plt.close(fig)

            if (self.interactive):
                plt.show()

            out_table = teca_py.teca_table.New()
            out_table.shallow_copy(in_table)
            return out_table
        return execute

    @staticmethod
    def render_poly(x, y, norm_x, norm_y, rad, edge_style, \
        edge_color, edge_width, face_color, face_alpha, z):
        """draw a polygon centered at x,y with a normal width
           given by rad in the current axes"""
        import matplotlib.path as plt_path
        import matplotlib.patches as plt_patches
        import matplotlib.pyplot as plt
        wr_x = rad*norm_x
        wr_y = rad*norm_y
        nids = len(x)
        q = 0
        while q < nids-1:
            # build node types for quad
            poly_node_types = np.ones(5, int)*plt_path.Path.LINETO
            poly_node_types[0] = plt_path.Path.MOVETO
            poly_node_types[4] = plt_path.Path.CLOSEPOLY
            # build nodes for quad
            q0 = q
            q1 = q+2
            xx = x[q0:q1]
            yy = y[q0:q1]
            wwx = wr_x[q0:q1]
            wwy = wr_y[q0:q1]
            poly_nodes = np.zeros((5, 2))
            poly_nodes[0:2,0] = xx + wwx
            poly_nodes[0:2,1] = yy + wwy
            poly_nodes[2:4,0] = (xx - wwx)[::-1]
            poly_nodes[2:4,1] = (yy - wwy)[::-1]
            poly_nodes[4,0] = poly_nodes[0,0]
            poly_nodes[4,1] = poly_nodes[0,1]
            # build types for for edge
            edge_node_types = np.array([plt_path.Path.MOVETO, plt_path.Path.LINETO, \
                plt_path.Path.LINETO if q == nids-2 else plt_path.Path.MOVETO, \
                plt_path.Path.LINETO, plt_path.Path.LINETO if q == 0 else \
                plt_path.Path.MOVETO])
            # check for bow-tie configuration which occurs with sharp
            # curvature. NOTE: this tests for self intersection,
            # another case that we should handle is intersection
            # with preceding geometry.
            # the order of nodes relative to my diagram is p0,p2,p3,p1
            p0x = poly_nodes[0,0]
            p0y = poly_nodes[0,1]
            p2x = poly_nodes[1,0]
            p2y = poly_nodes[1,1]
            p3x = poly_nodes[2,0]
            p3y = poly_nodes[2,1]
            p1x = poly_nodes[3,0]
            p1y = poly_nodes[3,1]
            # a = p0 - p2
            # b = p1 - p0
            # c = p3 - p2
            ax = p0x - p2x
            ay = p0y - p2y
            bx = p1x - p0x
            by = p1y - p0y
            cx = p3x - p2x
            cy = p3y - p2y
            # D = -cx*by + cy*bx
            # D1 = -ax*by + ay*bx
            # D2 = cx*ay - cy*ax
            D = -cx*by + cy*bx
            #D1 = -ax*by + ay*bx
            D2 = cx*ay - cy*ax
            # t0 = D2/D
            # t1 = D1/D
            if (abs(D) > 1e-6):
                t0 = D2/D
                #t1 = D1/D
                # pi = p0 + (p1 - p0)t0
                pix = p0x + bx*t0
                piy = p0y + by*t0
                # pi = p2 + (p3 - p2)t1
                #pi1x = p2x + cx*t1
                #pi1y = p2y + cy*t1
                if ((t0 > 0.0) and (t0 <= 0.5)):
                    # track is turning right
                    # convert path to triangle, with p0 removed and p2 replaced with pi
                    poly_nodes[1,:] = [pix, piy]
                    poly_node_types[1] = plt_path.Path.MOVETO

                    edge_node_types[1] = plt_path.Path.MOVETO
                    #sys.stderr.write('%d sharp right! t0=%f %f,%f\n'%(q,t0,pix,piy))
                    #plt.plot([pix],[piy],'gx')
                if ((t0 > 0.5) and (t0 < 1.0)):
                    # track is turning left
                    # convert path to triangle, remove p1, replace p3 with pi
                    poly_nodes[2,:] = [pix, piy]
                    poly_nodes[3,:] = poly_nodes[0,:]
                    poly_node_types[3] = plt_path.Path.CLOSEPOLY

                    edge_node_types[3] = edge_node_types[4]
                    edge_node_types[4] = plt_path.Path.MOVETO
                    #sys.stderr.write('%d sharp left! t0=%f %f,%f\n'%(q,t0,pix,piy))
                    #plt.plot([pix],[piy],'r+')
            # patch is added to the current axes, constructed from a path
            # which is constructed from nodes and node types
            plt.gca().add_patch(plt_patches.PathPatch( \
                plt_path.Path(poly_nodes, poly_node_types), \
                facecolor=face_color, edgecolor=face_color, \
                linewidth=1, alpha=face_alpha, zorder=z))

            plt.gca().add_patch(plt_patches.PathPatch( \
                plt_path.Path(poly_nodes, edge_node_types), \
                facecolor='none', edgecolor=edge_color, \
                linestyle=edge_style, linewidth=edge_width, \
                alpha=1.0, zorder=z))

            q += 1
        return

    @staticmethod
    def render_poly_simple(x, y, norm_x, norm_y, rad, edge_color, \
        edge_width, face_color, face_alpha, z):
        """draw a polygon centered at x,y with a normal width
           given by rad in the current axes"""
        import matplotlib.path as plt_path
        import matplotlib.patches as plt_patches
        import matplotlib.pyplot as plt
        wr_x = rad*norm_x
        wr_y = rad*norm_y
        nids = len(x)
        q = 0
        while q < nids-1:
            # build node types
            node_types = np.ones(5, int)*plt_path.Path.LINETO
            node_types[0] = plt_path.Path.MOVETO
            node_types[4] = plt_path.Path.CLOSEPOLY
            # build nodes
            q0 = q
            q1 = q+2
            xx = x[q0:q1]
            yy = y[q0:q1]
            wwx = wr_x[q0:q1]
            wwy = wr_y[q0:q1]
            nodes = np.zeros((5, 2))
            nodes[0:2,0] = xx - wwx
            nodes[0:2,1] = yy - wwy
            nodes[2:4,0] = (xx + wwx)[::-1]
            nodes[2:4,1] = (yy + wwy)[::-1]
            nodes[4,0] = nodes[0,0]
            nodes[4,1] = nodes[0,1]
            # patch is added to the current axes, constructed from a path
            # which is constructed from nodes and node types
            plt.gca().add_patch(plt_patches.PathPatch( \
                plt_path.Path(nodes, node_types), \
                facecolor=face_color, \
                linewidth=1, alpha=0.5, zorder=z))
            q += 1
        return

    @staticmethod
    def render_poly_convex(x, y, norm_x, norm_y, rad, edge_style, \
        edge_color, edge_width, face_color, face_alpha, z):
        """draw a polygon centered at x,y with a normal width
           given by rad in the current axes"""
        import matplotlib.path as plt_path
        import matplotlib.patches as plt_patches
        import matplotlib.pyplot as plt
        nids = len(x)
        if nids:
            nids2 = 2*nids
            nnodes = nids2+1
            # build node types
            node_types = np.ones(nnodes, int)*plt_path.Path.LINETO
            node_types[0] = plt_path.Path.MOVETO
            node_types[nids2] = plt_path.Path.CLOSEPOLY
            # build nodes
            nodes = np.zeros((nnodes, 2))
            wr_x = rad*norm_x
            wr_y = rad*norm_y
            nodes[0:nids,0] = x - wr_x
            nodes[0:nids,1] = y - wr_y
            nodes[nids:nids2,0] = x[::-1] + wr_x[::-1]
            nodes[nids:nids2,1] = y[::-1] + wr_y[::-1]
            nodes[nids2,0] = nodes[0,0]
            nodes[nids2,1] = nodes[0,1]
            # patch is added to the current axes, constructed from a path
            # which is constructed from nodes and node types
            plt.gca().add_patch(plt_patches.PathPatch( \
                plt_path.Path(nodes, node_types), \
                facecolor=face_color, edgecolor=edge_color, \
                linestyle=edge_style, linewidth=edge_width, \
                alpha=face_alpha, zorder=z))
        return

    def plot_wind_rad(self, x, y, norm_x, norm_y, wind_rad, \
        edge_style, edge_color, edge_width, face_color, face_alpha, z):
        """slpit the track into continuous segements where wind rad is defined
           and render them on the current axes as close polygons/polylines"""
        nqq = len(x)-1
        p0 = -1
        qq = 0
        while qq <= nqq:
            if (p0 < 0) and (wind_rad[qq] > 1e-6):
                # found start of poly
                p0 = qq
            elif (p0 >= 0) and ((wind_rad[qq] < 1e-6) or (qq == nqq)):
                # found end of poly, render it
                p1 = qq+1 if qq == nqq else qq
                if p1 - p0 > 1:
                    self.render_poly(x[p0:p1], y[p0:p1], \
                        norm_x[p0:p1], norm_y[p0:p1], wind_rad[p0:p1], \
                        edge_style, edge_color, edge_width, face_color, \
                        face_alpha, z)

                p0 = -1
            qq += 1
