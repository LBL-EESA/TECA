import numpy as np
import teca_py
import sys

class teca_ar_shields_kiehl(object):
    """
    ; ==========================================================================
    ; reads in vars to identify ARs along coastal areas
    ; TMQ, U850, V850, LANDFRAC
    ; computes zonal mean TMQ for each time slice
    ; and finds max longitudinal value for ZN Zanom thresholds
    ; ZN test from Zhu and Newell 1998 (Newman et al 2012)
    ; Q >= Qmean + .3(Qmax - Qmean)
    ; where mean is zonal mean and max is max value across all longitudes
    ; find broad U.S. west coast points ( landfrac < 0 but > 1)
    ; date and latitudes. bin by 1 deg latitude bins to test to
    ; find how AR landfall location is changing over time
    ; user-defined thresholds of all vars to search and find dates
    ; output all dates/lat bands and vars identified into netcdf file
    ; use/write another script to read dates and perform plots and stats
    ; note: u850,v850,tmq 6hrly
    ; fortran wrapper for searches
    ;
    ; NOTE: this is different from atmos_river_find_thresh suite in that
    ; above script finds only AR's for specified small regional box so that
    ; composites for regions work.
    ; this script allows the whole US coast (socal, ca, and pwn regions) combined
    ; and records latitude of landfall.
    ; in other scripts, latitudes can then binned to identify how each bin
    ; changes with time
    ; ==========================================================================
    """

    class region:
        def __init__(self):
            self.reg_lab = ""
            self.latmin = 0.
            self.latmax = 0.
            self.lonmin = 0.
            self.lonmax = 0.
            self.wdir_gt_thresh = 0.
            self.wdir_lt_thresh = 0.
            self.wsp_thresh = 0.

    @staticmethod
    def New():
        return teca_ar_shields_kiehl()

    def ar_find_zn_lat(tmq,ztmq,ztmqmx,wsp,wdir,landfrac,date, time,lat,lon,nlat,nlon,ntimes,imax, \
                       thresh_wsp, thresh_gt_wdir,thresh_lt_wdir, thresh_length,thresh_width,event_dates, \
                       event_times,event_lats,nevents):

        # ZN test from Zhu and Newell 1998 (Newman et al 2012)
        # Q >= Qmean + .3(Qmax - Qmean)
        # where mean is zonal mean and max is max value across all longitudes
        # ======neiman,ralph,wick (et al papers) thresholds for river definition
        #       2000km length 1000k widge, 2cm iwv
        #       for model: use grid points and tmq in kg/m2
        #       imax is max number of events per file
        #       latmax/min and lonmax/min are regional boundaries for search area
        #
        #       search all grid points in specified area for atmospheric rivers
        #       using above thresholds. high tmq values must also coincide with coastal
        #       points
        #
        #==== pass 1:  high tmq value along coastal points condition met
        #     pass 2:  wind speed and direction thresholds met
        #     pass 3:  shape thresholds met
        #
        # == for each timeslice flag if event has been found, if so, jump to next time
        #
        #===  return:  dates and times of events

        # == compute threshold

        nevents = 0
        for k in range(1, ntimes):
            flag = 0
            for j in range(1, nlat-thresh_length):
                for i in range(1, nlon-thresh_width):
                    length_flag = 0
                    width_flag = 0

                    thresh_tmq[j,k] = ztmq[j,k] + .3*(ztmqmx[j,k] - ztmq[j,k])

                    # pass 1
                    if landfrac[i,j] > 0 and landfrac[i,j] < 1 and tmq[i,j,k] >= thresh_tmq[j,k] :

                        # pass 2
                        if wsp[i,j,k] > thresh_wsp and wdir[i,j,k] > thresh_gt_wdir and wdir[i,j,k] < thresh_lt_wdir:

                            # pass 3
                            for m in range(1,thresh_length):
                                if tmq[i,j+m,k] > thresh_tmq[j+m,k] :
                                    length_flag = length_flag + 1

                            for m in range(1,thresh_width):
                                if tmq[i+m,j,k] > thresh_tmq[j,k] :
                                    width_flag = width_flag + 1

                            # if conditions met, assign date and lat

                            if length_flag >= thresh_length and width_flag >= thresh_width:
                               flag = 1
                               nevents = nevents + 1
                               event_dates.append(date[k])
                               event_times.append(time[k])
                               event_lats.append(lat[j])

                    if flag == 1:
                        break
                if flag == 1:
                    break

    def ar_sort(tmq,wsp,wdir,ux,vy,landfrac,lat,lon,time, nlat,nlon,ntimes,event_dates,event_times, \
                nevents,ar_tmq,ar_wsp,ar_dir,ar_ux,ar_vy):

        ar_tmp.fill(0)
        ar_wsp.fill(0)
        ar_dir.fill(0)
        ar_ux.fill(0)
        ar_vy.fill(0)

        # ====== read in ar data and dates/times associated with events and put
        #       into seperate arrays such that only event times are included
        #       in timeslices
        #
        #===  return:  dates and times of events

        for n in range(1,nevents):
            for k in range(1, ntimes):
                if event_times[n] == time[k]:
                    for j in range(1, nlat):
                        for i in range(1, nlon):
                           ar_tmq[i,j,n] = tmq[i,j,k]
                           ar_wsp[i,j,n] = wsp[i,j,k]
                           ar_dir[i,j,n] = wdir[i,j,k]
                           ar_ux[i,j,n] = ux[i,j,k]
                           ar_vy[i,j,n] = vy[i,j,k]

    def __init__(self):
        self.impl = teca_py.teca_programmable_algorithm.New()
        self.impl.set_number_of_input_connections(1)
        self.impl.set_number_of_output_ports(1)
        self.impl.set_report_callback(self.get_report_callback())
        self.impl.set_request_callback(self.get_request_callback())
        self.impl.set_execute_callback(self.get_execute_callback())


        # self.gridfile = "/project/projectdirs/m1949/model_input/cesm/inputdata/lnd/clm2/surfdata/surfdata_0.47x0.63_simyr1850_c100305.nc"

        self.windlab = "850" # ; "BOT"  # wind level label of filenames
        self.testlab = "tmq_Zanom_ZN"

        self.regions = {}

        self.regions["UK"] = teca_ar_shields_kiehl.region()

        self.regions["UK"].reg_lab = "UK"
        self.regions["UK"].latmin = 49.
        self.regions["UK"].latmax = 60.
        self.regions["UK"].lonmin = -15.
        self.regions["UK"].lonmax = 0.
        self.regions["UK"].wdir_gt_thresh = 180.
        self.regions["UK"].wdir_lt_thresh = 360.
        self.regions["UK"].wsp_thresh = 25.

        self.regions["Ib"] = teca_ar_shields_kiehl.region()
        self.regions["Ib"].reg_lab = "Ib"
        self.regions["Ib"].latmin = 35.
        self.regions["Ib"].latmax = 49.
        self.regions["Ib"].lonmin = -15.
        self.regions["Ib"].lonmax = 0.
        self.regions["Ib"].wdir_gt_thresh = 180.
        self.regions["Ib"].wdir_lt_thresh = 360.
        self.regions["Ib"].wsp_thresh = 15.

        self.regions["WEST_US"] = teca_ar_shields_kiehl.region()
        self.regions["WEST_US"].reg_lab = "WEST_US"
        self.regions["WEST_US"].latmin = 32.
        self.regions["WEST_US"].latmax = 52.5
        self.regions["WEST_US"].lonmin = -130.
        self.regions["WEST_US"].lonmax = -110.
        self.regions["WEST_US"].wdir_gt_thresh = 180.
        self.regions["WEST_US"].wdir_lt_thresh = 270.
        self.regions["WEST_US"].wsp_thresh = 10.

        self.length_thresh = 8  # grid pts;  1grid pt ~ 25km
        self.width_thresh = 2
        self.threshold_doc = "tmqanomZ> = ZN wsp>" + wsp_thresh + " " + wdir_lt_thresh + "<wdir>" + wdir_gt_thresh+" gpts_length="+length_thresh + " gpts_width="+width_thresh
        self.wl_lab = "8_2"

        self.iflip = 1  # flip longs to -180 + 180

        self.active_region = self.regions["WEST_US"]

        # ==== save/plot regions
        self.platmin = -90
        self.platmax = 90
        self.plonmin = -180.
        self.plonmax = 180.

        if self.windlab == "BOT":
            self.wind_lab = "Surface"
            self.ufield = "UBOT"
            self.vfield = "VBOT"
        elif self.windlab == "850":
            self.wind_lab = "850mb"
            self.ufield = "U850"
            self.vfield = "V850"
        else:
            raise "wind not coded yet"

    def set_input_connection(self, port, obj):
        self.impl.set_input_connection(obj)

    def get_output_port(self):
        return self.impl.get_output_port()

    def update(self):
        self.impl.update()

    def get_report_callback(self):
        """ Returns a proper TECA report callback function """
        def report_callback(port, md_in):
            """ Defines the variables that this class returns. """
            # add the names of the variables this method generates
            md_in[0].append('variables', 'ar_tmq')
            md_in[0].append('variables', 'ar_ux')
            md_in[0].append('variables', 'ar_vy')
            md_in[0].append('variables', 'ar_wsp')
            md_in[0].append('variables', 'ar_wdir')
            md_in[0].append('variables', 'ar_ladfrc')
            md_in[0].append('variables', 'ar_date')
            md_in[0].append('variables', 'ar_time')
            md_in[0].append('variables', 'lat_landfall')
            md_in[0].append('variables', 'lat_save')
            md_in[0].append('variables', 'lon_save')

            return md_in

        return report_callback

    def get_request_callback(self):
        """ Returns a proper TECA request callback function """
        def request_callback(port, md_in, req_in):
            """ Requests the variables needed to find ARs """
            # add the name of arrays that we need to find ARs
            req_in['arrays'] = ['TMQ', self.ufield, self.vfield, 'LANDFRAC', 'lat', 'lon', 'date', 'time']

            reqs_out = [req_in]
            return reqs_out
        return request_callback

    def get_execute_callback(self):
        """ Returns a proper TECA execute callback function """
        def execute_callback(port, data_in, req_in):

            # pass the incoming data through
            in_mesh = teca_py.as_teca_cartesian_mesh(data_in[0])
            out_mesh = teca_cartesian_mesh.New()
            out_mesh.shallow_copy(in_mesh)

            # extract input arrays
            arrays = out_mesh.get_point_arrays()

            # read common variables
            lat = arrays['lat']
            lon = arrays['lon']
            time = arrays['time']
            date = arrays['date']
            time = time/365. + 1850.
            lf = arrays['LANDFRAC']

            # read river data
            tmq = arrays['TMQ']
            ux = arrays[self.ufield]
            vy = arrays[self.vfield]

            # compute wind
            # r2d = 45.0/atan(1.0)     ; conversion factor (radians to degrees)
            # wsp = (ux^2 + vy^2)^0.5
            # wsp@units = "m/s"
            # wdir = atan2(ux, vy) * r2d + 180   ; ===> if u/v = 10, dir = 225.0
            # wdir@units = "degrees"
            # copy_VarCoords(ux,wsp)
            # copy_VarCoords(ux,wdir)
            # printVarSummary(wdir)

            r2d = 45.0/atan(1.0)     # conversion factor (radians to degrees)
            wsp = (ux^2 + vy^2)^0.5
            # wsp@units = "m/s"

            wdir = atan2(ux, vy) * r2d + 180   # ===> if u/v = 10, dir = 225.0
            # wdir@units = "degrees"

            # copy_VarCoords(ux,wsp)
            # copy_VarCoords(ux,wdir)
            #ux = wsp #??
            #ux = wdir #?? why is this the same?

            # ;=== compute zonal mean anomalies
            # ztmq = dim_avg_Wrap(tmq(time|:,lat|:,lon|:))
            # printVarSummary(ztmq)
            # ztmqmx = dim_max(tmq(time|:,lat|:,lon|:))
            # ztmqmx!0 = "time"
            # ztmqmx!1 = "lat"
            #ztmqmx&lat = lat
            # printVarSummary(ztmqmx)
            # print(max(ztmqmx))

            ztmq = np.mean(tmq, axis=0) # dim_avg_Wrap(tmq(time|:,lat|:,lon|:))
            ztmqmx = np.amax(tmq, axis=0) #dim_max(tmq(time|:,lat|:,lon|:))
            # ztmqmx!0 = "time"
            # ztmqmx!1 = "lat"
            #ztmqmx&lat = lat


            """
            ;=== isolate search and save(for nc and plots) regions
            if(iflip.eq.1)then
             tmqF = lonFlip(tmq)
             uxF = lonFlip(ux)
             vyF = lonFlip(vy)
             wspF = lonFlip(wsp)
             wdirF = lonFlip(wdir)
             lfF = lonFlip(lf)
             lonF = lonFlip(lon)
             lonF(0:dimsizes(lon)/2-1) = lonF(0:dimsizes(lon)/2-1) - 360.
            else
             tmqF = tmq
             uxF = ux
             vyF = vy
             wspF = wsp
             wdirF = wdir
             lfF = lf
             lonF = lon
            end if
            """

            tmqF = tmq
            uxF = ux
            vyF = vy
            wspF = wsp
            wdirF = wdir
            lfF = lf
            lonF = lon

            if self.iflip == 1:
                tmqF = np.flip(tmq, axis=0)
                uxF = np.flip(ux, axis=0)
                vyF = np.flip(vy, axis=0)
                wspF = np.flip(wsp, axis=0)
                wdirF = np.flip(wdir, axis=0)
                lfF = np.flip(lf, axis=0)
                lonF = np.flip(lon)


            """
            ; --search and save
            tmq_reg = tmqF(:,{latmin:latmax},{lonmin:lonmax})
            tmq_save = tmqF(:,{platmin:platmax},{plonmin:plonmax})

            ztmq_reg = ztmq(:,{latmin:latmax})
            ztmqmx_reg = ztmqmx(:,{latmin:latmax})

            ux_reg = uxF(:,{latmin:latmax},{lonmin:lonmax})
            ux_save = uxF(:,{platmin:platmax},{plonmin:plonmax})
            delete(uxF)

            vy_reg = vyF(:,{latmin:latmax},{lonmin:lonmax})
            vy_save = vyF(:,{platmin:platmax},{plonmin:plonmax})
            delete(vyF)

            wsp_reg = wspF(:,{latmin:latmax},{lonmin:lonmax})
            wsp_save = wspF(:,{platmin:platmax},{plonmin:plonmax})
            delete(wspF)

            wdir_reg = wdirF(:,{latmin:latmax},{lonmin:lonmax})
            wdir_save = wdirF(:,{platmin:platmax},{plonmin:plonmax})
            delete(wdirF)

            lf_reg = dble2flt(lfF({latmin:latmax},{lonmin:lonmax}))
            lf_save =dble2flt(lfF({platmin:platmax},{plonmin:plonmax}))
            delete(lfF)

            lon_reg = lonF({lonmin:lonmax})
            lat_reg = lat({latmin:latmax})
            nrlon = dimsizes(lon_reg)
            nrlat = dimsizes(lat_reg)
            lon_save = lonF({plonmin:plonmax})
            lat_save = lat({platmin:platmax})
            nslon = dimsizes(lon_save)
            nslat = dimsizes(lat_save)
            """

            region = self.active_region

            latmin = region.latmin
            latmax = region.latmin

            platmin = self.platmin
            platmax = self.platmax

            lonmin = self.active_region.lonmin
            lonmax = self.active_region.lonmin

            plonmin = self.plonmin
            plonmax = self.plonmax

            tmq_reg = tmqF[:,latmin:latmax,lonmin:lonmax]
            tmq_save = tmqF[:,platmin:platmax,plonmin:plonmax]

            ztmq_reg = ztmq[:,latmin:latmax]
            ztmqmx_reg = ztmqmx[:,latmin:latmax]

            ux_reg = uxF[:,latmin:latmax,lonmin:lonmax]
            ux_save = uxF[:,platmin:platmax,plonmin:plonmax]

            vy_reg = vyF[:,latmin:latmax,lonmin:lonmax]
            vy_save = vyF[:,platmin:platmax,plonmin:plonmax]

            wsp_reg = wspF[:,latmin:latmax,lonmin:lonmax]
            wsp_save = wspF[:,platmin:platmax,plonmin:plonmax]

            wdir_reg = wdirF[:,latmin:latmax,lonmin:lonmax]
            wdir_save = wdirF[:,platmin:platmax,plonmin:plonmax]

            lf_reg = lfF[latmin:latmax,lonmin:lonmax]
            lf_save = lfF[platmin:platmax,plonmin:plonmax]

            lon_reg = lonF[lonmin:lonmax]
            lat_reg = lat[latmin:latmax]

            #nrlon = dimsizes(lon_reg)
            #nrlat = dimsizes(lat_reg)
            #lon_save = lonF({plonmin:plonmax})
            #lat_save = lat({platmin:platmax})
            #nslon = dimsizes(lon_save)
            #nslat = dimsizes(lat_save)

            """
            ;=== call fortran shared object to find all dates( time slices) with ARs for given region
            lat_reg_f = dble2flt(lat_reg)
            lon_reg_f = dble2flt(lon_reg)
            ;timef = dble2flt(time)
            imax = ntimes
            nevents = 0
            event_dates = new(imax,integer)
            event_times = new(imax,double)
            event_lats = new(imax,float)
            ar_find_zn_lat::ar_find_zn_lat(tmq_reg,ztmq_reg,ztmqmx_reg,wsp_reg,wdir_reg,lf_reg,date,time,lat_reg_f,lon_reg_f, \
                            nrlat,nrlon,ntimes,imax,wsp_thresh, \
                            wdir_gt_thresh,wdir_lt_thresh,length_thresh,width_thresh,event_dates, \
                            event_times,event_lats,nevents)
            print("nevents = " + nevents)
            if(nevents.eq.0)then
             print("nevents = 0... exiting")
             exit
            end if
            ar_date = event_dates(0:nevents-1)
            ar_time = event_times(0:nevents-1)
            ar_lat = event_lats(0:nevents-1)
            print(ar_date)
            print(ar_time)
            print(ar_lat)
            """

            ar_find_zn_lat(tmq_reg, ztmq_reg, ztmqmx_reg, wsp_reg, wdir_reg, lf_reg, \
                           date, time, lat_reg, lon_reg, nrlat, nrlon, ntimes,  ntimes, \
                           imax, region.wsp_thresh, region.wdir_gt_thresh, region.wdir_lt_thresh, \
                           self.length_thresh, self.width_thresh, event_dates, event_times, event_lats, nevents)


            """
            ;====    i.e. isolating event timeslices
            ;    create new arrays for all vars with timeslices with events only

            lat_save_f = dble2flt(lat_save)
            lon_save_f = dble2flt(lon_save)
            ar_tmq = new((/nevents,nslat,nslon/),float)
            ar_wsp = new((/nevents,nslat,nslon/),float)
            ar_dir = new((/nevents,nslat,nslon/),float)
            ar_ux = new((/nevents,nslat,nslon/),float)
            ar_vy = new((/nevents,nslat,nslon/),float)
            ;==sort
            ar_sort::ar_sort(tmq_save,wsp_save,wdir_save,ux_save,vy_save,lf_save,\
                             lat_save_f,lon_save_f,time, \
                             nslat,nslon,ntimes,ar_date,ar_time,nevents, \
                             ar_tmq,ar_wsp,ar_dir,ar_ux,ar_vy)
            printVarSummary(ar_tmq)
            """
            # ar_sort(tmq_save, wsp_save,

            """
            ;==prep for outputfile
            ar_date!0 = "time"
            ar_time!0 = "time"
            ar_time&time = ar_time
            ar_lat!0 = "time"
            ar_tmq!0 = "time"
            ar_tmq!1 = "lat"
            ar_tmq!2 = "lon"
            ar_tmq&lat = lat_save
            ar_tmq&lon = lon_save
            copy_VarCoords(ar_tmq,ar_ux)
            copy_VarCoords(ar_tmq,ar_vy)
            copy_VarCoords(ar_tmq,ar_wsp)
            copy_VarCoords(ar_tmq,ar_dir)
            printVarSummary(ar_tmq)
            """

            """
            ;=== write out netcdf file for dates where ARs exist
            ;- write full global arrays as well as subsets

             ndates = dimsizes(ar_date)
             system("rm "+outfile)
             fout  = addfile ( outfile , "c" )      ; open the output file
             dimNames = (/ "lat","lon","time" /) ;    Specify output dimensions
             dimSizes = (/ nslat,nslon,nevents  /)
             dimUnlim = (/ False, False,True  /)
             filedimdef (fout, dimNames,dimSizes,dimUnlim)
             fout->TMQ = ar_tmq
             fout->$ufield$= ar_ux
             fout->$vfield$ = ar_vy
             fout->WSP = ar_wsp
             fout->WDIR = ar_dir
             fout->LANDFRAC= lf_save
             fout->date = ar_date
             fout->time = ar_time

             fout->lat_landfall = ar_lat
             fout->lat = lat_save
             fout->lon = lon_save
             fout@input = fout_string
             fout@casename = casename
             fout@nclscript = "atmos_river_find_lats_tmq_Zanom_ZN_20C.ncl  c.shields july2014"
             fout@reference = "Neiman et al 2007 J Hydromet, Newman et al 2012"
             fout@thresholds = threshold_doc
             print("nc file written")
            """

            arrays['ar_tmq'] = ar_tmq
            arrays['ar_ux'] = ar_ux
            arrays['ar_vy'] = ar_vy
            arrays['ar_wsp'] = ar_wsp
            arrays['ar_wdir'] = ar_dir
            arrays['ar_ladfrc'] = lf_save
            arrays['ar_date'] = ar_date
            arrays['ar_time'] = ar_time

            arrays['lat_landfall'] = ar_lat
            arrays['lat_save'] = lat_save
            arrays['lon_save'] = lon_save

            return out_mesh
        return execute_callback
