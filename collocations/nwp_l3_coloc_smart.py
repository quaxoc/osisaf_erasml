import pygrib
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
from glob import glob
import numpy as np
import datetime as dt
import pandas as pd
from collocations.curldiv import CurlDivRegularGrid, GradientsDxDy
import calendar

class RegularGridCollocator:
    def __init__(self, config):
        """
        Interpolates variables to desired grid and collocates them in space
        :param config: should contain
        config = SimpleNamespace(
        nwp_dir="/mnt/smart/users/makarova/ERAS_ML/data/ERA5_U10S/",
        #Glob currents dir
        currents_dir="/mnt/smart/users/makarova/ERAS_ML/data/CMEMS_currents/",
        #Prefix of ERA5 files
        nwp_prefix="nwp_",
        #Extention of ERA5 grib files
        nwp_extention=".grib",
        #Prefix of currents filename
        currents_prefix="dataset-uv-rep-daily_",
        #Target grid (other grids not tested)
        target_lon_grid=np.arange(0.0625, 360, .125),
        target_lat_grid=np.arange(-90 + 0.0625, 90, .125),
        #Names of input variables used in DNN model
        input_var_names=['lon', 'lat', 'eastward_model_wind', 'northward_model_wind', 'model_speed', 'model_dir',
                         'msl', 'air_temperature', 'q', 'sst', 'uo', 'vo'],
        #Path to metadata file required for L3 collocations
        nwp_l3_proc_metadata="../../metadata/nwp_l3_preproc.csv",
        )
        nwp_l3_preproc.csv:
            variable,source,name_source,derived
            lon,config,,1
            lat,config,,1
            eastward_model_wind,nwp,u10n,0
            northward_model_wind,nwp,v10n,0
            model_speed,nwp,,1
            model_dir,nwp,,1
            msl,nwp,msl,0
            air_temperature,nwp,t,0
            q,nwp,q,0
            sst,nwp_an,sst,0
            uo,currents,uo,0
            vo,currents,vo,0

        """
        self.config = config
        self.nwp_l3_proc_metadata = pd.read_csv(config.nwp_l3_proc_metadata)
        self.lats_mesh, self.lons_mesh = np.meshgrid(config.target_lat_grid, config.target_lon_grid, indexing='ij')
        self.input_var_names = config.input_var_names

    def get_nwp_date_hour(self, date_str: str, an: int, fc: int):
        hour = an + fc
        if hour >= 24:
            date_dt = dt.datetime.strptime(date_str, '%Y%m%d')
            date_dt = date_dt + dt.timedelta(days=1)
            date_str = date_dt.strftime("%Y%m%d")
            hour = hour - 24
        return date_str, hour

    def read_nwp_var(self, fpath: str, var_name: str):
        """
        Reads the values for the listed variables from the grib file
        :param fpath: Path to the grib file
        :param var_namelist: List with the variable names
        :return: list of numpy arrays or numpy masked arrays with the values
        """
        with pygrib.open(fpath) as grbs:
            selected_grbs = grbs.select(cfVarName=var_name)
            for grb in selected_grbs:
                nwp_var_data = grb.values
        return nwp_var_data

    def get_nwp_lat_lons(self, fpath: str):
        """
        Reads latitudes and longitudes from a Grib file. Required for L2Collocator
        :param fpath: Path to grib file
        :return:
        Tuple of numpy arrays of latitudes and longitudes
        """
        grbs = pygrib.open(fpath)
        grb = grbs.read(1)[0]
        nwp_lats, nwp_lons = grb.latlons()
        return nwp_lats, nwp_lons

    def get_1d_lat_lons(self, fpath):
        lats, lons = self.get_nwp_lat_lons(fpath)
        lat_1d = lats[:, 0]
        lon_1d = lons[0, :]
        return lat_1d, lon_1d

    def get_currents_lat_lons(self, fpath: str):
        """
        Reads latitudes and longitudes from a Grib file. Required for L2Collocator
        :param fpath: Path to grib file
        :return:
        Tuple of numpy arrays of latitudes and longitudes
        """
        with Dataset(fpath) as f:
            lon = f.variables['longitude'].__array__().data
            lat = f.variables['latitude'].__array__().data
        # Changing from [-180 180] convention to [0 360]
        return lat, lon

    def read_currents_var(self, fpath: str, var_name: str):
        """
        Reads the values for the listed variables from the grib file
        :param nwp_fpath: Path to the grib file
        :param var_namelist: List with the variable names
        :return: list of numpy arrays or numpy masked arrays with the values
        """
        with Dataset(fpath) as f:
            nwp_var_data = f.variables[var_name].__array__()[0][0]
        return nwp_var_data

    def calculate_dir(self, u, v):
        return (180 / np.pi * np.arctan2(u, v) + 360) % 360

    def calculate_norm(self, u, v):
        return (u ** 2 + v ** 2) ** 0.5

    def calculate_model_speed(self, fn):
        u = self.read_nwp_var(fn, "u10n")
        v = self.read_nwp_var(fn, "v10n")
        wsp = self.calculate_norm(u, v)
        return wsp

    def calculate_model_dir(self, fn):
        u = self.read_nwp_var(fn, "u10n")
        v = self.read_nwp_var(fn, "v10n")
        wdir = self.calculate_dir(u, v)
        return wdir

    def get_interpolated_curl_div(self, fn, convert_180_360=False):
        u = self.read_nwp_var(fn, "u10n")
        v = self.read_nwp_var(fn, "v10n")
        nwp_lats, nwp_lons = self.get_nwp_lat_lons(fn)
        derivatives = CurlDivRegularGrid(u, v, nwp_lats, nwp_lons, self.lats_mesh, self.lons_mesh)
        curl = derivatives.get_curl()
        divergence = derivatives.get_divergence()
        return curl, divergence

    def get_interpolated_sst_dx_dy(self, fn):
        sst = self.read_nwp_var(fn, "sst")
        nwp_lats, nwp_lons = self.get_nwp_lat_lons(fn)
        if isinstance(sst, np.ma.masked_array):
            mask = sst.mask
            var_data = sst.data
            var_data[mask] = np.nan
        gradients = GradientsDxDy(nwp_lats, nwp_lons, self.lats_mesh, self.lons_mesh)
        sst_dx, sst_dy = gradients.get_gradients(sst)
        return sst_dx, sst_dy

    def extended_1d_lons(self, lons_1d, positions):
        lons_extended_1d = lons_1d.copy()
        lon_delta = lons_1d[1] - lons_1d[0]
        b = np.arange(lons_1d[0] - lon_delta*positions, lons_1d[0],  lon_delta)
        e = (np.arange(positions)+1)*lon_delta + lons_1d[-1]
        lons_extended_1d = np.concatenate((b, lons_extended_1d, e))
        return lons_extended_1d

    def extend_nwp_var(self, nwp_var, positions):
        if isinstance(nwp_var, np.ma.masked_array):
            nwp_var_extended = np.ma.zeros([nwp_var.shape[0], nwp_var.shape[1] + positions * 2])
        else:
            nwp_var_extended = np.zeros([nwp_var.shape[0], nwp_var.shape[1] + positions*2])
        nwp_var_extended[:, :positions] = nwp_var[:, -positions:]
        nwp_var_extended[:, positions:-positions] = nwp_var
        nwp_var_extended[:, -positions:] = nwp_var[:, :positions]
        return nwp_var_extended

    def interpolate_var_reg2reg(self, orig_var, orig_lat_1d, orig_lon_1d, target_lats_mesh, target_lons_mesh,
                                masked=False, convert_180_360=False, method='linear', extend=True, extend_positions=2):
        if convert_180_360:
            #roll_id = len(orig_lon_1d)//2
            lons = np.where(orig_lon_1d < 0, orig_lon_1d + 360, orig_lon_1d)
            roll_id = np.argmin(lons)
            lons = np.roll(lons, roll_id)
            orig_var = np.roll(orig_var, roll_id, axis=1)
        else:
            lons = orig_lon_1d

        if extend:
            lons = self.extended_1d_lons(lons, extend_positions)
            orig_var = self.extend_nwp_var(orig_var, extend_positions)
        # Check if original lats are in ascending order
        
        if orig_lat_1d[0] > orig_lat_1d[-1]:
            var_data = np.flip(orig_var, axis=0)
            lats = np.flip(orig_lat_1d)
        else:
            lats = orig_lat_1d
            var_data = orig_var
        if masked:
            var_mask = var_data.mask
            var_data = var_data.data
            var_data[var_mask] = np.nan
            method = 'nearest'
        interp = RegularGridInterpolator((lats, lons), var_data,
                                         method=method, bounds_error=False)
        interp_data = interp((target_lats_mesh, target_lons_mesh))
        return interp_data

    def get_normalized_date(self, date_str: str, an: int, fc: int):
        date_str, hour = self.get_nwp_date_hour(date_str, an, fc)
        date_dt = dt.datetime.strptime(date_str, '%Y%m%d')
        year = date_dt.year
        day = int(date_dt.strftime('%j'))
        days_in_year = 366 if calendar.isleap(year) else 365
        return day/days_in_year


    def collocate_regular_datasets(self, date_str, an, fc):
        date_dt = dt.datetime.strptime(date_str, '%Y%m%d')
        interpolated_dataset = np.zeros(
            (len(self.config.input_var_names), len(self.config.target_lat_grid), len(self.config.target_lon_grid)))
        for input_var_id, input_var_name in enumerate(self.config.input_var_names):
            metadata = self.nwp_l3_proc_metadata[self.nwp_l3_proc_metadata['variable'] == input_var_name]
            print(f"Intepolating variable {input_var_name} with metadata \n {metadata}")
            if len(metadata) > 0:
                metadata_source = metadata["source"].values[0]
                if metadata_source == 'config':
                    if input_var_name=='lon':
                        interpolated_dataset[input_var_id] = self.lons_mesh
                    if input_var_name=='lat':
                        interpolated_dataset[input_var_id] = self.lats_mesh
                    if input_var_name == 'era5_fc_hour':
                        interpolated_dataset[input_var_id] = np.full(self.lats_mesh.shape, fc)
                    if input_var_name == 'date':
                        day_of_year = self.get_normalized_date(date_str, an, fc)
                        interpolated_dataset[input_var_id] = np.full(self.lats_mesh.shape, day_of_year)
                else:
                    method = 'linear'
                    convert_180_360 = False
                    interpolation_not_done = True
                    if metadata_source == 'nwp':
                        fn = f"{self.config.nwp_dir}/{date_dt.strftime('%Y')}/{self.config.nwp_prefix}{date_dt.strftime('%Y%j')}_{an:02d}_" \
                             f"{fc:02d}{self.config.nwp_extention}"
                        masked = False
                        lats_1d, lons_1d = self.get_1d_lat_lons(fn)
                        reader = self.read_nwp_var
                    elif metadata_source == 'nwp_an':
                        fn = f"{self.config.nwp_dir}/{date_dt.strftime('%Y')}/{self.config.nwp_prefix}{date_dt.strftime('%Y%j')}_{an:02d}_00{self.config.nwp_extention}"
                        masked = True
                        lats_1d, lons_1d = self.get_1d_lat_lons(fn)
                        reader = self.read_nwp_var
                    elif metadata_source == 'currents':
                        nwp_real_date, nwp_real_hour = self.get_nwp_date_hour(date_str, an, fc)
                        nwp_real_date_dt = dt.datetime.strptime(nwp_real_date, "%Y%m%d")
                        dir_prefix = f"/{nwp_real_date_dt.strftime('%Y')}/{nwp_real_date_dt.strftime('%m')}/"
                        fn = glob(f"{self.config.currents_dir}{dir_prefix}{self.config.currents_prefix}{nwp_real_date}*")[0]
                        masked = True
                        method = 'nearest'
                        convert_180_360 = True
                        lats_1d, lons_1d = self.get_currents_lat_lons(fn)
                        reader = self.read_currents_var
                    if not metadata['derived'].values[0]:
                        var_data = reader(fn, metadata['name_source'].values[0])
                    else:
                        if input_var_name in ['se_model_wind_curl', 'se_model_wind_divergence']:
                            curl, divergence = self.get_interpolated_curl_div(fn)
                            interpolation_not_done = False
                        if input_var_name in ['sst_dx', 'sst_dy']:
                            sst_dx, sst_dy = self.get_interpolated_sst_dx_dy(fn)
                            interpolation_not_done = False
                        if input_var_name == 'se_model_wind_curl':
                            interpolated_var_data = curl
                        if input_var_name == 'se_model_wind_divergence':
                            interpolated_var_data = divergence
                        if input_var_name == 'sst_dx':
                            interpolated_var_data = sst_dx
                        if input_var_name == 'sst_dy':
                            interpolated_var_data = sst_dy
                        if input_var_name == 'model_speed':
                            var_data = self.calculate_model_speed(fn)
                        if input_var_name == 'model_dir':
                            var_data = self.calculate_model_dir(fn)
                            # It can avoid issues of linear interpolation for directions in limits 360 -> 0
                            method = 'nearest'

                    if interpolation_not_done:
                        interpolated_var_data = self.interpolate_var_reg2reg(var_data, lats_1d, lons_1d,
                                                                             self.lats_mesh, self.lons_mesh,
                                                                             masked=masked, method=method,
                                                                             convert_180_360=convert_180_360)
                    interpolated_dataset[input_var_id] = interpolated_var_data
        return interpolated_dataset