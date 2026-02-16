import torch
from collocations.nwp_l3_coloc_smart import RegularGridCollocator
from preprocessing_pytorch import Normalizer, denorm_torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import datetime as dt
from netCDF4 import Dataset
import subprocess

def create_out_nc(out_fn, config, u_ML, v_ML, u_diff, v_diff, date, AN, FC):
    dateRef = dt.datetime(1990, 1, 1, 0, 0, 0)
    missing_valueu = -32767
    dateUTC = date + dt.timedelta(hours=int(AN) + int(FC))
    t = (dateUTC-dateRef).total_seconds()

    ncfile = Dataset(out_fn, 'w', format='NETCDF4')
    lat_dim = ncfile.createDimension('lat', 1440)  # latitude axis
    lon_dim = ncfile.createDimension('lon', 2880)  # longitude axis
    time_dim = ncfile.createDimension('time', 1)  # unlimited axis (can be

    LAT = ncfile.createVariable('lat', float, ('lat',))
    LAT.standard_name = 'latitude'
    LAT.long_name = 'latitude'
    LAT.units = 'degrees_north'
    ncfile.variables['lat'][:] = config.target_lat_grid

    LON = ncfile.createVariable('lon', float, ('lon',))
    LON.standard_name = 'longitude'
    LON.long_name = 'longitude'
    LON.units = 'degrees_east'
    ncfile.variables['lon'][:] = config.target_lon_grid

    TIME = ncfile.createVariable('time', int, ('time',))
    TIME.standard_name = 'time'
    TIME.long_name = 'time'
    TIME.units = 'seconds since 1990-01-01 00:00:00'
    ncfile.variables['time'][:] = t

    U = ncfile.createVariable('ERAS_ML_u10s', np.short, ('time', 'lat', 'lon'),
                              fill_value=missing_valueu)
    U.standard_name = 'eastward_wind'
    U.long_name = 'ERAS_ML stress equivalent model wind u component at 10 m'
    U.units = 'm s-1'
    U.add_offset = 0.
    U.scale_factor = 0.01
    ncfile.variables['ERAS_ML_u10s'][:] = u_ML

    V = ncfile.createVariable('ERAS_ML_v10s', np.short, ('time', 'lat', 'lon'),
                              fill_value=missing_valueu)
    V.standard_name = 'northward_wind'
    V.long_name = 'ERAS_ML stress equivalent model wind v component at 10 m'
    V.units = 'm s-1'
    V.add_offset = 0.
    V.scale_factor = 0.01
    ncfile.variables['ERAS_ML_v10s'][:] = v_ML


    NN_cor_u = ncfile.createVariable('diff_u10s', np.short, ('time', 'lat', 'lon'),
                               fill_value=missing_valueu)
    NN_cor_u.standard_name = 'eastward_wind'
    NN_cor_u.long_name = 'ERAS_ML - ERA5 difference u component'
    NN_cor_u.units = 'm s-1'
    NN_cor_u.add_offset = 0.
    NN_cor_u.scale_factor = 0.01
    ncfile.variables['diff_u10s'][:] = u_diff

    NN_cor_v = ncfile.createVariable('diff_v10s', np.short, ('time', 'lat', 'lon'),
                               fill_value = missing_valueu)

    NN_cor_v.standard_name = 'northward_wind'
    NN_cor_v.long_name = 'ERAS_ML - ERA5 difference v component'
    NN_cor_v.units = 'm s-1'
    NN_cor_v.add_offset = 0.
    NN_cor_v.scale_factor = 0.01
    ncfile.variables['diff_v10s'][:] = v_diff
    attdict = {
        'date_created': dt.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    }
    ncfile.setncatts(attdict)
    ncfile.close()
    return "Output " + out_fn + " created"

def convert_nc_to_grib(nc_fn, grib_fn, date, an, fc, var_names="ERAS_ML_u10s,ERAS_ML_v10s",
                       cdo_path="/home/makarova/utils/bin/cdo"):
    """
    Routine that copies ML_u10s,ML_v10s variables from output NetCDF files to grib and sets parameters of analisys and forecast times
    :param nc_fn: NetCDF file to be converted to grib
    :param grib_fn: Output grib file
    :param date: Analysis cycle date in datetime format
    :param an: analysis time in string format, for example '06'
    :param fc: forecast time in string format, for example '03'
    :return: string that the nc_fn was converted to grib_fn
    """
    grib_tmp = grib_fn + "tmp"
    cmd = f"{cdo_path} -a -f grb copy -selname,{var_names} {nc_fn} {grib_tmp}"
    p = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    dum = str(p.stdout.read(), 'utf-8')[:-1]
    cmd = f"{cdo_path} -f grb chparam,1,165.128,2,166.128 {grib_tmp} {grib_fn}"
    p = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    dum = str(p.stdout.read(), 'utf-8')[:-1]
    cmd = f"grib_set -s dataDate={date.strftime('%Y%m%d')} {grib_fn} {grib_tmp}"
    p = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    dum = str(p.stdout.read(), 'utf-8')[:-1]
    cmd = f"grib_set -s endStep={fc},dataTime={an}00 {grib_tmp} {grib_fn}"
    p = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    dum = str(p.stdout.read(), 'utf-8')[:-1]
    cmd = f"rm {grib_tmp}"
    p = subprocess.Popen([cmd], stdout=subprocess.PIPE, shell=True)
    dum = str(p.stdout.read(), 'utf-8')[:-1]
    return f"{nc_fn} converted to {grib_fn}"


def generate_corrections_fnn(config, date_str, an, fc):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    l3_collocator = RegularGridCollocator(config)
    interpolated_dataset = l3_collocator.collocate_regular_datasets(date_str, an, fc)
    data_shape = interpolated_dataset.shape

    input_norm_metadata = config.norm_metadata_files['inputs']
    output_norm_metadata = config.norm_metadata_files['targets']

    input_normalizer = Normalizer(input_norm_metadata)
    inputs_norm, input_vars_norm = input_normalizer.normalize(interpolated_dataset, config.input_var_names)
    N_norm_vars = len(input_vars_norm)

    inputs_norm = inputs_norm.reshape([N_norm_vars, -1])
    inputs_norm = inputs_norm.transpose()

    inputs_norm_tf = torch.Tensor(inputs_norm)
    input_batch_generator = DataLoader(inputs_norm_tf, batch_size=config.batch_size)
    predictions = []

    model = config.model(config, N_norm_vars, device)
    model.load_state_dict(torch.load(config.model_path, map_location=device))
    model.to(device)
    model.eval()

    pbar = tqdm(input_batch_generator)
    for i, input_batch in enumerate(pbar):
        prediction_batch = model(input_batch.to(device)).cpu().detach().numpy()
        predictions.append(prediction_batch)
    predictions = np.vstack(predictions)
    denorm_predictions = denorm_torch(predictions, output_norm_metadata)
    denorm_predictions = denorm_predictions.transpose().reshape([-1, data_shape[1], data_shape[2]])

    mask_var_index = config.input_var_names.index(config.variable2extractmask)
    mask = np.isnan(interpolated_dataset[mask_var_index])
    denorm_predictions[:, mask] = np.nan

    lons = interpolated_dataset[config.input_var_names.index('lon')]
    lats = interpolated_dataset[config.input_var_names.index('lat')]
    model_u = interpolated_dataset[config.input_var_names.index('eastward_model_wind')]
    model_v = interpolated_dataset[config.input_var_names.index('northward_model_wind')]

    return lons, lats, model_u, model_v, denorm_predictions