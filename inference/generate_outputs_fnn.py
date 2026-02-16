from inference_l3_func import generate_corrections_fnn, create_out_nc, convert_nc_to_grib
from types import SimpleNamespace
import numpy as np
from model.fcnn import FCNN
import datetime as dt
import sys

from datetime import datetime
start_time = datetime.now()

config = SimpleNamespace(
    #ERA5 U10S input dir
    nwp_dir="../data/inputs/ERA5_U10S/",
    #Glob currents dir
    #2023
    currents_dir="../data/inputs/GLOBCURRENT/",
    #2022
    #currents_dir="/mnt/smart/scratch/satwinds/makarova/ERAS_ML/data/CMEMS_currents/MULTIOBS_GLO_PHY_MYNRT_015_003/cmems_obs_mob_glo_phy-cur_nrt_0.25deg_P1D-m_202311/",
    #Directory to store output netCDF files
    nc_out_dir="../data/ouputs/nc/",
    #Directory to store output grib files
    grib_out_dir="../data/ouputs/grib/",
    #Path to metadata files with normalization parameters
    norm_metadata_files={'inputs': '../../metadata/input_norm.csv',
                         'targets': '../../metadata/target_norm.csv'},
    #ERA5 analysis times
    nwp_an=[6],
    #List of ERA5 FC times
    nwp_fc=np.arange(3, 4),
    #Prefix of ERA5 files
    nwp_prefix="nwp_",
    #Extention of ERA5 grib files
    nwp_extention=".grib",
    #Prefix of currents filename
    #2023
    currents_prefix="dataset-uv-rep-daily_",
    #2022
    #currents_prefix="dataset-uv-nrt-daily_",
    #Target grid (other grids not tested)
    target_lon_grid=np.arange(0.0625, 360, .125),
    target_lat_grid=np.arange(-90 + 0.0625, 90, .125),
    #Names of input variables used in DNN model
    input_var_names=['lon', 'lat', 'eastward_model_wind', 'northward_model_wind','model_speed', 'model_dir',
                     'se_model_wind_curl', 'se_model_wind_divergence',
                     'msl', 'air_temperature', 'q', 'sst', 'sst_dx', 'sst_dy', 'uo', 'vo'], # 'date'],  'era5_fc_hour'],
    #This variables will be used to extract land mask and remove corrections over land
    variable2extractmask='uo',
    #Target variable names as in norm_metadata_files
    target_var_names=['u_diff', 'v_diff'],
    #Path to metadata file required for L3 collocations
    nwp_l3_proc_metadata="../../metadata/nwp_l3_preproc.csv",
    #DNN model class
    model=FCNN,
    #Path to model weights
    model_path='../weights/02',
    #Configuration of the model
    batch_size=512*1024,
    dropout=0.25,
    hidden_layers=[1024, 512, 256, 128, 64],
    #model_conf_name=""

)

start_date = dt.datetime.strptime(sys.argv[1], '%Y%m%d')
end_date = dt.datetime.strptime(sys.argv[2], '%Y%m%d')

date_dt = start_date
while date_dt <= end_date:

    date_str = date_dt.strftime('%Y%m%d')
    for an in config.nwp_an:
        for fc in config.nwp_fc:

            print(f"Processing {date_str} AN {an:02d} FC {fc:02d}")

            lons, lats, model_u, model_v, corrections = generate_corrections_fnn(config, date_str, an, fc)
            u_ML = np.nansum([model_u, corrections[0]], axis=0)
            v_ML = np.nansum([model_v, corrections[1]], axis=0)

            nc_out_fn = f"{config.nc_out_dir}ERAS_ML_{date_str}_{an:02d}_{fc:02d}.nc"
            grib_out_fn = f"{config.grib_out_dir}nwp_{date_dt.strftime('%Y%j')}_{an:02d}_{fc:02d}.grib"

            output = create_out_nc(nc_out_fn, config, u_ML, v_ML, corrections[0], corrections[1], date_dt, an, fc)
            print(output)
            #For grib conversion cdo package is required
            '''
            output = convert_nc_to_grib(nc_out_fn, grib_out_fn, date_dt, an, fc, cdo_path='cdo')
            print(output)
            '''
    date_dt = date_dt + dt.timedelta(days=1)


end_time = datetime.now()
print(f'Run time for period {sys.argv[1]} - {sys.argv[2]}: {end_time - start_time}')