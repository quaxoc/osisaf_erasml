import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
import torch
from dataclasses import dataclass
import os
from glob import glob
from torch.utils.data import ConcatDataset
from collections.abc import Iterable


# from scipy.spatial import Delaunay
# from scipy.interpolate import LinearNDInterpolator
def denorm(data, metadata_file):
    metadata = pd.read_csv(metadata_file)
    var_mean = metadata['mean'].values
    var_std = metadata['std'].values
    denorm_output = data * var_std + var_mean
    return denorm_output


def multiply_along_axis(A, B, axis):
    return np.swapaxes(np.swapaxes(A, axis, -1) * B, -1, axis)


def sum_along_axis(A, B, axis):
    return np.swapaxes(np.swapaxes(A, axis, -1) + B, -1, axis)


def denorm_torch(data, metadata_file, axis=0):
    metadata = pd.read_csv(metadata_file)
    var_mean = metadata['mean'].values
    #var_mean = np.reshape(var_mean, [len(var_mean), 1, 1])
    var_std = metadata['std'].values
    #var_std = np.reshape(var_std, [len(var_std), 1, 1])
    #denorm_output = multiply_along_axis(var_std, data, axis=axis)
    #denorm_output = sum_along_axis(var_mean, denorm_output, axis=axis)
    denorm_output = var_std * data
    denorm_output = var_mean + denorm_output
    return denorm_output


class Normalizer():
    '''
    Normalizes the dataset, the normalization type and parameters are read from metadata csv file per each variable

    :param metadata_file: CSV file that describes the normalization type
    The required columns are 'variable' and 'type'
        variable: the nane of the variable, generally corresponds to the short name in the netCDF data file

        type: possible types are

            copy: the data is copied without the normalization

            minmax: for minmax normalization, for this variable min and max columns should be defined in the metadata

            norm: equivalent to StandardScaler, if the initial distribution is normal, normalizes to standard normal
                  distribution, mean and std columns should be present

            sin: used to normalize the latitudes

            sincos: will create 2 variables, var_name_sin and var_name_cos, used to normalize angular values


    Example of use:

    metadata/target_norm.csv:

    variable,type,mean,std
    u_diff,norm,0,1.363518482
    v_diff,norm,0,1.461169906

    >> target_metadata_file = 'metadata/target_norm.csv'

    >> target_var_names = ['u_diff', 'v_diff']

    >> target_normalizer = Normalizer(target_metadata_file)

    >> targets_norm, target_vars_norm = target_normalizer.normalize(targets, target_var_names)
    '''

    def __init__(self, metadata_file):
        self.metadata_filepath = metadata_file
        self.norm_metadata = pd.read_csv(metadata_file)

    def normalize_sincos(self, var):
        sin = np.sin(var * np.pi / 180)
        cos = np.cos(var * np.pi / 180)
        return sin, cos

    def normalize_sin(self, var):
        return np.sin(var * np.pi / 180)

    def normalize_min_max(self, var, min, max):
        # Normalization between [-1, 1] to correspond to sine and cosine normalization
        norm_var = 2 * (var - min) / (max - min) - 1
        return norm_var

    def StandardScaler(self, var, mean, std):
        return (var - mean) / std

    def normalize(self, dataset, var_names, verbose=0):
        """
        Normalizes the input dataset with the method and parameters set in the metadata file. If the input data is a
        masked array, after the normalization the masked values will be set to 0.
        :param dataset: Dataset where the first dimension is the variable id
        :param var_names: List with the variable names (same should be in the metadata file)
                         If variable is not present in the metadata file or the normalization type is not valid,
                         the variable will be skipped
        :return: Normalized dataset and a list of updated variable names if any new are created
        """
        vars_norm = []
        var_names_norm = np.array([])
        for i, var_name in enumerate(var_names):
            metadata = self.norm_metadata[self.norm_metadata['variable'] == var_name]
            if len(metadata) > 0:
                var_data = dataset[i]
                if isinstance(var_data, np.ma.masked_array):
                    filter = np.logical_or(var_data.mask, np.isnan(var_data))
                    var_data = var_data.data
                else:
                    filter = np.isnan(var_data)
                if verbose:
                    print(f"Detected {np.sum(np.isnan(var_data))} nan values in {var_name}")
                norm_type = metadata['type'].values[0]

                if norm_type == 'sincos':
                    if var_name == "date":
                        var_data = var_data*360
                    sin, cos = self.normalize_sincos(var_data)
                    sin[filter] = 0
                    cos[filter] = 0
                    var_names_norm = np.append(var_names_norm, [var_name + '_sin', var_name + '_cos'])
                    vars_norm.append(sin)
                    vars_norm.append(cos)
                else:
                    if norm_type == 'sin':
                        norm_data = self.normalize_sin(var_data)
                        var_names_norm = np.append(var_names_norm, var_name + '_sin')
                    elif norm_type == 'minmax':
                        var_min = metadata['min'].values[0]
                        var_max = metadata['max'].values[0]
                        norm_data = self.normalize_min_max(var_data, var_min, var_max)
                        var_names_norm = np.append(var_names_norm, var_name)
                    elif norm_type == 'norm':
                        var_mean = metadata['mean'].values[0]
                        var_std = metadata['std'].values[0]
                        norm_data = self.StandardScaler(var_data, var_mean, var_std)
                        var_names_norm = np.append(var_names_norm, var_name)
                    elif norm_type == 'copy':
                        var_names_norm = np.append(var_names_norm, var_name)
                        norm_data = var_data
                    else:
                        print(f'{norm_type} normalization type not found, skipping variable {var_name}')

                    norm_data[filter] = 0
                    vars_norm.append(norm_data)
            else:
                print(f'{var_name} is not in the metadata file {self.metadata_filepath}, skipping variable')

        dataset_norm = np.stack(vars_norm)
        if isinstance(dataset_norm, np.ma.masked_array):
            dataset_norm = dataset_norm.data
        return dataset_norm, var_names_norm


class SwathPatchGenerator():
    """
    split_swath: bool = False
    swath_split_axis: int = 1
    """

    def __init__(self, split_swath=False, swath_split_axis=2):
        self.split_swath = split_swath
        self.swath_split_axis = swath_split_axis

    def separate_ascat_swaths(self, data):
        """
        Separates left and right part of the ASCAT swath as there is a gap between them
        :param data: Input data Nd numpy array
        :return: tuple of 2 arrays for the left and right part splint in axis swath_split_axis
        """
        return np.split(data, 2, axis=self.swath_split_axis)

    def split2patches(self, data, height, overlap, offset=0):
        patches = []
        if self.split_swath:
            lr_data = self.separate_ascat_swaths(data)
        else:
            lr_data = [data]
        for swath_data in lr_data:
            i = offset
            if swath_data.shape[self.swath_split_axis] % 2 != 0:
                # print("Cropping first column of swath")
                swath_data = np.delete(swath_data, 0, axis=self.swath_split_axis)
            while (i + height) <= swath_data.shape[1]:
                patch = swath_data[:, i:i + height]
                patches.append(patch)
                i = int(i + height * (1 - overlap))
        return np.stack(patches, axis=0)


class TrainPatchGenerator(SwathPatchGenerator):
    def __init__(self, input_var_names, target_var_names, metadata_files, input_grid="l2", split_swath=True, input_dir=None,
                 output_dir=None, swath_split_axis=2, file_list=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_metadata_file = metadata_files['inputs']
        self.target_metadata_file = metadata_files['targets']
        self.scat_model_var_names = {'scat': ['eastward_wind', 'northward_wind'],
                                     'model': ['eastward_model_wind', 'northward_model_wind']}
        self.input_var_names = input_var_names
        self.target_var_names = target_var_names
        self.split_swath = split_swath
        self.swath_split_axis = swath_split_axis
        self.separator = '/'
        if input_dir:
            self.file_list = glob(input_dir + '*.nc')
            self.file_list.sort()
        else:
            self.file_list = file_list
        if input_grid == "l3":
            self.nc_reader = self.read_l3_nc_data
        else:
            self.nc_reader = self.read_l2_nc_data

    def create_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def read_l2_nc_data(self, fpath: str, var_names: Iterable) -> np.array:
        f = Dataset(fpath)
        selected_data = []
        hg_vars = ['se_model_wind_curl', 'se_model_wind_divergence']

        for idx, var in enumerate(var_names):
            var_data = f.variables[var].__array__()
            var_data.data[var_data.mask] = np.nan
            if var in hg_vars:
                restored_var = np.full(f.variables['lon'].__array__().shape, np.nan)
                var_data_cr = (var_data[:, 1:] + var_data[:, :-1]) / 2
                var_data_cr.mask = var_data[:, 1:].mask & var_data[:, :-1].mask
                var_data_cr = (var_data_cr[1:, :] + var_data_cr[:-1, :]) / 2
                var_data_cr.mask = var_data_cr[1:, :].mask & var_data_cr[:-1, :].mask
                var_data_cr[var_data_cr.mask] = np.nan
                restored_var[1:-1, 1:-1] = var_data_cr
                var_data = restored_var
            #print(var, var_data.shape)
            selected_data.append(var_data)
        f.close()
        data = np.ma.stack(selected_data)
        return data

    def read_l3_nc_data(self, fpath: str, var_names: Iterable) -> np.array:
        f = Dataset(fpath)
        selected_data = []
        hg_vars = ['se_model_wind_curl', 'se_model_wind_divergence']

        for idx, var in enumerate(var_names):
            var_data = f.variables[var].__array__()
            var_data.data[var_data.mask] = np.nan
            selected_data.append(var_data)
        f.close()
        data = np.ma.stack(selected_data)
        return data

    def calculate_target(self, fpath: str):
        scat_data = self.nc_reader(fpath, self.scat_model_var_names['scat'])
        model_data = self.nc_reader(fpath, self.scat_model_var_names['model'])
        targets = scat_data - model_data
        return targets

    def generate_batches(self, fpath, height, overlap=0.3, offset=0, discard_ratio=0.5, verbose=0):
        input_data = self.read_l2_nc_data(fpath, self.input_var_names)
        targets = self.calculate_target(fpath)
        input_normalizer = Normalizer(self.input_metadata_file)
        target_normalizer = Normalizer(self.target_metadata_file)

        inputs_norm, input_vars_norm = input_normalizer.normalize(input_data, self.input_var_names)
        targets_norm, target_vars_norm = target_normalizer.normalize(targets, self.target_var_names)

        input_patches = self.split2patches(inputs_norm, height, overlap, offset=offset)
        target_patches = self.split2patches(targets_norm, height, overlap, offset=offset)
        # Cleaning data with patches that have too much filtered values, for example over land
        if verbose:
            print("Patches shape ", input_patches.shape, target_patches.shape)
            print("Initial number of patches:", len(target_patches))
        idx2delete = []
        for patch_id, patch in enumerate(target_patches):
            ratio = np.sum(patch == 0) / patch.size
            if ratio > discard_ratio:
                idx2delete.append(patch_id)
        input_patches = np.delete(input_patches, idx2delete, axis=0)
        target_patches = np.delete(target_patches, idx2delete, axis=0)
        if verbose:
            print(f'Number of patches to discard {len(idx2delete)}')
            print("After cleaning:", len(target_patches))
        return input_patches, target_patches

    def convert_data2tf(self, input_patches, target_patches):
        input_patches = torch.Tensor(input_patches)
        target_patches = torch.Tensor(target_patches)
        return input_patches, target_patches

    def generate_tensor_files(self, height, overlap=0.3, offset=0, discard_ratio=0.5):
        self.create_dir(self.output_dir)
        for fpath in self.file_list:
            print(f"Processing {fpath}")
            fname = fpath.split(self.separator)[-1][:-3]
            self.create_dir(self.output_dir)
            input_patches, target_patches = self.generate_batches(fpath, height, overlap=overlap, offset=offset,
                                                                  discard_ratio=discard_ratio)
            tf_patches = self.convert_data2tf(input_patches, target_patches)
            tf_data_path = self.output_dir + self.separator + fname + '.pt'
            torch.save(tf_patches, tf_data_path)
            # full_data = train_dataset.concatenate(valid_dataset)
        tf_file_list = glob(self.output_dir + "*.pt")
        return tf_file_list


class DataFromDir(torch.utils.data.Dataset):
    def __init__(self, data_dir, file_extension=".pt"):
        self.data_dir = data_dir
        self.file_extension = file_extension
        self.inputs, self.outputs = self.concatenate_tensors()

    def concatenate_tensors(self):
        self.filelist = glob(self.data_dir + "*" + self.file_extension)
        print("Creating dataset with files:")
        for idx, fpath in enumerate(self.filelist):
            print(idx, fpath)
            if idx == 0:
                inputs, outputs = torch.load(fpath)
            else:
                inputs_f, outputs_f = torch.load(fpath)
                inputs = ConcatDataset([inputs, inputs_f])
                outputs = ConcatDataset([outputs, outputs_f])
        return inputs, outputs

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

    def __len__(self):
        return len(self.inputs)


class BatchesDataset(torch.utils.data.Dataset):
    def __init__(self, np_batches):
        self.tensor = torch.Tensor(np_batches)
    def __getitem__(self, idx):
        return self.tensor[idx]
    def __len__(self):
        return len(self.tensor)

class TrainFFNGenerator:
    def __init__(self, input_var_names, target_var_names, metadata_files, input_grid='l2', input_dir=None,
                 output_dir=None, file_list=None, downsample_ratio=1, random=True, records_per_file=1e9, out_file_prefix="",
                 filter_sigma=False, n_sigma=3, seed=None, reshape_order='C'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_metadata_file = metadata_files['inputs']
        self.target_metadata_file = metadata_files['targets']
        self.scat_model_var_names = {'scat': ['eastward_wind', 'northward_wind'],
                                    'model': ['eastward_model_wind', 'northward_model_wind']}
        self.input_var_names = input_var_names
        self.target_var_names = target_var_names
        self.downsample_ratio = downsample_ratio
        self.filter_sigma = filter_sigma
        self.n_sigma = n_sigma
        self.seed = seed
        self.random = random
        self.reshape_order = reshape_order
        self.records_per_file = records_per_file
        self.out_file_prefix = out_file_prefix
        self.time_var_name = 'time'
        if input_dir:
            self.file_list = glob(input_dir + '*.nc')
            self.file_list.sort()
        else:
            self.file_list = file_list
        if input_grid == "l3":
            self.nc_reader = self.read_l3_nc_data
        else:
            self.nc_reader = self.read_l2_nc_data
    def create_dir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def read_l2_nc_data(self, fpath: str, var_names: Iterable) -> np.array:
        f = Dataset(fpath)
        selected_data = []
        hg_vars = ['se_model_wind_curl', 'se_model_wind_divergence']

        for idx, var in enumerate(var_names):
            if var == "date":
                var_data = self.get_normalized_day_of_year(f)
            else:
                var_data = f.variables[var].__array__()
                var_data.data[var_data.mask] = np.nan
                if var in hg_vars:
                    restored_var = np.full(f.variables['lon'].__array__().shape, np.nan)
                    var_data_cr = (var_data[:, 1:] + var_data[:, :-1]) / 2
                    var_data_cr.mask = var_data[:, 1:].mask & var_data[:, :-1].mask
                    var_data_cr = (var_data_cr[1:, :] + var_data_cr[:-1, :]) / 2
                    var_data_cr.mask = var_data_cr[1:, :].mask & var_data_cr[:-1, :].mask
                    var_data_cr[var_data_cr.mask] = np.nan
                    restored_var[1:-1, 1:-1] = var_data_cr
                    var_data = restored_var
            #print(var, var_data.shape)
            selected_data.append(var_data)
        f.close()
        data = np.ma.stack(selected_data)
        return data

    def get_normalized_day_of_year(self, nc_dataset):
        time_var = nc_dataset.variables[self.time_var_name][:]
        time_units = nc_dataset.variables[self.time_var_name].units
        time_dates = num2date(time_var, units=time_units, only_use_cftime_datetimes=False,
                              only_use_python_datetimes=True)
        year_start = time_dates.astype('datetime64[Y]')
        dates = time_dates.astype('datetime64[D]')
        day_of_year = (dates - year_start).astype(int) + 1
        year_int = year_start.astype(int) + 1970
        is_leap = ((year_int % 4 == 0) & ((year_int % 100 != 0) | (year_int % 400 == 0)))
        days_in_year = np.where(is_leap, 366, 365)
        normalized_day_of_year = day_of_year / days_in_year.astype(float)
        normalized_day_of_year[time_var.mask] = np.nan
        return normalized_day_of_year

    def read_l3_nc_data(self, fpath: str, var_names: Iterable) -> np.array:
        f = Dataset(fpath)
        selected_data = []
        lon = f.variables['lon'].__array__()
        lat = f.variables['lat'].__array__()
        lats_mesh, lons_mesh = np.meshgrid(lat, lon, indexing='ij')
        for idx, var in enumerate(var_names):
            print(idx, var)
            if var == 'lon':
                #print("adding lon mesh")
                selected_data.append(lons_mesh)
            elif var == 'lat':
                #print("adding lat mesh")
                selected_data.append(lats_mesh)
            else:
                var_data = f.variables[var].__array__()
                mask = var_data.mask
                var_data.data[mask] = np.nan
                selected_data.append(np.squeeze(var_data, axis=0))
        f.close()
        data = np.ma.stack(selected_data)
        return data

    def calculate_target(self, fpath: str):
        scat_data = self.nc_reader(fpath, self.scat_model_var_names['scat'])
        model_data = self.nc_reader(fpath, self.scat_model_var_names['model'])
        targets = scat_data - model_data
        return targets

    def downsample(self, inputs, targets):
        total_len = targets.shape[1]
        downsampled_len = int(total_len * self.downsample_ratio)
        print(f"Randomly reducing size from {total_len} to {downsampled_len}")
        if self.random:
            if self.seed:
                np.random.seed(self.seed)
            print(f"Reducing the dataset to {self.downsample_ratio * 100}%")
            random_ids = np.random.randint(total_len, size=downsampled_len)
        else:
            random_ids = np.linspace(0, total_len - 1, downsampled_len, dtype=int)
        inputs = inputs[:, random_ids]
        targets = targets[:, random_ids]
        return inputs, targets

    def generate_norm_dataset(self, fpath):
        #Reads netcdf swath data with shape n_variables x rows x swath_width
        input_data = self.nc_reader(fpath, self.input_var_names)
        targets = self.calculate_target(fpath)
        #Reshape the data into n_variables x number of samples
        input_data = input_data.reshape(input_data.shape[0], -1, order=self.reshape_order)
        targets = targets.reshape(targets.shape[0], -1, order=self.reshape_order)
        #Filter the data that has both u and v scat winds masked at the same time
        filter = ~np.all(targets.mask, axis=0)
        #First dimention are the variable ids, filtering by second
        input_data = input_data.data[:, filter]
        targets = targets.data[:, filter]
        if self.downsample_ratio < 1:
            input_data, targets = self.downsample(input_data, targets)

        input_normalizer = Normalizer(self.input_metadata_file)
        target_normalizer = Normalizer(self.target_metadata_file)

        inputs_norm, input_vars_norm = input_normalizer.normalize(input_data, self.input_var_names)
        targets_norm, target_vars_norm = target_normalizer.normalize(targets, self.target_var_names)


        #Reshape the data so that vars are the last dimension
        inputs_norm = np.transpose(inputs_norm)
        targets_norm = np.transpose(targets_norm)

        if self.filter_sigma:
            filter = np.any(np.abs(targets_norm) <= self.n_sigma, axis=1)
            targets_norm = targets_norm[filter]
            inputs_norm = inputs_norm[filter]
        return inputs_norm, targets_norm

    def convert_data2tf(self, input_patches, target_patches):
        input_patches = torch.Tensor(input_patches)
        target_patches = torch.Tensor(target_patches)
        return input_patches, target_patches

    def generate_tensor_files(self):
        self.create_dir(self.output_dir)
        accumulated_inputs = []
        accumulated_targets = []
        out_file_counter = 0
        out_files = []
        total_files = len(self.file_list)
        i = 0
        for fpath in self.file_list:
            i+=1
            print(f"Processing {fpath}, {i} of {total_files} {i/total_files*100:.2f}%")
            fname = os.path.basename(fpath)[:-3]
            #fname = fpath.split(self.separator)[-1][:-3]
            inputs, targets = self.generate_norm_dataset(fpath)
            accumulated_inputs.append(inputs)
            accumulated_targets.append(targets)
            if len(np.vstack(accumulated_targets)) >= self.records_per_file:
                accumulated_inputs = np.vstack(accumulated_inputs)
                accumulated_targets = np.vstack(accumulated_targets)
                # Split array
                chunk_ids = np.arange(0, len(accumulated_inputs), self.records_per_file).astype(int)
                print(chunk_ids)
                inputs_arr = np.split(accumulated_inputs, chunk_ids)
                target_arr = np.split(accumulated_targets, chunk_ids)
                print(len(inputs_arr))
                # Save files
                for idx in range(len(inputs_arr)):
                    if len(inputs_arr[idx]) == self.records_per_file:
                        save_path = os.path.join(self.output_dir, f"{self.out_file_prefix}{out_file_counter:03d}.pt")
                        print(f"saving rows inputs {inputs_arr[idx].shape} {target_arr[idx].shape} in {save_path}")
                        pt_data = self.convert_data2tf(inputs_arr[idx], target_arr[idx])
                        torch.save(pt_data, save_path)
                        out_file_counter += 1
                        out_files.append(save_path)
                    else:
                        accumulated_inputs = [inputs_arr[idx]]
                        accumulated_targets = [target_arr[idx]]

        accumulated_inputs = np.vstack(accumulated_inputs)
        accumulated_targets = np.vstack(accumulated_targets)
        save_path = os.path.join(self.output_dir, f"{self.out_file_prefix}{out_file_counter:03d}.pt")
        print(f"saving rows inputs {accumulated_inputs.shape} {accumulated_targets.shape} in {save_path}")
        pt_data = self.convert_data2tf(accumulated_inputs, accumulated_targets)
        torch.save(pt_data, save_path)
        out_files.append(save_path)
        return out_files
