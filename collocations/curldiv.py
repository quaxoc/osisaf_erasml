import numpy as np
from scipy.interpolate import RegularGridInterpolator

class GradientsDxDy:
    def __init__(self, original_lats: np.ndarray, original_lons: np.ndarray,
                 target_lats: np.ndarray = None, target_lons: np.ndarray = None, extend=True):
        #self.original_var = var
        self.original_lons = original_lons
        self.original_lats = original_lats
        self.lat_1d = original_lats[:, 0]
        self.lon_1d = original_lons[0, :]
        self.extend = extend

        if (target_lons is None) or (target_lats is None):
            self.target_lats, self.target_lons = self.derived_grid()
        else:
            self.target_lats = target_lats
            self.target_lons = target_lons
        #self.dvar_dx, self.dvar_dy = self.get_gradients(var)

    def distance_np(self, origins, destinations):
        """
        Calculate the Haversine distance.

        Parameters
        ----------
        origin : tuple of np.arrays same size
            (lat, long)
        destination : tuple of np.arrays same size
            (lat, long)

        Returns
        -------
        distance_in_km : np.array

        Examples
        --------
        >>> origin = (48.1372, 11.5756)  # Munich
        >>> destination = (52.5186, 13.4083)  # Berlin
        >>> round(distance(origin, destination), 1)
        504.2
        """
        lats1, lons1 = origins
        lats2, lons2 = destinations
        radius = 6371E3  # m

        dlat = np.radians(lats2 - lats1)
        dlon = np.radians(lons2 - lons1)
        a = (np.sin(dlat / 2) * np.sin(dlat / 2) +
             np.cos(np.radians(lats1)) * np.cos(np.radians(lats2)) *
             np.sin(dlon / 2) * np.sin(dlon / 2))
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = radius * c

        return distance

    def extended_1d_lons(self, lons_1d: np.ndarray, positions: int) -> np.ndarray:
        """
        Extends the longitudes by n positions
        :param lons_1d: longitudes
        :param positions: number of grid points to extend in each direction
        :return: extended lons

        Example:
        >>> lons_1d = np.arange(0, 10, 2)
        >>> extended_1d_lons(lons_1d, 2)
        array([-4, -2,  0,  2,  4,  6,  8, 10, 12])
        """
        lons_extended_1d = lons_1d.copy()
        lon_delta = lons_1d[1] - lons_1d[0]
        b = np.arange(lons_1d[0] - lon_delta * positions, lons_1d[0], lon_delta)
        e = (np.arange(positions) + 1) * lon_delta + lons_1d[-1]
        lons_extended_1d = np.concatenate((b, lons_extended_1d, e))
        return lons_extended_1d

    def extend_nwp_var(self, nwp_var: np.ndarray, positions: int) -> np.ndarray:
        """
        In case of global variables, copies several columns to the left and right side of the grid to be able to
        calculate the gradients where the grid was cut (at 0 lon for example)
        Args:
            nwp_var: 2d array of the variable to extend
            positions: how many extra columns will be added to each side

        Returns:
            2d array of extended variable

        """
        nwp_var_extended = np.zeros([nwp_var.shape[0], nwp_var.shape[1] + positions * 2])
        nwp_var_extended[:, :positions] = nwp_var[:, -positions:]
        nwp_var_extended[:, positions:-positions] = nwp_var
        nwp_var_extended[:, -positions:] = nwp_var[:, :positions]
        return nwp_var_extended

    def interpolate_var_reg2reg(self, orig_var: np.ndarray, orig_lat_1d: np.ndarray, orig_lon_1d: np.ndarray,
                                target_lats_mesh: np.ndarray, target_lons_mesh: np.ndarray,
                                convert_180_360: bool = False, method: str = 'linear',
                                extend_positions: int = 2) -> np.ndarray:
        """
        Interpolates variable from one regular grid to another
        :param orig_var: Variable that will be interpolated in 2d numpy array
        :param orig_lat_1d: Source latitudes in 1d array
        :param orig_lon_1d: Source longitudes in 1d array
        :param target_lats_mesh: 2-D mesh of target latitudes
        :param target_lons_mesh: 2-D mesh of target longitudes
        :param convert_180_360: If original grid is in -180 180 format if it should be converted to 0 360
        :param method: Interpolation method, linear by default, see available methods for scipy.interpolate.RegularGridInterpolator
        :param extend_positions: Number of positions to extend left and right to avoid artifacts on borders
        :return:  Interpolated variable
        """
        if isinstance(orig_var, np.ma.MaskedArray):
            mask = orig_var.mask
            orig_var = orig_var.data
            orig_var[mask] = np.nan

        if convert_180_360:
            lons = np.where(orig_lon_1d < 0, orig_lon_1d + 360, orig_lon_1d)
            roll_id = np.argmin(lons)
            lons = np.roll(lons, roll_id)
            orig_var = np.roll(orig_var, roll_id, axis=1)
        else:
            lons = orig_lon_1d

        if self.extend:
            lons = self.extended_1d_lons(lons, extend_positions)
            orig_var = self.extend_nwp_var(orig_var, extend_positions)

        # Check if original lats are in ascending order, requirement of RegularGridInterpolator
        if orig_lat_1d[0] > orig_lat_1d[-1]:
            var_data = np.flip(orig_var, axis=0)
            lats = np.flip(orig_lat_1d)
        else:
            lats = orig_lat_1d
            var_data = orig_var

        interp = RegularGridInterpolator((lats, lons), var_data,
                                         method=method, bounds_error=False)
        interp_data = interp((target_lats_mesh, target_lons_mesh))
        return interp_data

    def get_gradients(self, var_data):
        # Calculate the dy gradients depending on if latitudes are in ascending or descending order
        if self.lat_1d[0] < self.lat_1d[-1]:
            sign = 1
        else:
            sign = -1

        if self.extend:
            lons_1d_ext = self.extended_1d_lons(self.lon_1d, 2)
            var_ext = self.extend_nwp_var(var_data, 2)
        else:
            lons_1d_ext = self.lon_1d
            var_ext = var_data
        lons, lats = np.meshgrid(self.lon_1d, self.lat_1d)
        lons_ext, lats_ext = np.meshgrid(lons_1d_ext, self.lat_1d)

        distance_x = self.distance_np((lats_ext[:, 1:], lons_ext[:, 1:]), (lats_ext[:, 1:], lons_ext[:, :-1]))
        distance_x[distance_x < 1] = np.nan

        distance_y = self.distance_np((lats[1:, :], lons[1:, :]), (lats[:-1, :], lons[1:, :]))
        distance_y[distance_y < 1] = np.nan
        dvar_dx = (var_ext[:, 1:] - var_ext[:, :-1]) / distance_x
        lons_1d_dx = (lons_1d_ext[1:] + lons_1d_ext[:-1]) / 2
        dvar_dy = sign * (var_data[1:, :] - var_data[:-1, :]) / distance_y
        lats_1d_dy = (self.lat_1d[1:] + self.lat_1d[:-1]) / 2


        dvar_dx_target_grid = self.interpolate_var_reg2reg(dvar_dx, self.lat_1d, lons_1d_dx, self.target_lats,
                                                           self.target_lons)
        dvar_dy_target_grid = self.interpolate_var_reg2reg(dvar_dy, lats_1d_dy, self.lon_1d, self.target_lats,
                                                           self.target_lons)



        return dvar_dx_target_grid, dvar_dy_target_grid

    def derived_grid(self):
        """
        If the target grid is not defined returns the half grid resulting from the input grid
        :return:
        """
        derived_lats = (self.original_lats[1:, :] + self.original_lats[:-1, :]) / 2
        derived_lats = derived_lats[:, :-1]
        derived_lons = (self.original_lons[:, 1:] + self.original_lons[:, :-1]) / 2
        derived_lons = derived_lons[:-1, :]
        return derived_lats, derived_lons


class CurlDivRegularGrid(GradientsDxDy):
    def __init__(self, u: np.ndarray, v: np.ndarray, original_lats: np.ndarray, original_lons: np.ndarray,
                 target_lats: np.ndarray = None, target_lons: np.ndarray = None, extend=True):
        """
        Class that calculated curl and divergence on L3 grid and interpolates them to the target grid
        :param u: Meridional component in 2d np.array
        :param v: Zonal component in 2d np.array
        :param original_lats: Latitudes in 2d np.array
        :param original_lons: Longitudes in 2d np.array
        :param target_lats:  Target lat grid in 2d np.array
        :param target_lons: Target lon grid in 2d np.array
        :param extend: If the fields are extended to avoid artifacts at border longitude (0/360) or (-180/180)

        Example of use:
        >>> target_lon_grid = np.arange(0.0625, 360, .125)
        >>> target_lat_grid = np.arange(-90+0.0625, 90, .125)
        >>> lon_mesh, lat_mesh = np.meshgrid(target_lon_grid, target_lat_grid)
        >>> [u_era5, v_era5] = read_nwp_vars(grib_fn, ["u10n", "v10n"])
        >>> nwp_lats, nwp_lons = get_nwp_lat_lons(grib_fn)
        >>> derivatives = CurlDivRegularGrid(u_era5,  v_era5, nwp_lats, nwp_lons, lat_mesh, lon_mesh)
        >>> curl = derivatives.get_curl()
        >>> divergence = derivatives.get_divergence()
        """

        self.u = u
        self.v = v
        self.original_lons = original_lons
        self.original_lats = original_lats
        self.lat_1d = original_lats[:, 0]
        self.lon_1d = original_lons[0, :]
        self.extend = extend

        if (target_lons is None) or (target_lats is None):
            self.target_lats, self.target_lons = self.derived_grid()
        else:
            self.target_lats = target_lats
            self.target_lons = target_lons
        self.du_dx, self.du_dy = self.get_gradients(u)
        self.dv_dx, self.dv_dy = self.get_gradients(v)
        self.curl, self.divergence = self.get_curl_div()

    def get_curl(self):
        return self.curl

    def get_divergence(self):
        return self.divergence

    def get_curl_div(self):
        divergence = self.du_dx + self.dv_dy
        curl = self.dv_dx - self.du_dy
        return curl, divergence


