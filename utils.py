"""
Utility functions for WRF-CFS data processing.
"""

import os
import logging
import yaml
from pathlib import Path
from datetime import datetime
import numpy as np
import xarray as xr
import xesmf as xe
import fcntl
try:
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str or Path
        Path to the configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_dir, console=True, file_logging=True):
    """
    Set up logging configuration.
    Only keeps the latest log file, removes previous ones.
    
    Parameters
    ----------
    log_dir : str or Path
        Directory for log files
    console : bool, optional
        Enable console logging (default: True)
    file_logging : bool, optional
        Enable file logging (default: True)
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('WRF_CFS_Processing')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_logging:
        # Remove all previous log files
        for old_log in log_path.glob('process_wrf_cfs_*.log'):
            try:
                old_log.unlink()
            except Exception:
                pass  # Ignore errors when removing old logs
        
        # Create new log file with fixed name
        log_file = log_path / 'wrf_cfs_processing.log'
        file_handler = logging.FileHandler(log_file, mode='w')  # Overwrite mode
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Log file created: {log_file}")
    
    return logger


def get_file_list(base_path, year, months, file_pattern="wrfoutPL_d01.*.nc"):
    """
    Get list of WRF output files for specified year and months.
    
    Parameters
    ----------
    base_path : str or Path
        Base directory containing year folders
    year : int
        Year to process
    months : list of int
        List of months to include
    file_pattern : str, optional
        File pattern to match (default: "wrfoutPL_d01.*.nc")
        
    Returns
    -------
    list of Path
        List of file paths
    """
    year_dir = Path(base_path) / f"{year}0301-CFS"
    
    if not year_dir.exists():
        raise FileNotFoundError(f"Year directory not found: {year_dir}")
    
    files = []
    for nc_file in sorted(year_dir.glob(file_pattern)):
        # Extract date from filename
        # Pattern 1: wrfoutPL_d01.YYYY-MM-DD_HH:MM:SS.nc
        # Pattern 2: wrfout_d01_YYYY-MM-DD_HH:MM:SS
        filename = nc_file.name
        
        if filename.startswith("wrfoutPL_d01"):
            # wrfoutPL_d01.YYYY-MM-DD_HH:MM:SS.nc
            date_part = filename.split('.')[-2]  # YYYY-MM-DD_HH:MM:SS
            date_str = date_part.split('_')[0]  # YYYY-MM-DD
        elif filename.startswith("wrfout_d01"):
            # wrfout_d01_YYYY-MM-DD_HH:MM:SS (no .nc extension)
            parts = filename.split('_')
            if len(parts) >= 3:
                # parts[2] is already "YYYY-MM-DD"
                date_str = parts[2]
            else:
                continue
        else:
            continue
        
        try:
            month = int(date_str.split('-')[1])
            if month in months:
                files.append(nc_file)
        except (ValueError, IndexError):
            continue
    
    return files


def extract_china_region(ds, lat_bounds, lon_bounds):
    """
    Extract China region from dataset with curvilinear coordinates.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with XLAT and XLONG variables
    lat_bounds : tuple
        (lat_min, lat_max)
    lon_bounds : tuple
        (lon_min, lon_max)
        
    Returns
    -------
    xarray.Dataset
        Subset dataset
    """
    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds
    
    # For curvilinear coordinates, we need to find grid indices
    # Get lat/lon arrays (use first time step if time dimension exists)
    if 'Time' in ds.XLAT.dims:
        lat = ds.XLAT.isel(Time=0).values
        lon = ds.XLONG.isel(Time=0).values
    else:
        lat = ds.XLAT.values
        lon = ds.XLONG.values
    
    # Create mask for region of interest
    mask = (
        (lat >= lat_min) & (lat <= lat_max) &
        (lon >= lon_min) & (lon <= lon_max)
    )
    
    # Find bounding box in grid coordinates
    rows, cols = np.where(mask)
    if len(rows) == 0:
        raise ValueError("No grid points found in specified region")
    
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()
    
    # Extract subset with some padding to ensure full coverage
    pad = 2
    row_min = max(0, row_min - pad)
    row_max = min(lat.shape[0] - 1, row_max + pad)
    col_min = max(0, col_min - pad)
    col_max = min(lat.shape[1] - 1, col_max + pad)
    
    # Subset the dataset
    ds_subset = ds.isel(
        south_north=slice(row_min, row_max + 1),
        west_east=slice(col_min, col_max + 1)
    )
    
    return ds_subset


def get_valid_latlon_bounds(ds):
    """从数据集中提取有效的经纬度范围"""
    if 'Time' in ds.XLAT.dims:
        lat = ds.XLAT.isel(Time=0).values
        lon = ds.XLONG.isel(Time=0).values
    else:
        lat = ds.XLAT.values
        lon = ds.XLONG.values
    
    return {
        'lat_min': float(lat.min()),
        'lat_max': float(lat.max()),
        'lon_min': float(lon.min()),
        'lon_max': float(lon.max())
    }


def create_target_grid(lat_range, lon_range, resolution):
    """
    Create target regular lat-lon grid.
    
    Parameters
    ----------
    lat_range : tuple
        (lat_min, lat_max)
    lon_range : tuple
        (lon_min, lon_max)
    resolution : float
        Grid resolution in degrees
        
    Returns
    -------
    xarray.Dataset
        Target grid dataset
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    
    # Explicitly use float64 to ensure coordinate precision
    lat = np.arange(lat_min, lat_max + resolution, resolution, dtype=np.float64)
    lon = np.arange(lon_min, lon_max + resolution, resolution, dtype=np.float64)
    
    # Create 2D grid
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    
    # Create xarray dataset with dimension coordinates
    grid = xr.Dataset({
        'lat': (['y', 'x'], lat_2d),
        'lon': (['y', 'x'], lon_2d)
    })
    # CF attrs for coordinates (help visualization tools like Panoply)
    grid['lat'].attrs.update({
        'standard_name': 'latitude',
        'long_name': 'latitude',
        'units': 'degrees_north'
    })
    grid['lon'].attrs.update({
        'standard_name': 'longitude',
        'long_name': 'longitude',
        'units': 'degrees_east'
    })
    
    # Add dimension coordinates
    grid = grid.assign_coords({
        'y': np.arange(len(lat)),
        'x': np.arange(len(lon))
    })
    
    return grid


def interpolate_to_regular_grid(ds, target_grid, method='scipy', reuse_weights=False, weights_file=None):
    """
    Interpolate dataset to regular grid using fast scipy interpolation.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Source dataset with curvilinear coordinates
    target_grid : xarray.Dataset
        Target regular grid
    method : str, optional
        Interpolation method ('scipy' for fast scipy interpolation)
    reuse_weights : bool, optional
        Whether to reuse existing weights (not used for scipy)
    weights_file : str, optional
        Path to weights file (not used for scipy)
    
    Returns
    -------
    xarray.Dataset
        Interpolated dataset
    """
    import logging
    logger = logging.getLogger('WRF_CFS_Processing')
    
    # Prepare source grid
    # Extract lat/lon (use first time if time dimension exists)
    if 'Time' in ds.XLAT.dims:
        source_lat = ds.XLAT.isel(Time=0).values
        source_lon = ds.XLONG.isel(Time=0).values
    else:
        source_lat = ds.XLAT.values
        source_lon = ds.XLONG.values
    
    # Basic validation
    if np.any(np.isnan(source_lat)) or np.any(np.isnan(source_lon)):
        raise ValueError("Source grid contains NaN values in lat/lon coordinates")
    
    # Validate target grid
    target_lat = target_grid['lat'].values
    target_lon = target_grid['lon'].values
    if np.any(np.isnan(target_lat)) or np.any(np.isnan(target_lon)):
        raise ValueError("Target grid contains NaN values in lat/lon coordinates")
    
    # Use optimized interpolation methods
    logger.info("Using optimized interpolation for better performance...")
    
    # Create output dataset
    ds_interp = xr.Dataset()
    
    # Copy Times variable if it exists
    if 'Times' in ds:
        ds_interp['Times'] = ds['Times']
    
    # Copy Time coordinate if it exists
    if 'Time' in ds.coords:
        ds_interp = ds_interp.assign_coords({'Time': ds.coords['Time']})
    
    # Prepare source and target coordinates
    source_points = np.column_stack([source_lon.ravel(), source_lat.ravel()])
    target_lon_2d, target_lat_2d = np.meshgrid(
        target_grid['lon'].values[0, :],
        target_grid['lat'].values[:, 0]
    )
    target_points = np.column_stack([target_lon_2d.ravel(), target_lat_2d.ravel()])
    
    # Interpolate each variable
    for var in ds.data_vars:
        if var in ['XLAT', 'XLONG', 'Times', 'lat', 'lon', 'latitude', 'longitude', 'crs']:
            continue
        
        var_data = ds[var]
        logger.debug(f"Interpolating {var} using scipy...")
        
        # Handle time dimension if present
        if 'Time' in var_data.dims:
            # Optimized vectorized interpolation using xarray's built-in methods
            logger.debug(f"Using xarray interpolation for {len(var_data.Time)} time steps...")

            try:
                # Method 1: xarray interp on renamed dims (fast & vectorized)
                ds_for_interp = ds.rename({
                    k: v for k, v in {
                        'south_north': 'lat',
                        'west_east': 'lon'
                    }.items() if k in ds.dims
                })

                ds_for_interp = ds_for_interp.assign_coords(
                    lat=(('lat', 'lon'), source_lat),
                    lon=(('lat', 'lon'), source_lon)
                )

                var_da = ds_for_interp[var]

                # 1D target vectors for interp
                target_lat_vec = target_grid['lat'].isel(x=0).astype(np.float64)
                target_lon_vec = target_grid['lon'].isel(y=0).astype(np.float64)

                var_interp = var_da.interp(
                    lat=target_lat_vec,
                    lon=target_lon_vec,
                    method='linear',
                    kwargs={'bounds_error': False, 'fill_value': np.nan}
                )

                # Ensure coords are 1D lat/lon only
                var_interp = var_interp.assign_coords(lat=target_lat_vec, lon=target_lon_vec)
                var_interp = var_interp.drop_vars([c for c in var_interp.coords if c not in ['latitude', 'longitude', 'time'] and c not in var_interp.dims], errors='ignore')

                ds_interp[var] = var_interp
                logger.debug(f"Successfully used xarray interpolation for {var} with dims {var_interp.dims}")

            except Exception as e:
                logger.warning(f"xarray interp failed ({e}), trying nearest neighbor approximation...")

                try:
                    # Method 2: Fast nearest neighbor approximation for large datasets
                    from scipy.spatial import cKDTree

                    # Build KDTree for fast nearest neighbor search
                    tree = cKDTree(source_points)

                    # Find nearest neighbors for all target points at once
                    distances, indices = tree.query(target_points, k=1)

                    # Get source values shape for reshaping
                    source_shape = source_lat.shape

                    # Initialize output array
                    interp_data = np.full((len(var_data.Time), target_points.shape[0]), np.nan)

                    # Process in time batches
                    batch_size = min(1000, len(var_data.Time))  # Larger batches for nearest neighbor
                    for batch_start in range(0, len(var_data.Time), batch_size):
                        batch_end = min(batch_start + batch_size, len(var_data.Time))
                        batch_data = var_data.values[batch_start:batch_end]

                        # Vectorized nearest neighbor lookup
                        for i in range(len(indices)):
                            idx = indices[i]
                            row, col = np.unravel_index(idx, source_shape)
                            interp_data[batch_start:batch_end, i] = batch_data[:, row, col]

                    interp_data = interp_data.reshape(len(var_data.Time), target_lat_2d.shape[0], target_lat_2d.shape[1])
                    ds_interp[var] = (['Time', 'latitude', 'longitude'], interp_data)
                    logger.debug(f"Successfully used nearest neighbor approximation for {var}")

                except Exception as e2:
                    # Method 3: Fallback to original griddata method
                    logger.warning(f"Nearest neighbor failed ({e2}), falling back to griddata...")

            except Exception as e:
                # Fallback to original method if xarray interp fails
                logger.warning(f"xarray interp failed ({e}), falling back to griddata...")

                # Get all data at once: shape (time, lat, lon) -> (time, lat*lon)
                all_source_values = var_data.values.reshape(len(var_data.Time), -1)

                # Initialize output array
                interp_data = np.full((len(var_data.Time), target_points.shape[0]), np.nan)

                # Process in larger batches for better performance
                batch_size = min(500, len(var_data.Time))  # Process 500 time steps at a time
                for batch_start in range(0, len(var_data.Time), batch_size):
                    batch_end = min(batch_start + batch_size, len(var_data.Time))
                    logger.debug(f"Processing time steps {batch_start}:{batch_end}...")

                    # Vectorized processing for the batch
                    batch_data = all_source_values[batch_start:batch_end]
                    for i, t_idx in enumerate(range(batch_start, batch_end)):
                        source_values = batch_data[i]
                        valid_mask = ~np.isnan(source_values)
                        if np.any(valid_mask):
                            interp_data[t_idx] = griddata(
                                source_points[valid_mask],
                                source_values[valid_mask],
                                target_points,
                                method='linear',
                                fill_value=np.nan
                            )

                # Reshape to final dimensions
                interp_data = interp_data.reshape(len(var_data.Time), target_lat_2d.shape[0], target_lat_2d.shape[1])
                ds_interp[var] = (['Time', 'latitude', 'longitude'], interp_data)
        else:
            # No time dimension
            source_values = var_data.values.ravel()
            valid_mask = ~np.isnan(source_values)
            if np.any(valid_mask):
                interp_values = griddata(
                    source_points[valid_mask],
                    source_values[valid_mask],
                    target_points,
                    method='linear',
                    fill_value=np.nan
                )
            else:
                interp_values = np.full(target_points.shape[0], np.nan)
            
            ds_interp[var] = (['latitude', 'longitude'], interp_values.reshape(target_lat_2d.shape))
    
    # Add coordinates (ensure float64 dtype for precision)
    # Use CF-compliant names: latitude and longitude
    lat_coord = target_grid['lat'].isel(x=0).astype(np.float64)
    lon_coord = target_grid['lon'].isel(y=0).astype(np.float64)
    ds_interp = ds_interp.assign_coords({
        'latitude': lat_coord,
        'longitude': lon_coord
    })
    
    # Add coordinate attributes
    ds_interp['latitude'].attrs.update({
        'standard_name': 'latitude',
        'long_name': 'latitude',
        'units': 'degrees_north'
    })
    ds_interp['longitude'].attrs.update({
        'standard_name': 'longitude',
        'long_name': 'longitude',
        'units': 'degrees_east'
    })
    
    return ds_interp


def compute_seasonal_mean(ds, months):
    """
    Compute seasonal mean for specified months.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with time dimension
    months : list of int
        List of months to include in seasonal mean
        
    Returns
    -------
    xarray.Dataset
        Dataset with seasonal mean
    """
    import pandas as pd
    
    # WRF files have Times variable with actual datetime strings
    if 'Times' in ds:
        # Load Times variable into memory first to avoid file handle issues
        # This is crucial when using Dask with delayed computations
        times_data = ds.Times.load().values
        
        # Convert Times (bytes array) to datetime
        time_strings = []
        for t in times_data:
            if isinstance(t, bytes):
                time_str = t.decode('utf-8')
            elif isinstance(t, np.ndarray):
                time_str = ''.join([c.decode('utf-8') if isinstance(c, bytes) else c for c in t])
            else:
                time_str = str(t)
            
            # WRF format: YYYY-MM-DD_HH:MM:SS, replace underscore with space for pandas
            time_str = time_str.replace('_', ' ')
            time_strings.append(time_str)
        
        time_pd = pd.to_datetime(time_strings)
        
        # Replace Time coordinate with proper datetime
        ds = ds.assign_coords({'Time': time_pd})
    elif 'Time' in ds.coords:
        # Try to convert existing Time coordinate
        time_vals = ds.Time.values
        if isinstance(time_vals[0], (pd.Timestamp, np.datetime64)):
            time_pd = pd.DatetimeIndex(time_vals)
            ds = ds.assign_coords({'Time': time_pd})
        else:
            raise ValueError("Cannot determine time information from dataset")
    else:
        raise ValueError("No time information found in dataset")
    
    # Filter for specified months
    time_month = ds.Time.dt.month
    mask = time_month.isin(months)
    ds_season = ds.sel(Time=mask)
    
    if len(ds_season.Time) == 0:
        raise ValueError(f"No data found for months {months}")
    
    # Compute mean
    ds_mean = ds_season.mean(dim='Time', keep_attrs=True)
    
    # Update attributes
    season_name = 'JJA' if months == [6, 7, 8] else f"months_{'-'.join(map(str, months))}"
    ds_mean.attrs['seasonal_mean'] = season_name
    ds_mean.attrs['months_included'] = str(months)
    ds_mean.attrs['n_time_steps'] = len(ds_season.Time)
    
    return ds_mean


def accumulate_yearly_data(existing_data, new_data, year, variable_name):
    """
    Accumulate yearly processed data into a multi-year dataset.
    
    Parameters
    ----------
    existing_data : xarray.Dataset or None
        Existing dataset with accumulated data (None for first year)
    new_data : xarray.Dataset
        New yearly data to add
    year : int
        Year of the new data
    variable_name : str
        Name of the variable being processed
        
    Returns
    -------
    xarray.Dataset
        Dataset with accumulated data including the new year
    """
    # Ensure new_data is loaded into memory
    if hasattr(new_data, 'load'):
        new_data = new_data.load()
    
    # Add year as a coordinate to the new data
    new_data = new_data.expand_dims('year')
    new_data = new_data.assign_coords(year=[year])
    
    if existing_data is None:
        return new_data
    else:
        # Ensure existing_data is also loaded
        if hasattr(existing_data, 'load'):
            existing_data = existing_data.load()
        
        # Concatenate along year dimension
        combined = xr.concat([existing_data, new_data], dim='year')
        
        # Load the combined result to ensure it's in memory
        combined = combined.load()
        
        return combined


def save_processed_data(ds, output_path, pressure_level, year, variable_prefix='ght', compression=True, complevel=4):
    """
    Save processed data to NetCDF file with proper naming and metadata.
    (Legacy function for single-year saving - deprecated, use save_variable_multi_year instead)
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to save
    output_path : str or Path
        Output directory
    pressure_level : int
        Pressure level in hPa
    year : int
        Year of data
    variable_prefix : str, optional
        Prefix for variable names (default: 'ght')
    compression : bool, optional
        Enable compression (default: True)
    complevel : int, optional
        Compression level (default: 4)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create output filename using variable_prefix instead of hardcoded "GHT_PL"
    filename = f"{variable_prefix}_{pressure_level}hPa_JJA_{year}.nc"
    filepath = output_path / filename
    
    # Prepare dataset for saving
    ds_save = ds.copy()
    
    # Rename variable to standardized name
    var_name = f"{variable_prefix}_{pressure_level}"
    
    # Find and rename the data variable (support multiple variable types)
    data_vars = list(ds_save.data_vars)
    if len(data_vars) == 1:
        # If there's only one data variable, rename it
        old_var_name = data_vars[0]
        if old_var_name != var_name:
            ds_save = ds_save.rename({old_var_name: var_name})
    elif len(data_vars) > 1:
        # If multiple variables, try to find the main one based on common patterns
        for pattern in ['GHT_PL', 'U_PL', 'V_PL', 'T_PL']:
            if pattern in data_vars:
                ds_save = ds_save.rename({pattern: var_name})
                # Drop other data variables
                for v in data_vars:
                    if v != pattern:
                        ds_save = ds_save.drop_vars(v)
                break
    
    # Add variable attributes based on variable type
    if variable_prefix == 'ght':
        long_name = f'Geopotential Height at {pressure_level} hPa'
        standard_name = 'geopotential_height'
        units = 'm'
    elif variable_prefix == 'u':
        long_name = f'Zonal Wind Component at {pressure_level} hPa'
        standard_name = 'eastward_wind'
        units = 'm s-1'
    elif variable_prefix == 'v':
        long_name = f'Meridional Wind Component at {pressure_level} hPa'
        standard_name = 'northward_wind'
        units = 'm s-1'
    else:
        # Generic attributes
        long_name = f'{variable_prefix.upper()} at {pressure_level} hPa'
        standard_name = variable_prefix
        units = ds_save[var_name].attrs.get('units', 'unknown')
    
    ds_save[var_name].attrs.update({
        'long_name': long_name,
        'standard_name': standard_name,
        'units': units,
        'pressure_level': f'{pressure_level} hPa',
        'season': 'JJA',
        'year': year
    })
    
    # Ensure lat/lon coordinate attributes are present for CF compliance
    if 'lat' in ds_save.coords:
        ds_save['lat'].attrs.update({
            'standard_name': 'latitude',
            'long_name': 'latitude',
            'units': 'degrees_north'
        })
    if 'lon' in ds_save.coords:
        ds_save['lon'].attrs.update({
            'standard_name': 'longitude',
            'long_name': 'longitude',
            'units': 'degrees_east'
        })

    # Add global attributes
    var_title = variable_prefix.upper()
    if variable_prefix == 'ght':
        var_title = 'Geopotential Height'
    elif variable_prefix == 'u':
        var_title = 'Zonal Wind'
    elif variable_prefix == 'v':
        var_title = 'Meridional Wind'
    
    ds_save.attrs.update({
        'title': f'WRF-CFS {var_title} {pressure_level}hPa JJA Mean {year}',
        'source': 'WRF V4.6.0 Model with CFS forcing',
        'institution': 'Seasonal Forecast',
        'history': f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        'processing': 'Interpolated to regular grid and averaged over JJA season',
        'conventions': 'CF-1.8'
    })
    
    # Encoding for compression
    encoding = {}
    if compression:
        for var in ds_save.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': complevel,
                'dtype': 'float32'
            }
    
    # Save to NetCDF
    ds_save.to_netcdf(filepath, encoding=encoding, format='NETCDF4')
    
    return filepath


def save_variable_multi_year(ds, output_path, variable_name, pressure_level=None, compression=True, complevel=4):
    """
    Save multi-year processed data to NetCDF file.
    Each variable gets one file with all years along time dimension.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with year dimension containing all years
    output_path : str or Path
        Output directory
    variable_name : str
        Output variable name (e.g., 'ght', 'u', 'v', 'rain')
    pressure_level : int or None, optional
        Pressure level in hPa (None for 2D variables like precipitation)
    compression : bool, optional
        Enable compression (default: True)
    complevel : int, optional
        Compression level (default: 4)
        
    Returns
    -------
    Path
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create output filename
    if pressure_level is not None:
        filename = f"{variable_name}_{pressure_level}hPa_JJA.nc"
    else:
        filename = f"{variable_name}_JJA.nc"
    filepath = output_path / filename
    
    # Prepare dataset for saving
    ds_save = ds.copy()
    
    # Rename year dimension to time for CF compliance
    import pandas as pd
    import numpy as np
    if 'year' in ds_save.dims:
        years = ds_save.year.values
        # Convert numpy array to list for safe boolean checking
        if isinstance(years, np.ndarray):
            years = years.tolist()
        else:
            years = list(years)
        ds_save = ds_save.rename({'year': 'time'})
        # Convert year coordinate to time coordinate with proper datetime
        time_coords = pd.to_datetime([f"{int(y)}-07-15" for y in years])  # Mid-season date
        ds_save = ds_save.assign_coords(time=time_coords)
    else:
        # If already has time dimension, extract years from it
        if 'time' in ds_save.coords:
            time_vals = ds_save.time.values
            if len(time_vals) > 0 and isinstance(time_vals[0], pd.Timestamp):
                years = [t.year for t in time_vals]
            elif len(time_vals) > 0:
                years = time_vals
                # Convert numpy array to list if needed
                if isinstance(years, np.ndarray):
                    years = years.tolist()
                else:
                    years = list(years)
            else:
                years = []
        else:
            years = []
    
    # Get the data variable name (should be only one)
    data_vars = list(ds_save.data_vars)
    if len(data_vars) != 1:
        raise ValueError(f"Expected exactly one data variable, found {len(data_vars)}")
    
    var_name = data_vars[0]
    
    # Get existing attributes from processed data (preserved from compute_seasonal_mean)
    existing_attrs = ds_save[var_name].attrs.copy()
    existing_description = existing_attrs.get('description', '')
    existing_long_name = existing_attrs.get('long_name', var_name)
    original_units = existing_attrs.get('units', None) or existing_attrs.get('original_units', None)
    processing_method = existing_attrs.get('processing_method', 'unknown')
    
    # Update variable attributes based on variable type
    # Preserve and enhance descriptions from compute_seasonal_mean
    if variable_name == 'ght':
        # WRF GHT_PL units: typically 'm' or 'gpm' (geopotential meters, 1 gpm ≈ 1 m)
        units = original_units if original_units in ['m', 'gpm', 'meter', 'meters'] else 'm'
        if pressure_level is not None:
            long_name = f'Geopotential Height at {pressure_level} hPa'
            description = (
                f"Geopotential height at {pressure_level} hPa pressure level. "
                f"{existing_description if existing_description else 'Instantaneous geopotential height from WRF model, averaged over JJA season.'} "
                f"Geopotential height represents the height of a pressure surface above sea level. "
                f"Original WRF variable: GHT_PL. Units: {units}."
            )
            ds_save[var_name].attrs.update({
                'long_name': long_name,
                'standard_name': 'geopotential_height',
                'description': description,
                'units': units,
                'pressure_level': f'{pressure_level} hPa',
                'original_wrf_variable': 'GHT_PL'
            })
        else:
            description = (
                f"Geopotential height. "
                f"{existing_description if existing_description else 'Instantaneous geopotential height from WRF model, averaged over JJA season.'} "
                f"Original WRF variable: GHT_PL. Units: {units}."
            )
            ds_save[var_name].attrs.update({
                'long_name': 'Geopotential Height',
                'standard_name': 'geopotential_height',
                'description': description,
                'units': units,
                'original_wrf_variable': 'GHT_PL'
            })
    elif variable_name == 'u':
        # WRF U_PL units: typically 'm s-1'
        units = original_units if original_units and 'm' in original_units and 's' in original_units else 'm s-1'
        if pressure_level is not None:
            long_name = f'Zonal Wind Component at {pressure_level} hPa'
            description = (
                f"Zonal (eastward) wind component at {pressure_level} hPa pressure level. "
                f"{existing_description if existing_description else 'Instantaneous zonal wind from WRF model, averaged over JJA season.'} "
                f"Positive values indicate eastward (westerly) wind. "
                f"Original WRF variable: U_PL. Units: {units}."
            )
            ds_save[var_name].attrs.update({
                'long_name': long_name,
                'standard_name': 'eastward_wind',
                'description': description,
                'units': units,
                'pressure_level': f'{pressure_level} hPa',
                'original_wrf_variable': 'U_PL'
            })
        else:
            description = (
                f"Zonal (eastward) wind component. "
                f"{existing_description if existing_description else 'Instantaneous zonal wind from WRF model, averaged over JJA season.'} "
                f"Positive values indicate eastward (westerly) wind. "
                f"Original WRF variable: U_PL. Units: {units}."
            )
            ds_save[var_name].attrs.update({
                'long_name': 'Zonal Wind Component',
                'standard_name': 'eastward_wind',
                'description': description,
                'units': units,
                'original_wrf_variable': 'U_PL'
            })
    elif variable_name == 'v':
        # WRF V_PL units: typically 'm s-1'
        units = original_units if original_units and 'm' in original_units and 's' in original_units else 'm s-1'
        if pressure_level is not None:
            long_name = f'Meridional Wind Component at {pressure_level} hPa'
            description = (
                f"Meridional (northward) wind component at {pressure_level} hPa pressure level. "
                f"{existing_description if existing_description else 'Instantaneous meridional wind from WRF model, averaged over JJA season.'} "
                f"Positive values indicate northward wind. "
                f"Original WRF variable: V_PL. Units: {units}."
            )
            ds_save[var_name].attrs.update({
                'long_name': long_name,
                'standard_name': 'northward_wind',
                'description': description,
                'units': units,
                'pressure_level': f'{pressure_level} hPa',
                'original_wrf_variable': 'V_PL'
            })
        else:
            description = (
                f"Meridional (northward) wind component. "
                f"{existing_description if existing_description else 'Instantaneous meridional wind from WRF model, averaged over JJA season.'} "
                f"Positive values indicate northward wind. "
                f"Original WRF variable: V_PL. Units: {units}."
            )
            ds_save[var_name].attrs.update({
                'long_name': 'Meridional Wind Component',
                'standard_name': 'northward_wind',
                'description': description,
                'units': units,
                'original_wrf_variable': 'V_PL'
            })
    elif variable_name == 'rain':
        # WRF RAINNC/RAINC units: typically 'mm' (accumulated precipitation)
        # Original WRF variables are ACCUMULATED (from model start)
        units = original_units if original_units and 'mm' in original_units.lower() else 'mm'
        description = (
            f"Precipitation averaged over the JJA season. "
            f"Original WRF variables: RAINNC (ACCUMULATED TOTAL GRID SCALE PRECIPITATION) + "
            f"RAINC (ACCUMULATED TOTAL CUMULUS PRECIPITATION) + "
            f"RAINSH (ACCUMULATED SHALLOW CUMULUS PRECIPITATION, if available). "
            f"These are accumulated precipitation variables in WRF, measured from model start. "
            f"Processing method: Time average of accumulated values over JJA season. "
            f"Note: This represents the time-averaged accumulated precipitation values, "
            f"not the total accumulated precipitation over the season. "
            f"The units ({units}) refer to the average of accumulated values at each time step. "
            f"For total seasonal precipitation, this value would need to be multiplied by the number of time steps, "
            f"or the difference method (final - initial) should be used instead."
        )
        ds_save[var_name].attrs.update({
            'long_name': 'Precipitation (Time-Averaged Accumulated)',
            'standard_name': 'precipitation_amount',
            'description': description,
            'units': units,
            'original_wrf_variables': 'RAINNC + RAINC + RAINSH',
            'processing_note': 'Time average of accumulated values, not total accumulation'
        })
        # Rain doesn't have pressure level
    else:
        # Generic variable attributes
        if pressure_level is not None:
            ds_save[var_name].attrs.update({
                'long_name': f'{variable_name.upper()} at {pressure_level} hPa',
                'units': ds_save[var_name].attrs.get('units', 'unknown'),
                'pressure_level': f'{pressure_level} hPa'
            })
        else:
            ds_save[var_name].attrs.update({
                'long_name': variable_name.upper(),
                'units': ds_save[var_name].attrs.get('units', 'unknown')
            })
    
    # Add season attribute to all variables
    ds_save[var_name].attrs['season'] = 'JJA'
    
    # Ensure lat/lon coordinate attributes are present for CF compliance
    if 'lat' in ds_save.coords:
        ds_save['lat'].attrs.update({
            'standard_name': 'latitude',
            'long_name': 'latitude',
            'units': 'degrees_north'
        })
    if 'lon' in ds_save.coords:
        ds_save['lon'].attrs.update({
            'standard_name': 'longitude',
            'long_name': 'longitude',
            'units': 'degrees_east'
        })
    
    # Add time coordinate attributes
    if 'time' in ds_save.coords:
        ds_save['time'].attrs.update({
            'standard_name': 'time',
            'long_name': 'time',
            'description': 'Mid-season date for JJA seasonal mean'
        })
    
    # Add global attributes
    var_title_map = {
        'ght': 'Geopotential Height',
        'u': 'Zonal Wind',
        'v': 'Meridional Wind',
        'rain': 'Precipitation'
    }
    var_title = var_title_map.get(variable_name, variable_name.upper())
    
    title_parts = [f'WRF-CFS {var_title}']
    if pressure_level is not None:
        title_parts.append(f'{pressure_level}hPa')
    title_parts.append('JJA Mean')
    
    ds_save.attrs.update({
        'title': ' '.join(title_parts),
        'source': 'WRF V4.6.0 Model with CFS forcing',
        'institution': 'Seasonal Forecast',
        'history': f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        'processing': 'Interpolated to regular grid and averaged over JJA season',
        'conventions': 'CF-1.8',
        'years_included': str(sorted([int(y) for y in years])) if len(years) > 0 else 'unknown',
        'variable_name': variable_name
    })
    
    # Encoding for compression
    encoding = {}
    if compression:
        for var in ds_save.data_vars:
            encoding[var] = {
                'zlib': True,
                'complevel': complevel,
                'dtype': 'float32'
            }
        # Also encode coordinates
        for coord in ['time', 'lat', 'lon']:
            if coord in ds_save.coords:
                encoding[coord] = {
                    'zlib': True,
                    'complevel': complevel
                }
    
    # Save to NetCDF
    ds_save.to_netcdf(filepath, encoding=encoding, format='NETCDF4')
    
    return filepath


def build_standard_coords(years, lat_values, lon_values):
    """
    Build standard coordinates according to specifications:
    - latitude: descending order (large to small)
    - longitude: ascending order (small to large)
    - time: YYYY-07-01 format, ascending by year
    
    Parameters
    ----------
    years : list or array
        List of years to include
    lat_values : array-like
        Latitude values (will be sorted descending)
    lon_values : array-like
        Longitude values (will be sorted ascending)
        
    Returns
    -------
    dict
        Dictionary with 'latitude', 'longitude', 'time' coordinates
    """
    import pandas as pd
    import numpy as np
    
    # Ensure years is a list and sorted
    if isinstance(years, np.ndarray):
        years = years.tolist()
    years = sorted([int(y) for y in years])
    
    # Build time coordinate: YYYY-07-01, ascending by year
    time_coords = pd.to_datetime([f"{y}-07-01" for y in years])
    
    # Build latitude coordinate: descending (large to small)
    latitude = np.sort(np.array(lat_values))[::-1]  # Sort ascending then reverse
    
    # Build longitude coordinate: ascending (small to large)
    longitude = np.sort(np.array(lon_values))
    
    return {
        'time': time_coords,
        'latitude': latitude,
        'longitude': longitude
    }


def align_to_existing_coords(ds_new, existing_latitude, existing_longitude):
    """
    Align new dataset's coordinates to match existing file's coordinate order.
    
    Parameters
    ----------
    ds_new : xarray.Dataset
        New dataset to align (should have 'lat' or 'latitude', 'lon' or 'longitude')
    existing_latitude : array-like
        Target latitude values (in desired order: descending)
    existing_longitude : array-like
        Target longitude values (in desired order: ascending)
        
    Returns
    -------
    xarray.Dataset
        Dataset with aligned coordinates renamed to 'latitude' and 'longitude'
    """
    import numpy as np
    
    # Identify current coordinate names
    lat_name = 'latitude' if 'latitude' in ds_new.coords else 'lat'
    lon_name = 'longitude' if 'longitude' in ds_new.coords else 'lon'
    
    # Get current coordinate values
    current_lat = ds_new[lat_name].values
    current_lon = ds_new[lon_name].values
    
    # Reindex to match existing coordinate order
    # Use xarray's reindex with method='nearest' for floating point tolerance
    ds_aligned = ds_new.copy()
    
    # Ensure standard coordinate names (latitude/longitude)
    if lat_name == 'lat':
        ds_aligned = ds_aligned.rename({'lat': 'latitude'})
        lat_name = 'latitude'
    if lon_name == 'lon':
        ds_aligned = ds_aligned.rename({'lon': 'longitude'})
        lon_name = 'longitude'
    
    # Reindex latitude
    ds_aligned = ds_aligned.reindex({lat_name: existing_latitude}, method='nearest', tolerance=0.01)
    
    # Reindex longitude
    ds_aligned = ds_aligned.reindex({lon_name: existing_longitude}, method='nearest', tolerance=0.01)
    
    # Rename coordinates to standard names if needed
    if lat_name != 'latitude':
        ds_aligned = ds_aligned.rename({lat_name: 'latitude'})
    if lon_name != 'longitude':
        ds_aligned = ds_aligned.rename({lon_name: 'longitude'})
    
    return ds_aligned


def check_year_exists_in_file(output_path, variable_name, year, pressure_level=None):
    """
    Check if a year already exists in the output NetCDF file.
    Supports both old (lat/lon) and new (latitude/longitude) coordinate names.
    
    Parameters
    ----------
    output_path : str or Path
        Output directory
    variable_name : str
        Output variable name (e.g., 'ght', 'u', 'v', 'rain')
    year : int
        Year to check
    pressure_level : int or None, optional
        Pressure level in hPa (None for 2D variables like precipitation)
        
    Returns
    -------
    bool
        True if year exists in file, False otherwise
    """
    output_path = Path(output_path)
    
    # Create output filename
    if pressure_level is not None:
        filename = f"{variable_name}_{pressure_level}hPa_JJA.nc"
    else:
        filename = f"{variable_name}_JJA.nc"
    filepath = output_path / filename
    
    # If file doesn't exist, year doesn't exist
    if not filepath.exists():
        return False
    
    try:
        import pandas as pd
        # Open file and check if year exists
        with xr.open_dataset(filepath) as ds:
            if 'time' in ds.coords:
                existing_years = pd.to_datetime(ds.time.values).year
                year_exists = year in existing_years.values
                return year_exists
            else:
                return False
    except Exception:
        # If there's any error reading the file, assume year doesn't exist
        # and let the processing continue (will be handled by append_to_netcdf_file)
        return False


def _standardize_lat_lon_coords(ds):
    """
    Ensure dataset uses 1D latitude/longitude coords with CF-compliant names.
    Standardizes coordinate names to 'latitude' and 'longitude' for CF conventions.
    """
    ds_std = ds.copy()
    rename_dict = {}
    # Rename lat -> latitude, lon -> longitude for CF compliance
    if 'lat' in ds_std.coords:
        rename_dict['lat'] = 'latitude'
    if 'lon' in ds_std.coords:
        rename_dict['lon'] = 'longitude'
    if rename_dict:
        ds_std = ds_std.rename(rename_dict)
    # Drop any stray 2D lat/lon variables that are not dimension coords
    for extra in ['lat', 'lon']:
        if extra in ds_std.coords:
            ds_std = ds_std.drop_vars(extra)
        if extra in ds_std.data_vars:
            ds_std = ds_std.drop_vars(extra)
    # Keep only latitude/longitude coords as 1D if possible
    for coord in ['latitude', 'longitude']:
        if coord in ds_std.coords and ds_std[coord].ndim > 1:
            # take unique along flattened order
            vals = ds_std[coord].values
            if coord == 'latitude':
                ds_std = ds_std.assign_coords(latitude=np.unique(vals)[::-1])
            else:
                ds_std = ds_std.assign_coords(longitude=np.unique(vals))
    return ds_std


def append_to_netcdf_file(ds, output_path, variable_name, year, pressure_level=None, compression=True, complevel=4):
    """
    Append single-year processed data to NetCDF file.
    If file doesn't exist, create it. If it exists, append along time dimension.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Single-year dataset to append (should have spatial dimensions only)
    output_path : str or Path
        Output directory
    variable_name : str
        Output variable name (e.g., 'ght', 'u', 'v', 'rain')
    year : int
        Year of the data
    pressure_level : int or None, optional
        Pressure level in hPa (None for 2D variables like precipitation)
    compression : bool, optional
        Enable compression (default: True)
    complevel : int, optional
        Compression level (default: 4)
        
    Returns
    -------
    Path
        Path to saved/updated file
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get logger
    logger = logging.getLogger('WRF_CFS_Processing')
    
    # Create output filename
    if pressure_level is not None:
        filename = f"{variable_name}_{pressure_level}hPa_JJA.nc"
    else:
        filename = f"{variable_name}_JJA.nc"
    filepath = output_path / filename
    
    import pandas as pd
    
    # Prepare single-year dataset
    ds_year = _standardize_lat_lon_coords(ds)
    
    # Add time coordinate using standard format: YYYY-07-01
    time_coord = pd.to_datetime(f"{year}-07-01")
    ds_year = ds_year.expand_dims('time')
    ds_year = ds_year.assign_coords(time=[time_coord])
    
    # Get the data variable name (should be only one)
    data_vars = list(ds_year.data_vars)
    if len(data_vars) != 1:
        raise ValueError(f"Expected exactly one data variable, found {len(data_vars)}")
    
    var_name = data_vars[0]
    
    # Check original units from processed data (preserved through interpolation and averaging)
    # These units come from the original WRF data and are preserved through:
    # 1. xr.open_mfdataset() - reads original units
    # 2. interpolate_to_regular_grid() - keep_attrs=True preserves units
    # 3. compute_seasonal_mean() - keep_attrs=True preserves units
    original_units = ds_year[var_name].attrs.get('units', None)
    if original_units:
        logger.info(f"Found original units for {variable_name} ({var_name}): {original_units}")
    else:
        logger.warning(f"No original units found for {variable_name} ({var_name}), using default")
    
    # Get existing attributes from processed data (preserved from compute_seasonal_mean)
    existing_attrs = ds_year[var_name].attrs.copy()
    existing_description = existing_attrs.get('description', '')
    existing_long_name = existing_attrs.get('long_name', var_name)
    processing_method = existing_attrs.get('processing_method', 'unknown')
    
    # Update variable attributes based on variable type
    # Use original units if available and reasonable, otherwise use defaults
    # Preserve and enhance descriptions from compute_seasonal_mean
    if variable_name == 'ght':
        # WRF GHT_PL units: typically 'm' or 'gpm' (geopotential meters, 1 gpm ≈ 1 m)
        units = original_units if original_units in ['m', 'gpm', 'meter', 'meters'] else 'm'
        if pressure_level is not None:
            long_name = f'Geopotential Height at {pressure_level} hPa'
            description = (
                f"Geopotential height at {pressure_level} hPa pressure level. "
                f"{existing_description if existing_description else 'Instantaneous geopotential height from WRF model, averaged over JJA season.'} "
                f"Geopotential height represents the height of a pressure surface above sea level. "
                f"Original WRF variable: GHT_PL. Units: {units}."
            )
            ds_year[var_name].attrs.update({
                'long_name': long_name,
                'standard_name': 'geopotential_height',
                'description': description,
                'units': units,
                'pressure_level': f'{pressure_level} hPa',
                'original_wrf_variable': 'GHT_PL'
            })
        else:
            description = (
                f"Geopotential height. "
                f"{existing_description if existing_description else 'Instantaneous geopotential height from WRF model, averaged over JJA season.'} "
                f"Original WRF variable: GHT_PL. Units: {units}."
            )
            ds_year[var_name].attrs.update({
                'long_name': 'Geopotential Height',
                'standard_name': 'geopotential_height',
                'description': description,
                'units': units,
                'original_wrf_variable': 'GHT_PL'
            })
    elif variable_name == 'u':
        # WRF U_PL units: typically 'm s-1'
        units = original_units if original_units and 'm' in original_units and 's' in original_units else 'm s-1'
        if pressure_level is not None:
            long_name = f'Zonal Wind Component at {pressure_level} hPa'
            description = (
                f"Zonal (eastward) wind component at {pressure_level} hPa pressure level. "
                f"{existing_description if existing_description else 'Instantaneous zonal wind from WRF model, averaged over JJA season.'} "
                f"Positive values indicate eastward (westerly) wind. "
                f"Original WRF variable: U_PL. Units: {units}."
            )
            ds_year[var_name].attrs.update({
                'long_name': long_name,
                'standard_name': 'eastward_wind',
                'description': description,
                'units': units,
                'pressure_level': f'{pressure_level} hPa',
                'original_wrf_variable': 'U_PL'
            })
        else:
            description = (
                f"Zonal (eastward) wind component. "
                f"{existing_description if existing_description else 'Instantaneous zonal wind from WRF model, averaged over JJA season.'} "
                f"Positive values indicate eastward (westerly) wind. "
                f"Original WRF variable: U_PL. Units: {units}."
            )
            ds_year[var_name].attrs.update({
                'long_name': 'Zonal Wind Component',
                'standard_name': 'eastward_wind',
                'description': description,
                'units': units,
                'original_wrf_variable': 'U_PL'
            })
    elif variable_name == 'v':
        # WRF V_PL units: typically 'm s-1'
        units = original_units if original_units and 'm' in original_units and 's' in original_units else 'm s-1'
        if pressure_level is not None:
            long_name = f'Meridional Wind Component at {pressure_level} hPa'
            description = (
                f"Meridional (northward) wind component at {pressure_level} hPa pressure level. "
                f"{existing_description if existing_description else 'Instantaneous meridional wind from WRF model, averaged over JJA season.'} "
                f"Positive values indicate northward wind. "
                f"Original WRF variable: V_PL. Units: {units}."
            )
            ds_year[var_name].attrs.update({
                'long_name': long_name,
                'standard_name': 'northward_wind',
                'description': description,
                'units': units,
                'pressure_level': f'{pressure_level} hPa',
                'original_wrf_variable': 'V_PL'
            })
        else:
            description = (
                f"Meridional (northward) wind component. "
                f"{existing_description if existing_description else 'Instantaneous meridional wind from WRF model, averaged over JJA season.'} "
                f"Positive values indicate northward wind. "
                f"Original WRF variable: V_PL. Units: {units}."
            )
            ds_year[var_name].attrs.update({
                'long_name': 'Meridional Wind Component',
                'standard_name': 'northward_wind',
                'description': description,
                'units': units,
                'original_wrf_variable': 'V_PL'
            })
    elif variable_name == 'rain':
        # WRF RAINNC/RAINC units: typically 'mm' (accumulated precipitation)
        # Original WRF variables are ACCUMULATED (from model start)
        units = original_units if original_units and 'mm' in original_units.lower() else 'mm'
        description = (
            f"Precipitation averaged over the JJA season. "
            f"Original WRF variables: RAINNC (ACCUMULATED TOTAL GRID SCALE PRECIPITATION) + "
            f"RAINC (ACCUMULATED TOTAL CUMULUS PRECIPITATION) + "
            f"RAINSH (ACCUMULATED SHALLOW CUMULUS PRECIPITATION, if available). "
            f"These are accumulated precipitation variables in WRF, measured from model start. "
            f"Processing method: Time average of accumulated values over JJA season. "
            f"Note: This represents the time-averaged accumulated precipitation values, "
            f"not the total accumulated precipitation over the season. "
            f"The units ({units}) refer to the average of accumulated values at each time step. "
            f"For total seasonal precipitation, this value would need to be multiplied by the number of time steps, "
            f"or the difference method (final - initial) should be used instead."
        )
        ds_year[var_name].attrs.update({
            'long_name': 'Precipitation (Time-Averaged Accumulated)',
            'standard_name': 'precipitation_amount',
            'description': description,
            'units': units,
            'original_wrf_variables': 'RAINNC + RAINC + RAINSH',
            'processing_note': 'Time average of accumulated values, not total accumulation'
        })
    else:
        # Generic variable - preserve existing description if available
        if existing_description:
            description = existing_description
        else:
            description = f"Processed {variable_name} variable from WRF model."
        
        if pressure_level is not None:
            ds_year[var_name].attrs.update({
                'long_name': existing_long_name if existing_long_name else f'{variable_name} at {pressure_level} hPa',
                'description': description,
                'units': units if 'units' in locals() else original_units if original_units else 'unknown',
                'pressure_level': f'{pressure_level} hPa'
            })
        else:
            ds_year[var_name].attrs.update({
                'long_name': existing_long_name if existing_long_name else variable_name,
                'description': description,
                'units': units if 'units' in locals() else original_units if original_units else 'unknown'
            })
    
    # Add season attribute
    ds_year[var_name].attrs['season'] = 'JJA'
    
    # Ensure latitude/longitude coordinate attributes (using standard names)
    if 'latitude' in ds_year.coords:
        ds_year['latitude'].attrs.update({
            'standard_name': 'latitude',
            'long_name': 'latitude',
            'units': 'degrees_north'
        })
    if 'longitude' in ds_year.coords:
        ds_year['longitude'].attrs.update({
            'standard_name': 'longitude',
            'long_name': 'longitude',
            'units': 'degrees_east'
        })
    
    # Add time coordinate attributes with CF-compliant units and calendar
    import pandas as pd
    time_coord = ds_year['time']
    # CF convention: "days since YYYY-MM-DD HH:MM:SS"
    time_units = f"days since {year}-01-01 00:00:00"
    time_calendar = 'gregorian'
    
    # Set CF-compliant attributes (xarray will handle datetime conversion when saving)
    ds_year['time'].attrs.update({
        'standard_name': 'time',
        'long_name': 'time',
        'description': 'Mid-season date for JJA seasonal mean',
        'units': time_units,
        'calendar': time_calendar
    })
    
    # File-level lock to avoid concurrent writers corrupting the NetCDF
    lock_path = filepath.with_suffix(filepath.suffix + ".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        
        # Check if file exists and is valid
        file_exists = filepath.exists()
        file_valid = True
        ds_existing_loaded = None
        existing_lat = None
        existing_lon = None
        
        if file_exists:
            try:
                with xr.open_dataset(filepath) as ds_existing:
                    # Basic validation: has data_vars and lat/lon coords
                    has_coords = ('lat' in ds_existing.coords or 'latitude' in ds_existing.coords) and \
                                 ('lon' in ds_existing.coords or 'longitude' in ds_existing.coords)
                    if len(ds_existing.data_vars) == 0 or not has_coords:
                        logger.warning(f"Existing file {filename} is invalid (missing data or lat/lon). Will rebuild.")
                        file_valid = False
                    else:
                        # Check duplicate year
                        if 'time' in ds_existing.coords:
                            # Handle both datetime and numeric time coordinates
                            try:
                                if ds_existing.time.dtype.kind == 'M':  # datetime64
                                    existing_years = pd.to_datetime(ds_existing.time.values).year
                                else:  # numeric time
                                    # Try to parse units to get year
                                    time_units = ds_existing.time.attrs.get('units', '')
                                    if 'since' in time_units:
                                        ref_year = int(time_units.split('since')[1].strip().split('-')[0])
                                        # Approximate: days since year start, so year = ref_year
                                        existing_years = np.array([ref_year] * len(ds_existing.time))
                                    else:
                                        existing_years = np.array([year])  # fallback
                                if year in existing_years:
                                    logger.warning(f"Year {year} already exists in {filename}, skipping append")
                                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                                    return filepath
                            except Exception as e:
                                logger.warning(f"Could not check duplicate year: {e}")
                        # Standardize coordinate names to latitude/longitude
                        lat_coord_name = 'latitude' if 'latitude' in ds_existing.coords else 'lat'
                        lon_coord_name = 'longitude' if 'longitude' in ds_existing.coords else 'lon'
                        existing_lat = ds_existing[lat_coord_name].values.copy()
                        existing_lon = ds_existing[lon_coord_name].values.copy()
                        ds_existing_loaded = ds_existing.load()
                        # Rename coordinates to standard names if needed
                        if lat_coord_name == 'lat':
                            ds_existing_loaded = ds_existing_loaded.rename({'lat': 'latitude'})
                        if lon_coord_name == 'lon':
                            ds_existing_loaded = ds_existing_loaded.rename({'lon': 'longitude'})
            except Exception as e:
                logger.error(f"Error reading existing file {filename}, will rebuild: {str(e)}")
                file_valid = False
        
        # If invalid existing file, back it up and rebuild
        if file_exists and not file_valid:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = filepath.with_suffix(filepath.suffix + f".corrupt.{ts}")
            try:
                filepath.replace(backup_path)
                logger.warning(f"Backed up invalid file to {backup_path}")
            except Exception as e:
                logger.warning(f"Failed to backup invalid file {filepath}: {e}")
            file_exists = False
        
        if file_exists and ds_existing_loaded is not None:
            # Align new year's data to existing coordinate order
            ds_year_aligned = align_to_existing_coords(ds_year, existing_lat, existing_lon)
            
            # Concatenate along time dimension (now both datasets are in memory)
            ds_combined = xr.concat([ds_existing_loaded, ds_year_aligned], dim='time')
            
            # Sort by time to ensure chronological order
            ds_combined = ds_combined.sortby('time')
            
            ds_save = ds_combined
            
            # Update history attribute
            if 'history' in ds_save.attrs:
                ds_save.attrs['history'] += f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Appended year {year}"
            else:
                ds_save.attrs['history'] = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Appended year {year}"
        else:
            # Create new file with standard coordinate ordering
            # Ensure latitude is descending and longitude is ascending
            lat_coord_name = 'latitude' if 'latitude' in ds_year.coords else 'lat'
            lon_coord_name = 'longitude' if 'longitude' in ds_year.coords else 'lon'
            
            lat_vals = ds_year[lat_coord_name].values
            lon_vals = ds_year[lon_coord_name].values
            
            # Check and apply standard ordering
            ds_save = ds_year.copy()
            
            # Ensure latitude is descending (large to small)
            if len(lat_vals) > 1 and lat_vals[0] < lat_vals[-1]:
                ds_save = ds_save.reindex({lat_coord_name: lat_vals[::-1]})
            
            # Ensure longitude is ascending (small to large)
            if len(lon_vals) > 1 and lon_vals[0] > lon_vals[-1]:
                ds_save = ds_save.reindex({lon_coord_name: lon_vals[::-1]})
            
            # Set up global attributes
            var_title_map = {
                'ght': 'Geopotential Height',
                'u': 'Zonal Wind',
                'v': 'Meridional Wind',
                'rain': 'Precipitation'
            }
            var_title = var_title_map.get(variable_name, variable_name.upper())
            
            title_parts = [f'WRF-CFS {var_title}']
            if pressure_level is not None:
                title_parts.append(f'{pressure_level}hPa')
            title_parts.append('JJA Mean')
            
            ds_save.attrs.update({
                'title': ' '.join(title_parts),
                'source': 'WRF V4.6.0 Model with CFS forcing',
                'institution': 'Seasonal Forecast',
                'history': f'Created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                'processing': 'Interpolated to regular grid and averaged over JJA season',
                'conventions': 'CF-1.8',
                'variable_name': variable_name
            })
        
        # Update global attributes with years_included
        if 'years_included' in ds_save.attrs:
            try:
                years_str = ds_save.attrs['years_included']
                import ast
                years_list = ast.literal_eval(years_str) if isinstance(years_str, str) else list(years_str)
                if year not in years_list:
                    years_list.append(year)
                    years_list.sort()
                ds_save.attrs['years_included'] = str(years_list)
            except Exception:
                ds_save.attrs['years_included'] = str([year])
        else:
            ds_save.attrs['years_included'] = str([year])
        
        # Encoding for compression
        encoding = {}
        if compression:
            for var in ds_save.data_vars:
                encoding[var] = {
                    'zlib': True,
                    'complevel': complevel,
                    'dtype': 'float32'
                }
            coord_names = ['time']
            if 'latitude' in ds_save.coords:
                coord_names.append('latitude')
            elif 'lat' in ds_save.coords:
                coord_names.append('lat')
            if 'longitude' in ds_save.coords:
                coord_names.append('longitude')
            elif 'lon' in ds_save.coords:
                coord_names.append('lon')
            
            for coord in coord_names:
                if coord in ds_save.coords:
                    encoding[coord] = {
                        'zlib': True,
                        'complevel': complevel
                    }
        
        # Ensure coordinates are float64 for precision and consistency
        if 'latitude' in ds_save.coords:
            if ds_save['latitude'].dtype != np.float64:
                logger.debug(f"Converting latitude from {ds_save['latitude'].dtype} to float64")
                ds_save['latitude'] = ds_save['latitude'].astype(np.float64)
        elif 'lat' in ds_save.coords:
            if ds_save['lat'].dtype != np.float64:
                logger.debug(f"Converting lat from {ds_save['lat'].dtype} to float64")
                ds_save['lat'] = ds_save['lat'].astype(np.float64)
        
        if 'longitude' in ds_save.coords:
            if ds_save['longitude'].dtype != np.float64:
                logger.debug(f"Converting longitude from {ds_save['longitude'].dtype} to float64")
                ds_save['longitude'] = ds_save['longitude'].astype(np.float64)
        elif 'lon' in ds_save.coords:
            if ds_save['lon'].dtype != np.float64:
                logger.debug(f"Converting lon from {ds_save['lon'].dtype} to float64")
                ds_save['lon'] = ds_save['lon'].astype(np.float64)
        
        # Debug: Check data before saving
        logger.debug(f"Before saving: ds_save has {len(ds_save.data_vars)} data vars: {list(ds_save.data_vars.keys())}")
        logger.debug(f"Before saving: ds_save coords: {list(ds_save.coords.keys())}")
        logger.debug(f"Before saving: ds_save dims: {dict(ds_save.dims)}")
        if len(ds_save.data_vars) > 0:
            var_name = list(ds_save.data_vars.keys())[0]
            logger.debug(f"Before saving: {var_name} shape: {ds_save[var_name].shape}, dtype: {ds_save[var_name].dtype}")
        
        # Save to NetCDF (overwrite mode)
        try:
            ds_save.to_netcdf(filepath, encoding=encoding, format='NETCDF4', mode='w')
            logger.debug(f"File saved successfully: {filepath}")
        except Exception as e:
            logger.error(f"Error saving file {filepath}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        # Clean up to release memory
        del ds_save
        if file_exists and ds_existing_loaded is not None:
            del ds_combined
        
        # Unlock happens via context manager
    
    return filepath

