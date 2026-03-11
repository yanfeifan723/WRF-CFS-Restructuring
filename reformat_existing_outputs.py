#!/usr/bin/env python
"""
Reformat Existing NetCDF Outputs Script

This script reformats existing WRF-CFS processed NetCDF files to follow standardized conventions:
1. Rename coordinates: lat -> latitude, lon -> longitude
2. Ensure coordinate ordering:
   - latitude: descending (large to small, e.g., 58, 57, ..., 10)
   - longitude: ascending (small to large, e.g., 70, 71, ..., 140)
3. Standardize time dimension:
   - Format: YYYY-07-01 (instead of YYYY-07-15)
   - Order: ascending by year

Usage:
    python reformat_existing_outputs.py --input-dir /path/to/outputs [--backup]
    
Options:
    --input-dir: Directory containing NC files to reformat (default: WRF_CFS_processed)
    --backup: Create backup files (*_backup.nc) before overwriting (default: False)
    --dry-run: Show what would be done without actually modifying files
"""

import argparse
import sys
from pathlib import Path
import shutil
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime
import re


def set_variable_attributes(ds, var_name, pressure_level=None):
    """
    Set comprehensive attributes for a variable based on its type.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the variable
    var_name : str
        Variable name
    pressure_level : int or None
        Pressure level in hPa (None for 2D variables)

    Returns
    -------
    dict
        Dictionary of attributes to update
    """
    # Get existing attributes
    existing_attrs = ds[var_name].attrs.copy()
    existing_description = existing_attrs.get('description', '')
    existing_long_name = existing_attrs.get('long_name', var_name)
    original_units = existing_attrs.get('units', None) or existing_attrs.get('original_units', None)

    # Normalize variable name for matching
    var_lower = var_name.lower()

    # Determine variable type and set attributes
    if var_lower == 'ght' or 'ght' in var_lower:
        # Geopotential height
        units = original_units if original_units in ['m', 'gpm', 'meter', 'meters'] else 'm'
        if pressure_level is not None:
            long_name = f'Geopotential Height at {pressure_level} hPa'
            description = (
                f"Geopotential height at {pressure_level} hPa pressure level. "
                f"{existing_description if existing_description else 'Instantaneous geopotential height from WRF model, averaged over JJA season.'} "
                f"Geopotential height represents the height of a pressure surface above sea level. "
                f"Original WRF variable: GHT_PL. Units: {units}."
            )
            attrs = {
                'long_name': long_name,
                'standard_name': 'geopotential_height',
                'description': description,
                'units': units,
                'pressure_level': f'{pressure_level} hPa',
                'original_wrf_variable': 'GHT_PL'
            }
        else:
            description = (
                f"Geopotential height. "
                f"{existing_description if existing_description else 'Instantaneous geopotential height from WRF model, averaged over JJA season.'} "
                f"Original WRF variable: GHT_PL. Units: {units}."
            )
            attrs = {
                'long_name': 'Geopotential Height',
                'standard_name': 'geopotential_height',
                'description': description,
                'units': units,
                'original_wrf_variable': 'GHT_PL'
            }

    elif var_lower in ['u', 'u_pl'] or 'zonal' in existing_long_name.lower() or var_lower.startswith('u_'):
        # Zonal wind component
        units = original_units if original_units and 'm' in original_units and 's' in original_units else 'm s-1'
        if pressure_level is not None:
            long_name = f'Zonal Wind Component at {pressure_level} hPa'
            description = (
                f"Zonal (eastward) wind component at {pressure_level} hPa pressure level. "
                f"{existing_description if existing_description else 'Instantaneous zonal wind from WRF model, averaged over JJA season.'} "
                f"Positive values indicate eastward (westerly) wind. "
                f"Original WRF variable: U_PL. Units: {units}."
            )
            attrs = {
                'long_name': long_name,
                'standard_name': 'eastward_wind',
                'description': description,
                'units': units,
                'pressure_level': f'{pressure_level} hPa',
                'original_wrf_variable': 'U_PL'
            }
        else:
            description = (
                f"Zonal (eastward) wind component. "
                f"{existing_description if existing_description else 'Instantaneous zonal wind from WRF model, averaged over JJA season.'} "
                f"Positive values indicate eastward (westerly) wind. "
                f"Original WRF variable: U_PL. Units: {units}."
            )
            attrs = {
                'long_name': 'Zonal Wind Component',
                'standard_name': 'eastward_wind',
                'description': description,
                'units': units,
                'original_wrf_variable': 'U_PL'
            }

    elif var_lower in ['v', 'v_pl'] or 'meridional' in existing_long_name.lower() or var_lower.startswith('v_'):
        # Meridional wind component
        units = original_units if original_units and 'm' in original_units and 's' in original_units else 'm s-1'
        if pressure_level is not None:
            long_name = f'Meridional Wind Component at {pressure_level} hPa'
            description = (
                f"Meridional (northward) wind component at {pressure_level} hPa pressure level. "
                f"{existing_description if existing_description else 'Instantaneous meridional wind from WRF model, averaged over JJA season.'} "
                f"Positive values indicate northward wind. "
                f"Original WRF variable: V_PL. Units: {units}."
            )
            attrs = {
                'long_name': long_name,
                'standard_name': 'northward_wind',
                'description': description,
                'units': units,
                'pressure_level': f'{pressure_level} hPa',
                'original_wrf_variable': 'V_PL'
            }
        else:
            description = (
                f"Meridional (northward) wind component. "
                f"{existing_description if existing_description else 'Instantaneous meridional wind from WRF model, averaged over JJA season.'} "
                f"Positive values indicate northward wind. "
                f"Original WRF variable: V_PL. Units: {units}."
            )
            attrs = {
                'long_name': 'Meridional Wind Component',
                'standard_name': 'northward_wind',
                'description': description,
                'units': units,
                'original_wrf_variable': 'V_PL'
            }

    elif var_lower in ['rain', 'precip'] or 'rain' in var_lower or 'precip' in var_lower:
        # Precipitation
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
        attrs = {
            'long_name': 'Precipitation (Time-Averaged Accumulated)',
            'standard_name': 'precipitation_amount',
            'description': description,
            'units': units,
            'original_wrf_variables': 'RAINNC + RAINC + RAINSH',
            'processing_note': 'Time average of accumulated values, not total accumulation'
        }

    else:
        # Generic variable - keep existing attributes but ensure units are present
        units = existing_attrs.get('units', 'unknown')
        if pressure_level is not None:
            attrs = {
                'long_name': f'{var_name.upper()} at {pressure_level} hPa',
                'units': units,
                'pressure_level': f'{pressure_level} hPa'
            }
        else:
            attrs = {
                'long_name': var_name.upper(),
                'units': units
            }

    # Add season attribute to all variables
    attrs['season'] = 'JJA'

    return attrs


def reformat_netcdf_file(filepath, backup=False, dry_run=False):
    """
    Reformat a single NetCDF file to follow standardized conventions.
    
    Parameters
    ----------
    filepath : Path
        Path to NetCDF file
    backup : bool
        Whether to create a backup before modifying
    dry_run : bool
        If True, only show what would be done without modifying
        
    Returns
    -------
    dict
        Status information about the reformatting
    """
    result = {
        'file': str(filepath),
        'status': 'success',
        'changes': [],
        'errors': []
    }
    
    try:
        print(f"\nProcessing: {filepath.name}")
        
        # Open dataset
        ds = xr.open_dataset(filepath)
        
        # Track what needs to be changed
        needs_reformat = False
        
        # 1. Check and rename coordinates
        rename_dict = {}
        if 'lat' in ds.coords and 'latitude' not in ds.coords:
            rename_dict['lat'] = 'latitude'
            result['changes'].append("Rename 'lat' -> 'latitude'")
            needs_reformat = True
        if 'lon' in ds.coords and 'longitude' not in ds.coords:
            rename_dict['lon'] = 'longitude'
            result['changes'].append("Rename 'lon' -> 'longitude'")
            needs_reformat = True
        
        if rename_dict:
            ds = ds.rename(rename_dict)
        
        # 1.5. Check and add CF-compliant coordinate attributes
        lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
        
        # Check latitude attributes
        if lat_name in ds.coords:
            lat_attrs = ds[lat_name].attrs
            required_lat_attrs = {
                'standard_name': 'latitude',
                'long_name': 'latitude',
                'units': 'degrees_north'
            }
            for attr_name, attr_value in required_lat_attrs.items():
                if attr_name not in lat_attrs or lat_attrs[attr_name] != attr_value:
                    result['changes'].append(f"Add/update {lat_name}.{attr_name} = '{attr_value}'")
                    needs_reformat = True
        
        # Check longitude attributes
        if lon_name in ds.coords:
            lon_attrs = ds[lon_name].attrs
            required_lon_attrs = {
                'standard_name': 'longitude',
                'long_name': 'longitude',
                'units': 'degrees_east'
            }
            for attr_name, attr_value in required_lon_attrs.items():
                if attr_name not in lon_attrs or lon_attrs[attr_name] != attr_value:
                    result['changes'].append(f"Add/update {lon_name}.{attr_name} = '{attr_value}'")
                    needs_reformat = True
        
        # 2. Check coordinate ordering
        lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
        lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
        
        # Check latitude ordering (should be descending)
        if lat_name in ds.coords:
            lat_vals = ds[lat_name].values
            if len(lat_vals) > 1 and lat_vals[0] < lat_vals[-1]:
                # Currently ascending, need to reverse
                ds = ds.reindex({lat_name: lat_vals[::-1]})
                result['changes'].append(f"Reverse {lat_name} order to descending")
                needs_reformat = True
        
        # Check longitude ordering (should be ascending)
        if lon_name in ds.coords:
            lon_vals = ds[lon_name].values
            if len(lon_vals) > 1 and lon_vals[0] > lon_vals[-1]:
                # Currently descending, need to reverse
                ds = ds.reindex({lon_name: lon_vals[::-1]})
                result['changes'].append(f"Reverse {lon_name} order to ascending")
                needs_reformat = True
        
        # Check coordinate dtypes (should be float64)
        if lat_name in ds.coords:
            if ds[lat_name].dtype != np.float64:
                result['changes'].append(f"Convert {lat_name} from {ds[lat_name].dtype} to float64")
                needs_reformat = True
        
        if lon_name in ds.coords:
            if ds[lon_name].dtype != np.float64:
                result['changes'].append(f"Convert {lon_name} from {ds[lon_name].dtype} to float64")
                needs_reformat = True
        
        # 3. Check and correct units and attributes for data variables
        # Based on WRF standard units and processing flow
        unit_corrections = {}
        attribute_updates = {}

        # Try to extract pressure level from filename
        pressure_level = None
        filename = filepath.name
        # Look for pressure level patterns like "500hPa", "850hPa", etc.
        pressure_match = re.search(r'(\d+)hPa', filename)
        if pressure_match:
            pressure_level = int(pressure_match.group(1))

        for var in ds.data_vars:
            current_units = ds[var].attrs.get('units', '')
            var_lower = var.lower()

            # Get the comprehensive attributes for this variable
            new_attrs = set_variable_attributes(ds, var, pressure_level)

            # Check if units need correction
            expected_units = new_attrs.get('units', current_units)
            if current_units != expected_units:
                unit_corrections[var] = expected_units
                result['changes'].append(f"Correct {var} units: '{current_units}' -> '{expected_units}'")
                needs_reformat = True

            # Check if attributes need updating
            attrs_to_update = {}
            for attr_name, attr_value in new_attrs.items():
                current_attr_value = ds[var].attrs.get(attr_name)
                if current_attr_value != attr_value:
                    attrs_to_update[attr_name] = attr_value
                    if attr_name != 'units':  # Units change already logged above
                        result['changes'].append(f"Update {var} attribute '{attr_name}': '{current_attr_value}' -> '{attr_value}'")
                        needs_reformat = True

            if attrs_to_update:
                attribute_updates[var] = attrs_to_update
        
        # 4. Check and reformat time coordinate
        if 'time' in ds.coords:
            time_vals = ds.time.values
            time_attrs = ds.time.attrs
            
            # Check if time has CF-compliant attributes
            time_needs_attrs = False
            if 'units' not in time_attrs:
                result['changes'].append("Add time.units attribute")
                time_needs_attrs = True
                needs_reformat = True
            if 'calendar' not in time_attrs:
                result['changes'].append("Add time.calendar attribute")
                time_needs_attrs = True
                needs_reformat = True
            if 'standard_name' not in time_attrs:
                result['changes'].append("Add time.standard_name attribute")
                time_needs_attrs = True
                needs_reformat = True
            
            # Try to determine year from time values
            try:
                if ds.time.dtype.kind == 'M':  # datetime64
                    years = pd.to_datetime(time_vals).year
                else:  # numeric time
                    # Try to parse units to get reference year
                    time_units = time_attrs.get('units', '')
                    if 'since' in time_units:
                        ref_date_str = time_units.split('since')[1].strip()
                        ref_year = int(pd.to_datetime(ref_date_str).year)
                        # Approximate years (this is rough, but works for JJA means)
                        years = np.array([ref_year] * len(time_vals))
                    else:
                        # Fallback: try to infer from filename or use current year
                        years = np.array([2020] * len(time_vals))  # fallback
                
                # Build new time coordinates: YYYY-07-01
                new_time = pd.to_datetime([f"{y}-07-01" for y in years])
                
                # Check if time format needs changing
                if ds.time.dtype.kind == 'M':  # datetime64
                    if not all(pd.to_datetime(new_time) == pd.to_datetime(time_vals)):
                        result['changes'].append("Change time format to YYYY-07-01")
                        needs_reformat = True
                else:
                    # Numeric time - we'll convert to datetime for consistency
                    result['changes'].append("Convert numeric time to datetime (YYYY-07-01)")
                    needs_reformat = True
                
                # Check if time is sorted by year (ascending)
                if len(years) > 1 and not all(years[:-1] <= years[1:]):
                    result['changes'].append("Sort time dimension by year (ascending)")
                    needs_reformat = True
            except Exception as e:
                result['errors'].append(f"Could not process time coordinate: {e}")
                # Still mark as needing reformat for attributes
                if time_needs_attrs:
                    needs_reformat = True
        
        ds.close()
        
        if not needs_reformat:
            print(f"  ✓ Already in standard format, skipping")
            result['status'] = 'skipped'
            return result
        
        if dry_run:
            print(f"  [DRY RUN] Would apply changes:")
            for change in result['changes']:
                print(f"    - {change}")
            result['status'] = 'dry_run'
            return result
        
        # Apply changes
        print(f"  Applying changes:")
        for change in result['changes']:
            print(f"    - {change}")
        
        # Backup if requested
        if backup:
            backup_path = filepath.with_suffix('.backup.nc')
            shutil.copy2(filepath, backup_path)
            print(f"  Backup created: {backup_path.name}")
        
        # Reopen, apply transformations, and load into memory
        with xr.open_dataset(filepath) as ds:
            # Apply renames
            if rename_dict:
                ds = ds.rename(rename_dict)
            
            # Apply coordinate reordering
            lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
            lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
            
            if lat_name in ds.coords:
                lat_vals = ds[lat_name].values
                if len(lat_vals) > 1 and lat_vals[0] < lat_vals[-1]:
                    ds = ds.reindex({lat_name: lat_vals[::-1]})
                # Add CF-compliant attributes
                ds[lat_name].attrs.update({
                    'standard_name': 'latitude',
                    'long_name': 'latitude',
                    'units': 'degrees_north'
                })
            
            if lon_name in ds.coords:
                lon_vals = ds[lon_name].values
                if len(lon_vals) > 1 and lon_vals[0] > lon_vals[-1]:
                    ds = ds.reindex({lon_name: lon_vals[::-1]})
                # Add CF-compliant attributes
                ds[lon_name].attrs.update({
                    'standard_name': 'longitude',
                    'long_name': 'longitude',
                    'units': 'degrees_east'
                })
            
            # Apply unit corrections and attribute updates
            for var, correct_unit in unit_corrections.items():
                if var in ds.data_vars:
                    ds[var].attrs['units'] = correct_unit

            # Apply attribute updates
            for var, attrs_dict in attribute_updates.items():
                if var in ds.data_vars:
                    for attr_name, attr_value in attrs_dict.items():
                        ds[var].attrs[attr_name] = attr_value
            
            # Apply time reformatting
            if 'time' in ds.coords:
                time_vals = ds.time.values
                time_attrs = ds.time.attrs
                
                # Determine years from time values
                try:
                    if ds.time.dtype.kind == 'M':  # datetime64
                        years = pd.to_datetime(time_vals).year
                    else:  # numeric time
                        # Try to parse units to get reference year
                        time_units = time_attrs.get('units', '')
                        if 'since' in time_units:
                            ref_date_str = time_units.split('since')[1].strip()
                            ref_year = int(pd.to_datetime(ref_date_str).year)
                            # For numeric time, we need to convert to datetime
                            # Use the reference date + days
                            if isinstance(time_vals, np.ndarray) and len(time_vals) > 0:
                                # Approximate: assume each value represents a different year
                                # This is a simplification for JJA means
                                years = np.arange(ref_year, ref_year + len(time_vals))
                            else:
                                years = np.array([ref_year])
                        else:
                            # Fallback: use filename or default
                            years = np.array([2020] * len(time_vals))
                    
                    # Build new time coordinates: YYYY-07-01
                    new_time = pd.to_datetime([f"{int(y)}-07-01" for y in years])
                    
                    # Assign new time coordinates
                    ds = ds.assign_coords(time=new_time)
                    
                    # Sort by time if needed
                    if len(years) > 1 and not all(years[:-1] <= years[1:]):
                        ds = ds.sortby('time')
                    
                    # Add CF-compliant time attributes
                    # Use the first year as reference for units
                    ref_year = int(years[0])
                    time_units = f"days since {ref_year}-01-01 00:00:00"
                    
                    # Remove units from encoding if present (xarray doesn't allow overwriting encoding fields)
                    # This must be done before updating attrs
                    if 'time' in ds.encoding:
                        if 'units' in ds.encoding['time']:
                            del ds.encoding['time']['units']
                    # Also check the coordinate's own encoding
                    if hasattr(ds['time'], 'encoding') and ds['time'].encoding:
                        if 'units' in ds['time'].encoding:
                            del ds['time'].encoding['units']
                    
                    # Now safely update attributes
                    # Create a new dict to avoid conflicts
                    new_time_attrs = {
                        'standard_name': 'time',
                        'long_name': 'time',
                        'description': 'Mid-season date for JJA seasonal mean',
                        'units': time_units,
                        'calendar': 'gregorian'
                    }
                    # Clear existing attrs and set new ones to avoid conflicts
                    for key in list(ds['time'].attrs.keys()):
                        if key in new_time_attrs:
                            del ds['time'].attrs[key]
                    ds['time'].attrs.update(new_time_attrs)
                except Exception as e:
                    # If we can't determine years, at least add basic attributes
                    ref_year = 2020  # fallback
                    time_units = f"days since {ref_year}-01-01 00:00:00"
                    
                    # Remove units from encoding if present
                    if 'time' in ds.encoding:
                        if 'units' in ds.encoding['time']:
                            del ds.encoding['time']['units']
                    if hasattr(ds['time'], 'encoding') and ds['time'].encoding:
                        if 'units' in ds['time'].encoding:
                            del ds['time'].encoding['units']
                    
                    # Create a new dict to avoid conflicts
                    new_time_attrs = {
                        'standard_name': 'time',
                        'long_name': 'time',
                        'description': 'Mid-season date for JJA seasonal mean',
                        'units': time_units,
                        'calendar': 'gregorian'
                    }
                    # Clear existing attrs and set new ones to avoid conflicts
                    for key in list(ds['time'].attrs.keys()):
                        if key in new_time_attrs:
                            del ds['time'].attrs[key]
                    ds['time'].attrs.update(new_time_attrs)
                    result['errors'].append(f"Time coordinate processing warning: {e}")
            
            # Ensure coordinates are float64 for precision
            # This fixes issues where coordinates might be int64 (especially longitude)
            coord_dtype_changed = False
            if 'latitude' in ds.coords:
                if ds['latitude'].dtype != np.float64:
                    ds['latitude'] = ds['latitude'].astype(np.float64)
                    coord_dtype_changed = True
                    result['changes'].append(f"Convert latitude from {ds['latitude'].dtype} to float64")
            elif 'lat' in ds.coords:
                if ds['lat'].dtype != np.float64:
                    ds['lat'] = ds['lat'].astype(np.float64)
                    coord_dtype_changed = True
                    result['changes'].append(f"Convert lat from {ds['lat'].dtype} to float64")
            
            if 'longitude' in ds.coords:
                if ds['longitude'].dtype != np.float64:
                    ds['longitude'] = ds['longitude'].astype(np.float64)
                    coord_dtype_changed = True
                    result['changes'].append(f"Convert longitude from {ds['longitude'].dtype} to float64")
            elif 'lon' in ds.coords:
                if ds['lon'].dtype != np.float64:
                    ds['lon'] = ds['lon'].astype(np.float64)
                    coord_dtype_changed = True
                    result['changes'].append(f"Convert lon from {ds['lon'].dtype} to float64")
            
            if coord_dtype_changed:
                needs_reformat = True
            
            # Update global attributes
            if 'history' in ds.attrs:
                ds.attrs['history'] += f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Reformatted coordinates and time"
            else:
                ds.attrs['history'] = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Reformatted coordinates and time"
            
            # Load into memory to break file handle connection
            ds = ds.load()
        
        # Now file handle is closed, we can write
        # Clear any problematic encoding before saving
        if 'time' in ds.encoding:
            # Remove units from encoding if present (it's in attrs now)
            if 'units' in ds.encoding['time']:
                del ds.encoding['time']['units']
        
        # Save back to file (using default encoding for compatibility)
        # Don't include time in encoding to avoid conflicts
        encoding = {}
        for var in ds.data_vars:
            encoding[var] = {'zlib': True, 'complevel': 1}
        # Only encode coordinates that don't have units in attrs
        for coord in ['latitude', 'longitude']:
            if coord in ds.coords:
                encoding[coord] = {'zlib': True, 'complevel': 1}
        
        ds.to_netcdf(filepath, mode='w', encoding=encoding)
        
        print(f"  ✓ Successfully reformatted")
        result['status'] = 'success'
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        result['status'] = 'error'
        result['errors'].append(str(e))
    
    return result


def main():
    """Main function to reformat all NetCDF files in a directory."""
    parser = argparse.ArgumentParser(
        description='Reformat existing WRF-CFS processed NetCDF files to standard conventions'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='WRF_CFS_processed',
        help='Directory containing NC files to reformat (default: WRF_CFS_processed)'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Create backup files before overwriting'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually modifying files'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*_JJA.nc',
        help='File pattern to match (default: *_JJA.nc)'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Find all matching files
    nc_files = sorted(input_dir.glob(args.pattern))
    
    if not nc_files:
        print(f"No NetCDF files found matching pattern '{args.pattern}' in {input_dir}")
        sys.exit(1)
    
    print("="*80)
    print("NetCDF File Reformatter")
    print("="*80)
    print(f"Input directory: {input_dir}")
    print(f"Files found: {len(nc_files)}")
    print(f"Backup enabled: {args.backup}")
    print(f"Dry run: {args.dry_run}")
    print("="*80)
    
    # Process each file
    results = []
    for nc_file in nc_files:
        result = reformat_netcdf_file(nc_file, backup=args.backup, dry_run=args.dry_run)
        results.append(result)
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    error_count = sum(1 for r in results if r['status'] == 'error')
    dry_run_count = sum(1 for r in results if r['status'] == 'dry_run')
    
    print(f"Total files: {len(results)}")
    if args.dry_run:
        print(f"Would reformat: {dry_run_count}")
        print(f"Already standard: {skipped_count}")
    else:
        print(f"Successfully reformatted: {success_count}")
        print(f"Already standard: {skipped_count}")
        print(f"Errors: {error_count}")
    
    if error_count > 0:
        print("\nFiles with errors:")
        for r in results:
            if r['status'] == 'error':
                print(f"  - {Path(r['file']).name}: {r['errors'][0]}")
    
    print("="*80)
    
    if args.dry_run:
        print("\nThis was a dry run. Use without --dry-run to apply changes.")
    
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

