#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Final production WRF-CFS processing script
- Variable–pressure-layer → single file
- Append by year with consistency checks
- Thread-based year parallelism (safe for netCDF)
"""

import os
import sys
import yaml
import glob
import time
import tempfile
import logging
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from filelock import FileLock


# =========================
# Config & logging
# =========================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(outdir, level):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("WRF-CFS")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    # 优化：禁用缓冲，确保日志及时输出
    ch.setLevel(logging.NOTSET)
    logger.addHandler(ch)

    fh = logging.FileHandler(outdir / "wrf_cfs_processing.log")
    fh.setFormatter(fmt)
    # 优化：禁用缓冲，确保日志及时写入
    fh.setLevel(logging.NOTSET)
    logger.addHandler(fh)

    return logger


# =========================
# Utilities
# =========================

def build_target_grid(domain, resolution):
    lats = np.arange(domain["lat_min"], domain["lat_max"] + 1e-6, resolution)
    lons = np.arange(domain["lon_min"], domain["lon_max"] + 1e-6, resolution)
    lon2d, lat2d = np.meshgrid(lons, lats)
    return lats, lons, lat2d, lon2d


def parse_times(ds):
    if "Times" not in ds:
        return None
    times = []
    for t in ds["Times"].values:
        # Times字段可能是numpy.bytes_或字符数组
        if isinstance(t, (bytes, np.bytes_)):
            s = t.decode('ascii', errors='ignore').strip()
        elif isinstance(t, np.ndarray):
            # 如果是字符数组，尝试直接解码或拼接
            if t.dtype.kind in ['U', 'S']:
                try:
                    s = t.tobytes().decode('ascii', errors='ignore').strip()
                except:
                    s = "".join([c.decode() if isinstance(c, bytes) else str(c) for c in t])
            else:
                s = "".join([c.decode() if isinstance(c, bytes) else str(c) for c in t])
        else:
            s = str(t)
        try:
            times.append(datetime.strptime(s, "%Y-%m-%d_%H:%M:%S"))
        except Exception:
            times.append(None)
    return times


def choose_pressure_index(ds, target_hpa):
    target_pa = target_hpa * 100.0
    if "P_PL" not in ds:
        return None
    p = ds["P_PL"].isel(Time=0).values.ravel()
    idx = np.argmin(np.abs(p - target_pa))
    return idx


def interp_to_grid(data, src_lat, src_lon, tgt_lat2d, tgt_lon2d):
    pts = np.column_stack((src_lat.ravel(), src_lon.ravel()))
    vals = data.ravel()
    mask = np.isfinite(vals)
    out = griddata(
        pts[mask],
        vals[mask],
        (tgt_lat2d, tgt_lon2d),
        method="linear",
    )
    if np.all(np.isnan(out)):
        out = griddata(
            pts[mask],
            vals[mask],
            (tgt_lat2d, tgt_lon2d),
            method="nearest",
        )
    return out


# =========================
# Output (append by year)
# =========================

def append_year_file(
    outdir,
    varname,
    year,
    data2d,
    lat,
    lon,
    season,
    pressure=None,
    logger=None,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if pressure is None:
        fname = f"{varname}_{season}.nc"
    else:
        fname = f"{varname}_{pressure}hPa_{season}.nc"

    fpath = outdir / fname
    lock = FileLock(str(fpath) + ".lock", timeout=600)

    ds_new = xr.Dataset(
        {varname: (("year", "lat", "lon"), data2d[np.newaxis])},
        coords={"year": [year], "lat": lat, "lon": lon},
    )

    with lock:
        if not fpath.exists():
            tmp = tempfile.NamedTemporaryFile(dir=outdir, delete=False)
            tmp.close()
            ds_new.to_netcdf(tmp.name)
            os.replace(tmp.name, fpath)
            logger.info(f"Created {fname}")
            return

        ds_old = xr.open_dataset(fpath, engine='netcdf4')

        if not (
            np.array_equal(ds_old.lat, lat)
            and np.array_equal(ds_old.lon, lon)
        ):
            raise RuntimeError(f"Grid mismatch in {fname}")

        if year in ds_old.year.values:
            logger.warning(f"{fname}: year {year} exists, skip")
            ds_old.close()
            return

        ds_all = xr.concat([ds_old, ds_new], dim="year")
        ds_old.close()

        tmp = tempfile.NamedTemporaryFile(dir=outdir, delete=False)
        tmp.close()
        ds_all.to_netcdf(tmp.name)
        os.replace(tmp.name, fpath)

        logger.info(f"Appended {year} → {fname}")


# =========================
# Core processing per year
# =========================

def process_one_year(
    year,
    variable,
    cfg,
    lat,
    lon,
    lat2d,
    lon2d,
    logger,
):
    # 优化：直接搜索包含年份的路径，避免搜索所有文件
    year_str = str(year)
    input_path = Path(cfg["input_path"])
    
    # 先尝试在年份目录中搜索
    files = []
    for pattern_dir in input_path.rglob(f"*{year_str}*"):
        if pattern_dir.is_dir():
            files.extend(sorted(pattern_dir.glob(variable["file_pattern"])))
    
    # 如果没找到，回退到全路径搜索
    if not files:
        all_files = sorted(input_path.rglob(variable["file_pattern"]))
        files = [f for f in all_files if year_str in f.name]
    
    if not files:
        logger.warning(f"{variable['name']} {year}: no files found")
        return
    
    logger.info(f"{variable['name']} {year}: found {len(files)} files, processing...")
    
    acc = None
    cnt = None
    file_count = 0
    
    # 优化：缓存坐标和压力索引（同一年的所有文件使用相同的网格）
    src_lat = None
    src_lon = None
    pidx = None
    # 优化：使用chunks参数优化内存使用（如果配置了chunking）
    chunks = cfg.get("chunking", None)

    for f in files:
        try:
            # 优化：根据文件类型选择合适的引擎
            # 检查可用引擎，优先使用netcdf4（如果可用），否则直接使用scipy
            available_engines = list(xr.backends.list_engines().keys())
            engine = None
            open_kwargs = {'decode_times': False}
            
            # 优先尝试netcdf4引擎（如果可用）
            if 'netcdf4' in available_engines:
                engine = 'netcdf4'
                open_kwargs['engine'] = engine
                if chunks:
                    open_kwargs['chunks'] = chunks
            else:
                # 如果没有netcdf4，直接使用scipy
                engine = 'scipy'
                open_kwargs['engine'] = engine
            
            # 尝试打开文件
            try:
                ds = xr.open_dataset(f, **open_kwargs)
            except (ValueError, OSError, Exception) as e:
                # 如果netcdf4失败，尝试scipy引擎
                error_msg = str(e).lower()
                if (engine == 'netcdf4' and 
                    ('unrecognized engine' in error_msg or 
                     'not a valid id' in error_msg or
                     'not a valid netcdf 3 file' in error_msg or
                     'module' in error_msg)):
                    # 回退到scipy引擎
                    engine = 'scipy'
                    open_kwargs = {'decode_times': False, 'engine': engine}
                    # scipy引擎不支持chunks参数
                    try:
                        ds = xr.open_dataset(f, **open_kwargs)
                    except Exception as e2:
                        # 如果scipy也失败，记录错误但继续处理其他文件
                        logger.warning(f"{variable['name']} {year}: Cannot read {f.name} with any engine. Skipping. Error: {e2}")
                        continue
                else:
                    # 其他类型的错误，记录并跳过
                    logger.warning(f"{variable['name']} {year}: Error reading {f.name}: {e}. Skipping.")
                    continue
            times = parse_times(ds)
            if times is None:
                ds.close()
                continue

            # 优化：只在第一个文件读取坐标和压力索引
            if src_lat is None:
                src_lat = ds["XLAT"].isel(Time=0).values
                src_lon = ds["XLONG"].isel(Time=0).values
                if variable["has_pressure_levels"]:
                    pidx = choose_pressure_index(ds, cfg["pressure_levels"][0])

            for i, t in enumerate(times):
                if t is None or t.month not in cfg["season_months"]:
                    continue

                if variable["has_pressure_levels"]:
                    da = ds[variable["name"]].isel(
                        Time=i, num_press_levels_stag=pidx
                    )
                else:
                    # 处理RAINNC的combine_with逻辑
                    if variable["name"] == "RAINNC" and "combine_with" in variable:
                        rain_nc = ds[variable["name"]].isel(Time=i)
                        if variable["combine_with"] in ds:
                            rain_c = ds[variable["combine_with"]].isel(Time=i)
                            da = rain_nc + rain_c
                        else:
                            da = rain_nc
                    else:
                        da = ds[variable["name"]].isel(Time=i)

                # 优化：确保数据已加载到内存（如果使用chunks）
                if hasattr(da, 'compute'):
                    da_values = da.compute().values
                else:
                    da_values = da.values

                interp = interp_to_grid(
                    da_values, src_lat, src_lon, lat2d, lon2d
                )

                if acc is None:
                    acc = np.zeros_like(interp)
                    cnt = np.zeros_like(interp)

                m = np.isfinite(interp)
                acc[m] += interp[m]
                cnt[m] += 1

            ds.close()
            file_count += 1
            
            if file_count % 10 == 0:
                logger.info(f"{variable['name']} {year}: processed {file_count}/{len(files)} files")
                # 优化：刷新日志，确保及时输出
                for handler in logger.handlers:
                    handler.flush()
                
        except Exception as e:
            logger.error(f"{variable['name']} {year}: error processing {f}: {e}")
            continue

    if acc is None:
        logger.warning(f"{variable['name']} {year}: no valid data accumulated")
        return

    mean = np.where(cnt > 0, acc / cnt, np.nan)
    logger.info(f"{variable['name']} {year}: completed, writing output...")

    append_year_file(
        cfg["output_path"],
        variable["output_name"],
        year,
        mean,
        lat,
        lon,
        "JJA",
        None if not variable["has_pressure_levels"] else cfg["pressure_levels"][0],
        logger,
    )
    
    logger.info(f"{variable['name']} {year}: finished")


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logger(
        cfg["output_path"], cfg["logging"]["level"]
    )

    lat, lon, lat2d, lon2d = build_target_grid(
        cfg["domain"], cfg["target_grid"]["resolution"]
    )

    for var in cfg["variables"]:
        logger.info(f"=== Processing {var['name']} ===")

        with ThreadPoolExecutor(
            max_workers=cfg["parallel"]["max_year_workers"]
        ) as pool:
            futures = [
                pool.submit(
                    process_one_year,
                    y,
                    var,
                    cfg,
                    lat,
                    lon,
                    lat2d,
                    lon2d,
                    logger,
                )
                for y in cfg["years"]
            ]

            for f in as_completed(futures):
                f.result()

    logger.info("All processing finished.")


if __name__ == "__main__":
    main()
