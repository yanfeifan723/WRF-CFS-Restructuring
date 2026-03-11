# 数据说明文档（WRF-CFS 处理输出）

## 输出文件
- `WRF_CFS_processed/ght_850hPa_JJA.nc`
- `WRF_CFS_processed/ght_200hPa_JJA.nc`
- `WRF_CFS_processed/u_850hPa_JJA.nc`
- `WRF_CFS_processed/u_200hPa_JJA.nc`
- `WRF_CFS_processed/v_850hPa_JJA.nc`
- `WRF_CFS_processed/v_200hPa_JJA.nc`
- `WRF_CFS_processed/rain_JJA.nc`

## 坐标与维度
- 维度：`time` × `lat` × `lon`
- `time`：JJA 季节平均的代表日期 `YYYY-07-01`，按年份递增（当前含 2012、2013）
- `lat`：49 点，递减排序（北到南），范围约 [48, 0]
- `lon`：71 点，递增排序（西到东），范围约 [0, 70]（对应原域 [70, 140] 经过 0 基平移表示）

## 变量与单位
- `ght`：位势高度（m），850/200 hPa
- `u`：纬向风（m s-1），850/200 hPa
- `v`：经向风（m s-1），850/200 hPa
- `rain`：降水（mm），JJA 时段平均的累积量场
- 填充值：缺测为 `NaN`

## 网格与处理
- 源：WRF 输出（曲线网格，Lambert 投影）
- 目标：规则经纬度 1° 网格，域近似东亚扩展区
- 插值：优先 xarray 线性插值，异常时回退最近邻；保持时间维度
- 时间聚合：JJA 季节平均（使用原始累积量的时间平均）
- 压缩：NetCDF4，`zlib=True`，`complevel=1`

## 快速查看示例
```python
import xarray as xr
ds = xr.open_dataset("WRF_CFS_processed/ght_850hPa_JJA.nc")
print(ds)                    # 查看维度与坐标
print(ds.ght.mean().item())  # 全场均值
```

## 追加与再处理
- 主脚本会检查已存在年份并跳过；新年份会按同一坐标顺序追加
- 如需重新生成，删除目标文件后再运行 `process_wrf_cfs.py --config config.yaml`
