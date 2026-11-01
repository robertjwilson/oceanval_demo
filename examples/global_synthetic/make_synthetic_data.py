"""Make the synthetic data for the global OceanVal example.

EVERYTHING HERE IS SYNTHETIC. Nothing is a measurement or the output of a real model.
The fields are built from simple formulae of latitude, longitude, depth and season
(plus smooth random fields) so that the maps *look* like the real ocean: warm tropics,
subtropical salinity maxima, nutrient-rich high latitudes and upwelling zones, empty
gyres, western-boundary-current warm tongues, and so on, with the real coastlines.

It writes, under ``data/``:

* ``land_mask.nc``                      1 degree land mask (committed, made once from real coastlines)
* ``model/YYYY/MM/global_YYYYMM.nc``    the "model": monthly, 2000-2004
* ``obs/``                              "observations" in the layout and variable names of the
                                        built-in datasets (WOA23, COBE2, OCCCI, GLODAPv2.2016b)

The "truth" is one set of formulae. The model is the truth plus smooth, variable-specific
biases and its own noise; each observational dataset is the truth plus independent noise
and gaps, so the validation has something to find.

Run ``python make_synthetic_data.py`` (``--rebuild-mask`` to redo the land mask, which needs
the ``global-land-mask`` package). The output is deterministic.
"""

import argparse
import os

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SEED = 20260101
YEARS = list(range(2000, 2005))
WOA_PERIOD = "95A4"  # WOA23's 1995-2004 decade, the one 2000-2004 sits inside

LAT = np.arange(-89.5, 90, 1.0)
LON = np.arange(-179.5, 180, 1.0)
LON2, LAT2 = np.meshgrid(LON, LAT)
# model levels (m) for temperature; WOA23 starts at 0 m
MODEL_DEPTHS = np.array([5.0, 15, 30, 50, 100, 200, 400, 700])
WOA_DEPTHS = np.array([0.0, 10, 30, 50, 100, 200, 400, 700])

# --------------------------------------------------------------------------- land


def build_land_mask():
    from global_land_mask import globe

    land = globe.is_land(LAT2, LON2)
    # lakes and inland seas the package calls land are fine; keep it simple
    ds = xr.Dataset(
        {"ocean": (("lat", "lon"), (~land).astype("i1"), {"long_name": "1 where ocean"})},
        coords={"lat": ("lat", LAT, {"units": "degrees_north"}), "lon": ("lon", LON, {"units": "degrees_east"})},
    )
    os.makedirs(DATA, exist_ok=True)
    ds.to_netcdf(os.path.join(DATA, "land_mask.nc"))
    return ~land


def load_land():
    path = os.path.join(DATA, "land_mask.nc")
    if not os.path.exists(path):
        return ~build_land_mask()
    with xr.open_dataset(path) as ds:
        return ds["ocean"].values == 0


LAND = load_land()
OCEAN = ~LAND


def _shifted_land(direction, width=6):
    """Fraction of the `width` cells to the east (+1) or west (-1) that are land."""
    return np.mean([np.roll(LAND, -direction * k, axis=1) for k in range(1, width + 1)], axis=0)


LAND_EAST = gaussian_filter(_shifted_land(+1), 1.0)  # ocean next to a coast to its east: eastern boundary
LAND_WEST = gaussian_filter(_shifted_land(-1), 1.0)  # ocean next to a coast to its west: western boundary
# distance (cells) to land, for coastal effects
from scipy.ndimage import distance_transform_edt  # noqa: E402

COAST = distance_transform_edt(OCEAN)

# --------------------------------------------------------------------------- helpers

S = np.sin(np.radians(LAT2))
ABS_LAT = np.abs(LAT2)
HEMI = np.tanh(LAT2 / 8.0)  # +1 north, -1 south
PACIFIC = np.exp(-(((LON2 - 180 + 360) % 360 - 180) / 70.0) ** 2) * 0 + (
    ((LON2 > 140) | (LON2 < -75)).astype(float)
)
PACIFIC = gaussian_filter(PACIFIC, 3, mode="wrap")
ATLANTIC = gaussian_filter(((LON2 > -80) & (LON2 < 20)).astype(float), 3, mode="wrap")


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def season(month):
    """+1 at the northern summer peak (August) and -1 in February, for month 0-11."""
    return np.cos(2 * np.pi * ((month + 0.5) / 12 - 0.62))


def smooth_noise(rng, sigma, amp):
    """A smooth random field with standard deviation amp."""
    field = gaussian_filter(rng.standard_normal(LAT2.shape), sigma, mode=("nearest", "wrap"))
    return amp * field / field.std()


def mask(field):
    return np.where(OCEAN, field, np.nan)


# --------------------------------------------------------------------------- the truth
# Each function gives the field for a month (0-11) as an (lat, lon) array. "anom" is the
# interannual anomaly (an ENSO-like index and a slow warming).


def warming(year):
    return 0.02 * (year - YEARS[0])


def enso(year, rng_index):
    return rng_index[year - YEARS[0]]


ENSO_PATTERN = np.exp(-((LAT2 / 12.0) ** 2)) * np.cos(np.radians((LON2 + 360) % 360 - 200) * 1.3)


def sst_mean():
    base = 29.5 - 31.0 * S**2
    warm_pool = 2.0 * np.exp(-(((LON2 - 140) / 35.0) ** 2) - (LAT2 / 15.0) ** 2)
    cold_tongue = -3.0 * np.exp(-(((LON2 + 100) / 30.0) ** 2) - (LAT2 / 8.0) ** 2)
    eastern = -3.0 * LAND_EAST * np.exp(-((ABS_LAT - 20) ** 2) / (2 * 12**2))
    western = 2.5 * LAND_WEST * np.exp(-((ABS_LAT - 35) ** 2) / (2 * 10**2))
    return np.clip(base + warm_pool + cold_tongue + eastern + western, -1.8, 32)


SST_MEAN = sst_mean()


def sst(month, anom=0.0):
    amp = 0.5 + 14.0 * S**2 * (1 - S**2)
    return np.maximum(SST_MEAN + HEMI * season(month) * amp + anom * ENSO_PATTERN, -1.8)


def temperature(month, depth, anom=0.0):
    """Temperature (degC) at a depth: a thermocline whose scale is deeper in the subtropical gyres."""
    mean = SST_MEAN + anom * ENSO_PATTERN
    seasonal = HEMI * season(month) * (0.5 + 14.0 * S**2 * (1 - S**2))
    deep = np.minimum(mean, 4.0)
    scale = 70 + 260 * np.exp(-((ABS_LAT - 28) / 22.0) ** 2)
    vertical = np.exp(-depth / scale)
    return np.maximum(deep + (mean - deep) * vertical + seasonal * np.exp(-depth / 40.0), -1.8)


def salinity(month, anom=0.0):
    lat = LAT2
    s = 35.0
    s = s + 1.6 * np.exp(-((ABS_LAT - 23) / 12.0) ** 2)
    s = s - 1.2 * np.exp(-((lat - 5) / 8.0) ** 2)
    s = s - np.clip(ABS_LAT - 48, 0, None) * 0.12
    s = s + 1.0 * ATLANTIC * sig((52 - ABS_LAT) / 4.0) - 0.5 * PACIFIC
    s = s - 2.0 * np.exp(-COAST / 4.0) * np.exp(-((lat / 22.0) ** 2))  # river plumes, mostly tropical
    s = s - 0.15 * season(month) * HEMI * np.exp(-((ABS_LAT - 55) / 12.0) ** 2)
    return s - 0.3 * anom * ENSO_PATTERN


def nitrate(month, anom=0.0):
    """umol/kg."""
    polar = sig((ABS_LAT - 38) / 7.0)
    summer = HEMI * season(month)  # +1 in the local... northern summer; use hemisphere-aware sign
    local_summer = summer
    equatorial = 5.0 * np.exp(-((LAT2 / 6.0) ** 2)) * (0.4 + 0.6 * PACIFIC)
    upwell = 6.0 * LAND_EAST * np.exp(-((ABS_LAT - 20) ** 2) / (2 * 10**2))
    n = 0.2 + 26.0 * polar + equatorial + upwell
    n = n * (1 - 0.35 * np.clip(local_summer, 0, None) * polar)  # summer drawdown
    n = n * (1 + 0.15 * np.clip(-local_summer, 0, None) * polar)
    return np.clip(n + 0.8 * anom * ENSO_PATTERN * (-1), 0.02, None)


def phosphate(month, anom=0.0):
    return np.clip(nitrate(month, anom) / 15.0 + 0.08 * (ABS_LAT > 40), 0.02, None)


def silicate(month, anom=0.0):
    southern = 60.0 * sig((-LAT2 - 48) / 5.0)
    n_pacific = 14.0 * sig((LAT2 - 38) / 6.0) * PACIFIC
    equatorial = 6.0 * np.exp(-((LAT2 / 7.0) ** 2)) * PACIFIC
    return np.clip(2.0 + southern + n_pacific + equatorial + 4.0 * LAND_EAST * (ABS_LAT < 35), 0.5, None)


def oxygen(month, anom=0.0):
    """umol/kg, close to saturation."""
    t = sst(month, anom)
    return 352.0 - 5.2 * np.clip(t, 0, 30) + 4.0 * np.clip(season(month) * HEMI, -1, 0) * -1


def chlorophyll(month, anom=0.0):
    """mg m-3."""
    local_summer = HEMI * season(month)
    bloom = 1.0 + 1.6 * np.clip(local_summer, 0, None) * sig((ABS_LAT - 35) / 6.0)
    chl = 0.06 + 0.8 * sig((ABS_LAT - 42) / 7.0) * bloom
    chl = chl + 0.22 * np.exp(-((LAT2 / 5.0) ** 2)) * (0.5 + 0.5 * PACIFIC)
    chl = chl + 0.5 * np.exp(-COAST / 3.0) + 0.45 * LAND_EAST * np.exp(-((ABS_LAT - 22) ** 2) / (2 * 10**2))
    return np.clip(chl * (1 - 0.25 * anom * ENSO_PATTERN), 0.015, 30)


def kd490(month, anom=0.0):
    """m-1, from chlorophyll (Morel-style)."""
    return 0.0166 + 0.0773 * chlorophyll(month, anom) ** 0.67


def alkalinity(month, anom=0.0):
    """umol/kg, mostly a function of salinity."""
    return 66.4 * salinity(month, anom) - 24.0 + 18.0 * sig((-LAT2 - 40) / 5.0) + 8.0 * sig((LAT2 - 45) / 5.0)


def ph(month, anom=0.0):
    """Total scale."""
    return 8.17 - 0.0055 * sst(month, anom) - 0.05 * LAND_EAST * np.exp(-((ABS_LAT - 20) ** 2) / (2 * 12**2)) + 0.01 * (
        season(month) * HEMI * 0.5
    )


# --------------------------------------------------------------------------- model biases
# (additive unless noted). Smooth random parts come from one rng per variable.


def model_field(name, month, depth, rng_fields, anom, clim_only=False):
    """The model's version of `name`, in the model's own units."""
    r = rng_fields[name]
    if name == "thetao":
        t = temperature(month, depth, anom)
        bias = (0.9 * np.cos(np.radians(LAT2)) ** 2 - 0.6 + r["smooth"]) * np.exp(-depth / 300.0)
        return np.maximum(t + bias + r["noise"] * 0.15, -1.9)
    if name == "so":
        return salinity(month, anom) + 0.25 * ATLANTIC - 0.15 + r["smooth"] + r["noise"] * 0.03
    if name == "no3":  # mmol m-3
        n = nitrate(month, anom) * 1.025
        return np.clip(n * (1 + 0.3 * sig((-LAT2 - 45) / 6.0) - 0.1 + r["smooth"]) + r["noise"] * 0.05, 0.01, None)
    if name == "po4":
        p = phosphate(month, anom) * 1.025
        return np.clip(p * (1 + 0.2 * sig((-LAT2 - 40) / 6.0) + r["smooth"]) + r["noise"] * 0.005, 0.01, None)
    if name == "si":
        s = silicate(month, anom) * 1.025
        return np.clip(s * (1 - 0.2 * sig((-LAT2 - 50) / 6.0) + r["smooth"]) + r["noise"] * 0.1, 0.1, None)
    if name == "o2":
        return oxygen(month, anom) * 1.025 + 6.0 + r["smooth"] + r["noise"] * 0.6
    if name == "chl":
        c = chlorophyll(month, anom)
        return np.clip(c * np.exp(r["smooth"] - 0.1 * (ABS_LAT < 25) + 0.25 * sig((ABS_LAT - 45) / 5.0)) * np.exp(r["noise"] * 0.08), 0.01, None)
    if name == "kd490":
        return 0.0166 + 0.0773 * model_field("chl", month, depth, rng_fields, anom) ** 0.67 + r["noise"] * 0.0005
    if name == "talk":
        return alkalinity(month, anom) * 1.0 + 6.0 + r["smooth"] + r["noise"] * 1.0
    if name == "ph":
        return ph(month, anom) - 0.012 + r["smooth"] + r["noise"] * 0.0015
    raise KeyError(name)


# per variable: smooth bias amplitude (sigma cells, amplitude) in model units
MODEL_SMOOTH = {
    "thetao": (8, 0.5), "so": (8, 0.2), "no3": (8, 0.12), "po4": (8, 0.1), "si": (8, 0.12),
    "o2": (8, 5.0), "chl": (6, 0.18), "kd490": (6, 0.0), "talk": (8, 8.0), "ph": (8, 0.01),
}
# the model: name, long_name, units, standard_name
MODEL_VARIABLES = [
    ("so", "sea water salinity", "1"),
    ("no3", "dissolved nitrate concentration", "mmol m-3"),
    ("po4", "dissolved phosphate concentration", "mmol m-3"),
    ("si", "dissolved silicate concentration", "mmol m-3"),
    ("o2", "dissolved oxygen concentration", "mmol m-3"),
    ("chl", "mass concentration of chlorophyll in sea water", "mg m-3"),
    ("ph", "sea water ph reported on total scale", "1"),
    ("talk", "total alkalinity", "umol kg-1"),
    ("kd490", "diffuse attenuation coefficient at 490 nm", "m-1"),
]

# --------------------------------------------------------------------------- writing


def _encoding(variable_names, scale=None):
    out = {}
    for v in variable_names:
        out[v] = {"zlib": True, "complevel": 4, "shuffle": True, "_FillValue": np.float32(1e20), "dtype": "f4"}
    return out


def _pack(ds, variable_names, digits):
    """Pack floats as int16 with a scale factor, to keep files small."""
    enc = {}
    for v in variable_names:
        arr = ds[v].values
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        scale = (hi - lo) / 60000.0 or 1.0
        enc[v] = {
            "dtype": "int16", "scale_factor": scale, "add_offset": (hi + lo) / 2, "_FillValue": np.int16(-32768),
            "zlib": True, "complevel": 5, "shuffle": True,
        }
    return enc


def _coords(depths=None, time=None, time_units="days since 1900-01-01", tdata=None):
    coords = {
        "lat": ("lat", LAT.astype("f4"), {"units": "degrees_north", "long_name": "latitude", "standard_name": "latitude", "axis": "Y"}),
        "lon": ("lon", LON.astype("f4"), {"units": "degrees_east", "long_name": "longitude", "standard_name": "longitude", "axis": "X"}),
    }
    if depths is not None:
        coords["depth"] = ("depth", depths.astype("f4"), {"units": "m", "positive": "down", "long_name": "depth", "standard_name": "depth", "axis": "Z"})
    return coords


def _time_da(dates):
    return xr.DataArray(np.array(dates, dtype="datetime64[ns]"), dims="time", attrs={"long_name": "time", "axis": "T"})


def write(ds, path, packed, digits=3):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    enc = _pack(ds, packed, digits)
    ds.to_netcdf(path, encoding=enc, unlimited_dims=["time"])


def midmonth(year, month):  # month 1-12
    return np.datetime64(f"{year}-{month:02d}-15")


# One gridded noise bank per dataset, seeded separately so everything is reproducible.


def make_rng_fields(seed, names):
    rng = np.random.default_rng(seed)
    bank = {}
    for n in names:
        sigma, amp = MODEL_SMOOTH[n]
        bank[n] = {"smooth": smooth_noise(rng, sigma, amp) if amp else 0.0}
    return bank, rng


def enso_index():
    rng = np.random.default_rng(SEED + 7)
    idx = rng.standard_normal(len(YEARS))
    return 1.2 * idx / np.abs(idx).max()


def make_model():
    names = ["thetao"] + [v[0] for v in MODEL_VARIABLES]
    bank, rng = make_rng_fields(SEED + 1, names)
    index = enso_index()
    n = 0
    for year in YEARS:
        anom = index[year - YEARS[0]]
        for month in range(12):
            for name in names:
                bank[name]["noise"] = rng.standard_normal(LAT2.shape)
            data = {}
            warm = warming(year)
            depths = MODEL_DEPTHS
            temp = np.stack([model_field("thetao", month, d, bank, anom) + warm for d in depths])[None]
            data["thetao"] = (("time", "depth", "lat", "lon"), mask3(temp), {
                "long_name": "sea water potential temperature", "standard_name": "sea_water_potential_temperature", "units": "degC"})
            for name, long_name, units in MODEL_VARIABLES:
                field = model_field(name, month, 0.0, bank, anom)
                data[name] = (("time", "lat", "lon"), mask(field)[None].astype("f4"), {"long_name": long_name, "units": units})
            ds = xr.Dataset(data, coords={**_coords(depths), "time": _time_da([midmonth(year, month + 1)])})
            ds["time"].attrs.update({"standard_name": "time"})
            write(ds, os.path.join(DATA, "model", str(year), f"{month + 1:02d}", f"global_{year}{month + 1:02d}.nc"), list(data))
            n += 1
    print(f"model: {n} files")


def mask3(arr):
    return np.where(OCEAN[None, None], arr, np.nan).astype("f4")


def _obs_noise(seed, names):
    rng = np.random.default_rng(seed)
    return rng


def make_woa23():
    """Monthly climatologies, one file per month, in WOA23's variable names."""
    rng = np.random.default_rng(SEED + 2)
    smooth = {k: smooth_noise(rng, 5, 1.0) for k in "tsnoip"}
    specs = {
        # folder, file stem, variable, function, units, long_name, vertical?, abs noise
        "temperature": ("t", "t_an", lambda m, d: temperature(m, d), "degrees_celsius", 0.25, True),
        "salinity": ("s", "s_an", lambda m, d: salinity(m), "1", 0.05, False),
        "nitrate": ("n", "n_an", lambda m, d: nitrate(m), "micromoles_per_kilogram", 0.2, False),
        "phosphate": ("p", "p_an", lambda m, d: phosphate(m), "micromoles_per_kilogram", 0.02, False),
        "silicate": ("i", "i_an", lambda m, d: silicate(m), "micromoles_per_kilogram", 0.3, False),
        "oxygen": ("o", "o_an", lambda m, d: oxygen(m), "micromoles_per_kilogram", 2.5, False),
    }
    for name, (letter, var, func, units, noise, _vertical) in specs.items():
        period = WOA_PERIOD if name in ("temperature", "salinity") else "all"
        # data-sparse high latitudes are empty in winter
        for month in range(12):
            if name == "temperature":
                arr = np.stack([func(month, d) for d in WOA_DEPTHS])
                arr = arr + smooth[letter] * noise + rng.standard_normal(arr.shape) * noise * 0.4
                if True:
                    arr = np.where(OCEAN[None], arr, np.nan)
                dims = ("time", "depth", "lat", "lon")
                values = arr[None].astype("f4")
                depths = WOA_DEPTHS
            else:
                arr = func(month, 0.0) + smooth[letter] * noise + rng.standard_normal(LAT2.shape) * noise * 0.4
                dims = ("time", "depth", "lat", "lon")
                values = np.where(OCEAN, arr, np.nan)[None, None].astype("f4")
                depths = WOA_DEPTHS[:1]
            sparse = (ABS_LAT > 72) & (HEMI * season(month) < -0.2)
            values = np.where(sparse[None, None], np.nan, values)
            ds = xr.Dataset(
                {var: (dims, values, {"long_name": f"Objectively analyzed mean fields for {name} (synthetic)", "units": units})},
                coords={**_coords(depths), "time": _time_da([midmonth(1900, month + 1)])},
            )
            write(ds, os.path.join(DATA, "obs", "woa23", name, f"woa23_{period}_{letter}{month + 1:02d}_01.nc"), [var])
    print("woa23: done")


def make_cobe2():
    rng = np.random.default_rng(SEED + 3)
    index = enso_index()
    smooth = smooth_noise(rng, 4, 0.3)
    times, values = [], []
    for year in YEARS:
        for month in range(12):
            field = sst(month, index[year - YEARS[0]]) + warming(year) + smooth + rng.standard_normal(LAT2.shape) * 0.12
            values.append(np.where(OCEAN, field, np.nan))
            times.append(midmonth(year, month + 1))
    ds = xr.Dataset(
        {"sst": (("time", "lat", "lon"), np.stack(values).astype("f4"), {"long_name": "Monthly Means of Sea Surface Temperature (synthetic)", "units": "degC"})},
        coords={**_coords(), "time": _time_da(times)},
    )
    write(ds, os.path.join(DATA, "obs", "cobe2", "sst.mon.mean.nc"), ["sst"])
    print("cobe2: done")


def make_occci():
    rng = np.random.default_rng(SEED + 4)
    index = enso_index()
    smooth = smooth_noise(rng, 4, 0.12)
    for year in YEARS:
        for month in range(12):
            anom = index[year - YEARS[0]]
            chl = chlorophyll(month, anom) * np.exp(smooth + rng.standard_normal(LAT2.shape) * 0.18)
            kd = 0.0166 + 0.0773 * chlorophyll(month, anom) ** 0.67
            kd = kd * np.exp(smooth * 0.4 + rng.standard_normal(LAT2.shape) * 0.06)
            # no ocean colour in the polar night, and cloud/ice gaps
            gap = ((ABS_LAT > 55) & (HEMI * season(month) < -0.25)) | (ABS_LAT > 78) | (rng.random(LAT2.shape) < 0.08)
            for var, field, folder, stem, units, long_name in (
                ("chlor_a", chl, "chlor_a", "ESACCI-OC-L3S-CHLOR_A-MERGED-1M_MONTHLY_4km_GEO_PML_OCx", "milligram m-3", "Chlorophyll-a concentration in seawater (synthetic)"),
                ("kd_490", kd, "kd", "ESACCI-OC-L3S-K_490-MERGED-1M_MONTHLY_4km_GEO_PML_KD490_Lee", "m-1", "Downwelling attenuation coefficient at 490 nm (synthetic)"),
            ):
                values = np.where(OCEAN & ~gap, field, np.nan)[None].astype("f4")
                ds = xr.Dataset(
                    {var: (("time", "lat", "lon"), values, {"long_name": long_name, "units": units})},
                    coords={**_coords(), "time": _time_da([midmonth(year, month + 1)])},
                )
                write(ds, os.path.join(DATA, "obs", "occci", folder, str(year), f"{stem}-{year}{month + 1:02d}-fv6.0.nc"), [var])
    print("occci: done")


def make_glodap():
    rng = np.random.default_rng(SEED + 5)
    smooth_a, smooth_p = smooth_noise(rng, 5, 4.0), smooth_noise(rng, 5, 0.004)
    # annual mean of the monthly fields; gridded climatology, so one time step
    for var, func, smooth_field, noise, units, long_name in (
        ("TAlk", alkalinity, smooth_a, 3.0, "micro-mol kg-1", "Total alkalinity (synthetic)"),
        ("pHtsinsitutp", ph, smooth_p, 0.004, "total scale", "pH on total scale at in situ temperature and pressure (synthetic)"),
    ):
        mean = np.mean([func(m) for m in range(12)], axis=0)
        field = mean + smooth_field + rng.standard_normal(LAT2.shape) * noise
        field = np.where(OCEAN & (ABS_LAT < 80), field, np.nan)
        ds = xr.Dataset(
            {var: (("time", "depth", "lat", "lon"), field[None, None].astype("f4"), {"long_name": long_name, "units": units})},
            coords={**_coords(np.array([0.0])), "time": _time_da([np.datetime64("1900-07-01")])},
        )
        write(ds, os.path.join(DATA, "obs", "glodap", f"GLODAPv2.2016b.{var}.nc"), [var])
    print("glodap: done")


def make_point_profiles(n_profiles=7000):
    """Argo/CTD-like temperature profiles (csv: lon, lat, depth, year, month, day, observation).

    The values are the synthetic truth at the profile's place, month and depth plus
    instrument noise and a few spikes. No real profile is used."""
    import pandas as pd

    rng = np.random.default_rng(SEED + 6)
    index = enso_index()
    depths = np.array([2.0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 700])
    weight = np.cos(np.radians(LAT2)) * OCEAN * (1 + 1.5 * ATLANTIC + 1.0 * (LAT2 > 15)) * (ABS_LAT < 80)
    weight = weight / weight.sum()
    cells = rng.choice(weight.size, size=n_profiles, p=weight.ravel())
    iy, ix = np.unravel_index(cells, weight.shape)
    years = rng.integers(YEARS[0], YEARS[-1] + 1, n_profiles)
    months = rng.integers(0, 12, n_profiles)
    days = rng.integers(1, 29, n_profiles)
    # no profiles under winter sea ice
    keep = ~((ABS_LAT[iy, ix] > 65) & (HEMI[iy, ix] * season(months) < -0.2))
    iy, ix, years, months, days = iy[keep], ix[keep], years[keep], months[keep], days[keep]
    rows = []
    cache = {}
    for k in range(len(iy)):
        key = (years[k], months[k])
        if key not in cache:
            anom = index[years[k] - YEARS[0]]
            cache[key] = np.stack([temperature(months[k], d, anom) + warming(years[k]) for d in depths])
        profile = cache[key][:, iy[k], ix[k]]
        top = rng.integers(7, len(depths) + 1)  # how deep this profile goes
        lon = LON[ix[k]] + rng.uniform(-0.5, 0.5)
        lat = LAT[iy[k]] + rng.uniform(-0.5, 0.5)
        for d, v in zip(depths[:top], profile[:top]):
            v = v + rng.normal(0, 0.1)
            if rng.random() < 0.003:
                v += rng.normal(0, 2.0)
            rows.append((round(lon, 3), round(lat, 3), float(d), int(years[k]), int(months[k] + 1), int(days[k]), round(float(v), 3)))
    df = pd.DataFrame(rows, columns=["lon", "lat", "depth", "year", "month", "day", "observation"])
    path = os.path.join(DATA, "obs", "point", "synthetic_profiles.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"point: {len(df)} observations in {len(iy)} profiles")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rebuild-mask", action="store_true", help="remake data/land_mask.nc")
    args = parser.parse_args()
    if args.rebuild_mask:
        build_land_mask()
    make_model()
    make_woa23()
    make_cobe2()
    make_occci()
    make_glodap()
    make_point_profiles()


if __name__ == "__main__":
    main()
