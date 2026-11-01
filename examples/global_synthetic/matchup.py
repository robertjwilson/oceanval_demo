"""Match the synthetic global model up with the synthetic observations.

Every observational dataset here is a *synthetic stand-in* for one of OceanVal's built-in
global recipes (WOA23, COBE2, OCCCI, GLODAPv2.2016b): same variable names, files and units,
but made by make_synthetic_data.py and read from disk, so nothing is downloaded.
The recipes' own titles and source text are reused so the report reads as it would for
the real datasets.

Everything is surface-only, except WOA23 temperature, which is validated through the
water column (vertical=True).

Run from any directory; the matchups go to ./oceanval_matchups.
"""

import glob
import os

import oceanval
from oceanval.parsers import find_recipe

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OBS = os.path.join(DATA, "obs")
START, END = 2000, 2004

# (variable, recipe, model variable, files of the synthetic observations, vertical, obs_multiplier)
# The model reports nutrients and oxygen in mmol m-3 and WOA23 in micromoles per kilogram,
# so those observations are multiplied by 1.025 kg m-3.
RHO = 1.025


def files(pattern):
    """A glob pattern (which add_gridded_comparison takes as it is) for local observation files."""
    pattern = os.path.join(OBS, pattern)
    assert glob.glob(pattern), f"no synthetic observations match {pattern}: run make_synthetic_data.py first"
    return pattern


COMPARISONS = [
    ("temperature", "cobe2", "thetao", os.path.join(OBS, "cobe2", "sst.mon.mean.nc"), False, 1),
    ("temperature", "woa23", "thetao", files("woa23/temperature/*.nc"), True, 1),
    ("salinity", "woa23", "so", files("woa23/salinity/*.nc"), False, 1),
    ("nitrate", "woa23", "no3", files("woa23/nitrate/*.nc"), False, RHO),
    ("phosphate", "woa23", "po4", files("woa23/phosphate/*.nc"), False, RHO),
    ("silicate", "woa23", "si", files("woa23/silicate/*.nc"), False, RHO),
    ("oxygen", "woa23", "o2", files("woa23/oxygen/*.nc"), False, RHO),
    ("chlorophyll", "occci", "chl", files("occci/chlor_a/*/*.nc"), False, 1),
    ("kd490", "occci", "kd490", files("occci/kd/*/*.nc"), False, 1),
    ("alkalinity", "glodap", "talk", os.path.join(OBS, "glodap", "GLODAPv2.2016b.TAlk.nc"), False, 1),
    ("ph", "glodap", "ph", os.path.join(OBS, "glodap", "GLODAPv2.2016b.pHtsinsitutp.nc"), False, 1),
]


PROVIDER = "Zissou Oceanographic"
# a source name becomes part of file names that CDO is given, so it cannot hold a space
KEY = PROVIDER.replace(" ", "")
# every comparison of one variable needs its own source name, so temperature's three are told apart
SOURCE_NAMES = {
    ("temperature", "cobe2"): f"{KEY}SST",
    ("temperature", "woa23"): f"{KEY}Climatology",
}


def register():
    oceanval.reset()
    for variable, recipe, model_variable, obs_path, vertical, multiplier in COMPARISONS:
        info = find_recipe({variable: recipe}, start=START, end=END)
        oceanval.add_gridded_comparison(
            name=info["name"],
            long_name=info["long_name"],
            short_name=info["short_name"],
            short_title=info["short_title"],
            source=SOURCE_NAMES.get((variable, recipe), KEY),
            source_info=f"Synthetic data from {PROVIDER}, made by make_synthetic_data.py to stand in for {info['source']}. Not real observations.",
            obs_variable=info["obs_variable"],
            climatology=info["climatology"],
            model_variable=model_variable,
            obs_path=obs_path,
            vertical=vertical,
            obs_multiplier=multiplier,
            thredds=False,
        )


def register_points():
    """Synthetic Argo/CTD-like temperature profiles, validated through the water column."""
    info = find_recipe({"temperature": "cobe2"})
    oceanval.add_point_comparison(
        name="temperature",
        long_name=info["long_name"],
        short_name=info["short_name"],
        short_title=info["short_title"],
        source=f"{KEY}Profiles",
        source_info=f"Synthetic Argo/CTD-like temperature profiles from {PROVIDER}, made by make_synthetic_data.py. Not real observations.",
        model_variable="thetao",
        obs_path=os.path.join(OBS, "point", "synthetic_profiles.csv"),
        vertical=True,
        start=START,
        end=END,
        point_time_res=["year", "month"],
    )


if __name__ == "__main__":
    register()
    register_points()
    oceanval.matchup(
        sim_dir=os.path.join(DATA, "model"),
        start=START,
        end=END,
        n_dirs_down=2,
        thickness="z_level",
        cores=4,
        ask=False,
        overwrite=True,
    )
