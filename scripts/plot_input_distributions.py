"""Plot histograms of the tokenized particle features for a set of jet types.

Reads all tokenized parquet files per jet type and concatenates them, extracts
the `particle_features_tokenized` group, and plots a histogram (overlaid
across jet types) for each feature, saving the figures to disk.
"""

import argparse
import glob
import logging
import os
import random
import sys

import awkward as ak
import heputl.plotting.oneD as heplt

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from gabbro.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# maps convenient short names to the actual JetClass file-name prefixes
JET_TYPE_TO_FILE_PREFIX = [
    "HToBB",
    "HToCC",
    "HToGG",
    "HToWW2Q1L",
    "HToWW4Q",
    "TTBar",
    "TTBarLep",
    "WToQQ",
    "ZJetsToNuNu",
    "ZToQQ",
]

DEFAULT_DATA_DIR = (
    "/srv/beegfs/scratch/groups/rodem/lumea/datasets/JetClass/Pythia/"
    "cont_alfa_tokenized_allfeat/2026-08-04_14-03-40_gpu024_EquestrianSymbol/train_100M"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--jet_types",
    type=str,
    default=",".join(JET_TYPE_TO_FILE_PREFIX),
    help="Comma-separated list of jet types to plot.",
)
parser.add_argument(
    "--data_dir",
    type=str,
    default=DEFAULT_DATA_DIR,
    help="Directory containing the tokenized parquet files.",
)
parser.add_argument(
    "--max_files_per_type",
    type=int,
    default=40,
    help="Max number of files to read per jet type, randomly sampled from the file list.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed used to sample files when --max_files_per_type is set.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="analysis/backbone_inputs",
    help="Directory to save the plots to.",
)

parser.add_argument(
    "--bg_name",
    type=str,
    default=None,
    help="Optional jet type (must be in --jet_types) to draw as the filled background histogram.",
)


def main(
    jet_types: list,
    data_dir: str,
    output_dir: str,
    max_files_per_type: int = None,
    seed: int = 42,
    bg_name: str = None,
):
    """Read tokenized particle features for a set of jet types and plot per-feature histograms.

    Parameters
    ----------
    jet_types : list of str
        Jet types to load (keys of JET_TYPE_TO_FILE_PREFIX).
    data_dir : str
        Directory containing the tokenized parquet files.
    output_dir : str
        Directory to save the plots to.
    max_files_per_type : int, optional
        Max number of files to read per jet type, randomly sampled from the file list.
        If None, all files are used.
    seed : int, optional
        Random seed used to sample files when `max_files_per_type` is set.
    deps_path : str, optional
        Path to add to sys.path to make `heputl` importable.
    bg_name : str, optional
        Jet type (must be in `jet_types`) to draw as the filled background histogram.
    """
    if bg_name is not None and bg_name not in jet_types:
        raise ValueError(f"bg_name '{bg_name}' must be one of jet_types {jet_types}")

    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    arrays = {}
    for jet_type in jet_types:
        fps = sorted(glob.glob(os.path.join(data_dir, f"{jet_type}_*_tokenized.parquet")))
        if not fps:
            raise FileNotFoundError(f"No files found for jet type '{jet_type}' in {data_dir}")
        if max_files_per_type is not None and len(fps) > max_files_per_type:
            fps = rng.sample(fps, max_files_per_type)
        log.info(f"Loading {jet_type} from {len(fps)} files")
        arr = ak.concatenate([ak.from_parquet(fp) for fp in fps])
        arrays[jet_type] = arr.particle_features_tokenized

    fields = arrays[jet_types[0]].fields
    log.info(f"Plotting the following fields: {fields}")

    jet_types_suffix = "_".join(jet_types)

    for field in fields:
        log.info(f"Plotting feature '{field}'")
        data = [ak.to_numpy(ak.flatten(arrays[jet_type][field], axis=None)) for jet_type in jet_types]
        heplt.plot_feature_hist_for_n_samples(
            data=data,
            sample_names=jet_types,
            xlabel=field,
            bins=70,
            plot_name=f"{field}__{jet_types_suffix}",
            fig_dir=output_dir,
            show_plt=False,
            legend_outside=False,
            bg_name=bg_name,
            bg_alpha=0.3,
            fig_size=(6.0,4.5)
        )

    log.info(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        jet_types=args.jet_types.split(","),
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_files_per_type=args.max_files_per_type,
        bg_name=args.bg_name,
        seed=args.seed,
    )
    log.info("------------ Finished plotting. ------------")
