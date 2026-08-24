"""Plot histograms of the (derived) particle features for the Aspen open jets dataset.

Reads all Aspen open-jets HDF5 files, computes the same derived particle-level
kinematics as in `scripts/playground/inspect_aspen_dataset.ipynb` (pt, p, eta,
phi, eta/phi relative to the jet axis, mass, PID flags), and saves a histogram
for each feature to disk.
"""

import argparse
import glob
import logging
import os
import random

import h5py
import heputl.plotting.oneD as heplt
import numpy as np

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from gabbro.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

DEFAULT_DATA_DIR = "/srv/beegfs/scratch/groups/rodem/lumea/datasets/aspenJets"

CONSTITS_FEATURE_NAMES = [
    "px",
    "py",
    "pz",
    "E",
    "d0",
    "d0Err",
    "dz",
    "dzErr",
    "charge",
    "PDG_ID",
    "PUPPI_weight",
]
JET_FEATURE_NAMES = ["pt", "eta", "phi", "softdrop_mass"]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    default=DEFAULT_DATA_DIR,
    help="Directory containing the Aspen open-jets HDF5 files.",
)
parser.add_argument(
    "--file_pattern",
    type=str,
    default="*.h5",
    help="Glob pattern (relative to --data_dir) used to find the HDF5 files.",
)
parser.add_argument(
    "--max_files",
    type=int,
    default=None,
    help="Max number of files to read, randomly sampled from the file list. Default: all files.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed used to sample files when --max_files is set.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="analysis/aspen_jets",
    help="Directory to save the plots to.",
)


def pid_flags(pdg_id):
    """Return a dict of boolean-as-int PID flags, broadcasting over any input shape."""
    a = np.abs(pdg_id).astype(np.int32)
    return {
        "part_isChargedHadron": (a == 211).astype(int),
        "part_isNeutralHadron": (a == 130).astype(int),
        "part_isPhoton": (a == 22).astype(int),
        "part_isElectron": (a == 11).astype(int),
        "part_isMuon": (a == 13).astype(int),
    }


def load_file(file_path):
    """Load the constituent and jet feature arrays from one Aspen open-jets HDF5 file."""
    with h5py.File(file_path, "r") as h5_file:
        constits = h5_file["PFCands"][:]
        jets = h5_file["jet_kinematics"][:]

    dtype = np.dtype([(name, constits.dtype) for name in CONSTITS_FEATURE_NAMES])
    constits_named = np.empty(constits.shape[:-1], dtype=dtype)
    for i, name in enumerate(CONSTITS_FEATURE_NAMES):
        constits_named[name] = constits[..., i]

    dtype = np.dtype([(name, jets.dtype) for name in JET_FEATURE_NAMES])
    jets_named = np.empty(jets.shape[:-1], dtype=dtype)
    for i, name in enumerate(JET_FEATURE_NAMES):
        jets_named[name] = jets[..., i]

    return constits_named, jets_named


def compute_features(constits_named, jets_named):
    """Compute the derived particle-level kinematics, following the playground notebook."""
    p = np.sqrt(
        constits_named["px"] ** 2 + constits_named["py"] ** 2 + constits_named["pz"] ** 2
    )
    phi = np.arctan2(constits_named["py"], constits_named["px"])
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(
            p > 0,
            np.arctanh(np.clip(constits_named["pz"] / np.where(p > 0, p, 1), -1 + 1e-8, 1 - 1e-8)),
            0.0,
        )
    m2 = constits_named["E"] ** 2 - (
        constits_named["px"] ** 2 + constits_named["py"] ** 2 + constits_named["pz"] ** 2
    )

    features = {
        "part_pt": np.sqrt(constits_named["px"] ** 2 + constits_named["py"] ** 2),
        "part_p": p,
        "part_eta": eta,
        "part_phi": phi,
        "part_etarel": np.where(
            jets_named["pt"][:, None] > 0,
            jets_named["eta"][:, None] - eta,
            -(jets_named["eta"][:, None] - eta),
        ),
        "part_phirel": (jets_named["phi"][:, None] - phi + np.pi) % (2 * np.pi) - np.pi,
        "part_mass": np.sqrt(np.clip(m2, 0, None)),
        "part_charge": constits_named["charge"],
        "part_d0val": constits_named["d0"],
        "part_d0err": constits_named["d0Err"],
        "part_dzval": constits_named["dz"],
        "part_dzerr": constits_named["dzErr"],
        "part_PUPPI_weight": constits_named["PUPPI_weight"],
    }
    features.update(pid_flags(constits_named["PDG_ID"]))
    return features


def main(data_dir, file_pattern, output_dir, max_files=None, seed=42):
    """Read all Aspen open-jets files, compute derived features and plot a histogram per feature.

    Parameters
    ----------
    data_dir : str
        Directory containing the Aspen open-jets HDF5 files.
    file_pattern : str
        Glob pattern (relative to `data_dir`) used to find the HDF5 files.
    output_dir : str
        Directory to save the plots to.
    max_files : int, optional
        Max number of files to read, randomly sampled from the file list.
        If None, all files are used.
    seed : int, optional
        Random seed used to sample files when `max_files` is set.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)

    file_paths = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
    if not file_paths:
        raise FileNotFoundError(f"No files found matching '{file_pattern}' in {data_dir}")
    if max_files is not None and len(file_paths) > max_files:
        file_paths = rng.sample(file_paths, max_files)
    log.info(f"Loading Aspen jets from {len(file_paths)} file(s)")

    feature_arrays = {}
    for file_path in file_paths:
        log.info(f"Reading {file_path}")
        constits_named, jets_named = load_file(file_path)
        for name, arr in compute_features(constits_named, jets_named).items():
            feature_arrays.setdefault(name, []).append(arr)

    feature_arrays = {name: np.concatenate(arrs, axis=0) for name, arrs in feature_arrays.items()}

    n_samples = len(next(iter(feature_arrays.values())))
    log.info(f"Read {n_samples} samples for each feature")

    sample_name = "aspen_open_jets"
    for field, arr in feature_arrays.items():
        log.info(f"Plotting feature '{field}'")
        flat = arr.flatten()
        n_dropped = (~np.isfinite(flat)).sum()
        if n_dropped:
            log.info(f"Dropping {n_dropped} non-finite value(s) for '{field}'")
            flat = flat[np.isfinite(flat)]
        heplt.plot_feature_hist_for_n_samples(
            data=[flat],
            sample_names=[sample_name],
            xlabel=field,
            bins=70,
            plot_name=f"{field}__{sample_name}",
            fig_dir=output_dir,
            show_plt=False,
            legend_outside=False,
            fig_size=(6.0, 4.5),
        )

    log.info(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        file_pattern=args.file_pattern,
        output_dir=args.output_dir,
        max_files=args.max_files,
        seed=args.seed,
    )
    log.info("------------ Finished plotting. ------------")
