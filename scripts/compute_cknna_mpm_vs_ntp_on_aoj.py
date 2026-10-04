
import sys
import json
from pathlib import Path

# this kernel's numpy (2.1.2) + uproot (5.1.2) combo raises OverflowError reading JetClass
# .root files - prepend the deps folder with compatible versions before anything imports numpy
deps_path = "/home/users/w/wozniak/.local/ntp4jets-deps"
if deps_path not in sys.path:
    sys.path.insert(0, deps_path)

import pyrootutils

pyrootutils.setup_root(__file__ if "__file__" in dir() else ".", indicator=".project-root", pythonpath=True)

import h5py
import hydra
import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from gabbro.models.backbone_multihead import BackboneMultiHeadLightning
import gabbro.metrics.cknna as cknna
from gabbro.utils.pylogger import get_pylogger

logger = get_pylogger(__name__)

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_aoj_dataloader(cfg, file_path, n_jets_per_file=None):

    # switch off the tokenized-parquet codepath and jetclass-only extras, point at the AOJ file
    cfg.data.dataset_kwargs_common.dataset_type = "aoj"
    cfg.data.dataset_kwargs_common.token_id_cfg = None
    cfg.data.dataset_kwargs_common.feature_dict_labels_particles = None
    cfg.data.dataset_kwargs_common.feature_dict_jet = None
    cfg.data.dataset_kwargs_common.labels_to_load = ["label_QCD"]

    cfg.data.data_dir = "/srv/beegfs/scratch/groups/rodem/lumea/datasets/aspenJets"
    # `setup(stage="fit")` also builds the val dataset, so point that at the same single file too
    cfg.data.dataset_kwargs_train.files_dict = {"QCD": [file_path]}
    cfg.data.dataset_kwargs_train.max_n_files_per_type = 1
    cfg.data.dataset_kwargs_val.files_dict = {"QCD": [file_path]}
    cfg.data.dataset_kwargs_val.max_n_files_per_type = 1

    # disable all shuffling so batch order is just the file's on-disk order, identical every run
    for split in ["train", "val", "test"]:
        cfg.data[f"dataset_kwargs_{split}"].shuffle_files = False
        cfg.data[f"dataset_kwargs_{split}"].shuffle_data = False

    # without this, the dataset reloads + re-preprocesses the *entire* file from scratch
    # every time a new dataloader iterator is created (i.e. once per model checkpoint)
    cfg.data.dataset_kwargs_train.load_only_once = True
    cfg.data.dataset_kwargs_val.load_only_once = True

    if n_jets_per_file is not None:
        cfg.data.dataset_kwargs_train.n_jets_per_file = n_jets_per_file
        cfg.data.dataset_kwargs_val.n_jets_per_file = n_jets_per_file

    print(OmegaConf.to_yaml(cfg.data.dataset_kwargs_common.feature_dict, resolve=True))

    datamodule_aoj = hydra.utils.instantiate(cfg_aoj.data)
    # AOJ has no separate train/val/test split - just read the file(s) directly
    datamodule_aoj.setup(stage="fit")

    train_dataloader_aoj = datamodule_aoj.train_dataloader()

    return train_dataloader_aoj
    

def get_n_events_in_aoj_file(file_path):
    """Return the number of jets/events stored in an AOJ h5 file.

    Reads the leading dimension of the "PFCands" dataset directly, avoiding
    the need to iterate (or call len() on) the IterableDataset/dataloader.
    """
    with h5py.File(file_path, "r") as f:
        return f["PFCands"].shape[0]


UNTRAINED_CKPT_NAME = "untrained_model_state_dict.ckpt"  # random init, saved before fit
# backbone weights right after fine-tuning loaded the pretrained backbone (= fine-tuning start)
INIT_CKPT_NAME = "state_dict_after_loading_backbone_weights.ckpt"
# both are bare state dicts without hyper-parameters
BARE_STATE_DICT_NAMES = (UNTRAINED_CKPT_NAME, INIT_CKPT_NAME)
# fine-tuning loaded the pretraining runs' `last.ckpt`, which pretraining overwrote afterwards;
# these are the exact snapshots that were loaded (ntp, mpm), found by comparing weights
PRETRAIN_INIT_NAMES = (
    "epoch_013_step_1375000_loss_3.34477.ckpt",
    "epoch_013_step_1375000_loss_1.09179.ckpt",
)
PRETRAIN_INIT_EPOCH = 13
N_RANDOM_SEEDS = 5


def epoch_label_from_ckpt_path(path):
    """Return a checkpoint's epoch number as an int, or a label such as "untrained" for special files."""
    import re

    name = Path(path).name
    if name == UNTRAINED_CKPT_NAME:
        return "untrained"
    if name == INIT_CKPT_NAME:
        return "init"
    if name in PRETRAIN_INIT_NAMES:
        return "ft_src"
    if name.lower() == "best.ckpt":
        return "best"
    if name.lower() == "last.ckpt":
        return "last"
    match = re.search(r"epoch[=_-]?([0-9]+)", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else name


def epoch_sort_key(epoch):
    """Sort key: "untrained", "init" (fine-tuning start), numeric epochs, "ft_src", "best", "last"."""
    if epoch == "untrained":
        return (-2, 0)
    if epoch == "init":
        return (-1, 0)
    if epoch == "ft_src":
        return (0.5, 0)
    if isinstance(epoch, int):
        return (0, epoch)
    if epoch == "best":
        return (1, 0)
    if epoch == "last":
        return (2, 0)
    return (3, 0)


def pick_checkpoints_from_model_dir(model_path, max_epoch=None, init_ckpt=None, init_first=False):
    """Return `untrained_model_state_dict.ckpt` (if present), then one checkpoint per epoch
    (the one with the lowest step), then `last.ckpt`.

    If `max_epoch` is given, only epochs up to and including it are kept and `last.ckpt`
    is dropped, since it holds the state after the final epoch of training.

    `init_ckpt` is an extra checkpoint (the weights fine-tuning starts from, or the pretraining
    snapshot it loaded); it goes right after the untrained one if `init_first`, else at the end.

    Checkpoint filenames are expected to look like
    `epoch_000_step_100000_loss_0.63595.ckpt`.

    The function searches under <model_path>/checkpoints first and falls back to
    the model directory itself if needed.
    """
    from pathlib import Path
    import re

    model_dir = Path(model_path)
    checkpoint_dirs = [model_dir / "checkpoints", model_dir]

    ckpt_files = []
    for d in checkpoint_dirs:
        if d.exists() and d.is_dir():
            ckpt_files.extend(sorted(d.glob("*.ckpt")))

    # Deduplicate while preserving order, and exclude non-final checkpoint copies
    seen = set()
    unique_ckpts = []
    for p in ckpt_files:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique_ckpts.append(p)

    if not unique_ckpts:
        raise FileNotFoundError(f"No .ckpt files found under {model_dir!s}")

    # keep the lowest-step (first saved) checkpoint of each epoch
    first_per_epoch = {}
    for p in unique_ckpts:
        if p.name in PRETRAIN_INIT_NAMES:
            continue  # added separately via init_ckpt
        match = re.search(r"epoch[=_-]?([0-9]+)_step[=_-]?([0-9]+)", p.name, flags=re.IGNORECASE)
        if match is None:
            continue
        epoch, step = int(match.group(1)), int(match.group(2))
        if max_epoch is not None and epoch > max_epoch:
            continue
        if epoch not in first_per_epoch or step < first_per_epoch[epoch][0]:
            first_per_epoch[epoch] = (step, p)
    selected = [p for _, (_, p) in sorted(first_per_epoch.items())]

    last_ckpt = next((p for p in unique_ckpts if p.name.lower() == "last.ckpt"), None)
    if last_ckpt is not None and max_epoch is None:
        selected.append(last_ckpt)

    if init_ckpt is not None:
        selected.insert(0 if init_first else len(selected), Path(init_ckpt))

    # the untrained weights are saved in the run directory, next to (not inside) checkpoints/
    untrained_ckpt = model_dir / UNTRAINED_CKPT_NAME
    if untrained_ckpt.exists():
        selected.insert(0, untrained_ckpt)

    if not selected:
        raise FileNotFoundError(f"No epoch checkpoints or last.ckpt found under {model_dir!s}")
    return selected


def load_model_checkpoints(model_path, max_epoch=None, init_ckpt=None, init_first=False):
    """Load the untrained weights, one checkpoint per epoch (and last.ckpt) from a model directory.

    The selection of checkpoints is determined by `pick_checkpoints_from_model_dir`.
    Returns (models, checkpoint_paths).
    """
    checkpoint_paths = pick_checkpoints_from_model_dir(
        model_path, max_epoch=max_epoch, init_ckpt=init_ckpt, init_first=init_first
    )
    # bare state dicts have no hyper-parameters, so build their model from the first real
    # checkpoint and overwrite the weights
    reference_ckpt = first_real_ckpt(checkpoint_paths)
    models = []
    for ckpt_path in checkpoint_paths:
        if ckpt_path.name in BARE_STATE_DICT_NAMES:
            model = BackboneMultiHeadLightning.load_from_checkpoint(reference_ckpt, strict=False)
            missing, unexpected = model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=False)
            print(f"{ckpt_path.name}: {len(missing)} missing, {len(unexpected)} unexpected keys")
        else:
            model = BackboneMultiHeadLightning.load_from_checkpoint(ckpt_path, strict=False)
        model.eval()
        model.to(device)
        models.append(model)
    return models, checkpoint_paths


def first_real_ckpt(checkpoint_paths):
    """First path that is a Lightning checkpoint (i.e. not a bare state dict)."""
    return next(p for p in checkpoint_paths if p.name not in BARE_STATE_DICT_NAMES)


def random_init_embeddings(reference_ckpt, seed, dataloader, max_batches):
    """Pooled embeddings of a freshly initialised model with the architecture of `reference_ckpt`."""
    hparams = torch.load(reference_ckpt, map_location="cpu", weights_only=False)["hyper_parameters"]
    torch.manual_seed(seed)
    model = BackboneMultiHeadLightning(**hparams).eval().to(device)
    return do_inference_on_aoj(dataloader, model, max_batches=max_batches)



def do_inference_on_aoj(dataloader, model, max_batches=None):
    """Run inference on the AOJ dataloader and return per-jet embeddings of shape (n_jets, D).

    Each batch is mean-pooled over its valid (non-padded) particles, including the prepended
    start token, right away, so the full per-particle outputs are never kept in memory.
    """
    pooled = []
    with torch.no_grad():
        for batch in dataloader:
            if max_batches is not None and len(pooled) >= max_batches:
                break
            part_features = batch["part_features"].to(device)
            part_mask = batch["part_mask"].to(device).int()

            if model.hparams.start_token_type in ["trainable", "zero_pad"]:
                batch_size = part_features.shape[0]
                max_sequence_len = model.hparams.backbone_cfg.max_sequence_len
                part_features = part_features[:, : max_sequence_len - 1]
                part_mask = part_mask[:, : max_sequence_len - 1]
                start_token = model.start_token_continuous.unsqueeze(0).repeat(batch_size, 1, 1)
                part_features = torch.cat([start_token, part_features], dim=1)
                start_mask = torch.ones(batch_size, 1, dtype=part_mask.dtype, device=device)
                part_mask = torch.cat([start_mask, part_mask], dim=1)

            latent = model.backbone(part_features, mask=part_mask)
            pooled.append(cknna.pool_jets(latent, part_mask).cpu())
    return torch.cat(pooled, dim=0)


def compute_cknna_between_models(U, V, topks=(5, 10, 25, 50), n_sub=1500, n_draws=5):
    """Compute per-jet CKNNA (with permutation-null baseline) between two models' pooled embeddings.

    Returns the list of per-topk result dicts from `cknna.sweep_topk`.
    """
    rows = cknna.sweep_topk(U, V, topks=topks, n_sub=n_sub, n_draws=n_draws)
    print("\n--- alignment ---")
    for row in rows:
        print(f"topk={row['topk']:>4}  CKNNA={row['cknna']:.4f}+-{row['cknna_std']:.4f}   "
                f"null={row['null']:.4f}+-{row['null_std']:.4f}   z={row['z']:.1f}")
    return rows


def compute_identity_and_permutation_baselines(U, topks=(5, 10, 25, 50), n_sub=1500, n_draws=5):
    """Compute per-jet CKNNA baselines for a single model's pooled embeddings, against itself.

    Returns (rows, rows_perm, identity_baselines): the self-CKNNA sweep, the
    permuted-CKNNA sweep, and the identity-map baseline dict.
    """
    rows = cknna.sweep_topk(U, U, topks=topks, n_sub=n_sub, n_draws=n_draws)
    print("\n--- identity baseline ---")
    for row in rows:
        print(f"topk={row['topk']:>4}  CKNNA={row['cknna']:.4f}+-{row['cknna_std']:.4f}   "
                f"null={row['null']:.4f}+-{row['null_std']:.4f}   z={row['z']:.1f}")

    rows_perm = cknna.sweep_topk(U, U[torch.randperm(U.shape[0])], topks=topks, n_sub=n_sub, n_draws=n_draws)
    print("\n--- permutation baseline ---")
    for row in rows_perm:
        print(f"topk={row['topk']:>4}  CKNNA={row['cknna']:.4f}+-{row['cknna_std']:.4f}   "
                f"null={row['null']:.4f}+-{row['null_std']:.4f}   z={row['z']:.1f}")

    identity = cknna.identity_baselines(U, U)
    print("\n--- identity-map baselines ---")
    for k, val in identity.items():
        print(f"{k:>18}: {val:.4f}" if isinstance(val, float) else f"{k:>18}: {val}")

    return rows, rows_perm, identity


def _plot_cknna_and_z_vs_epoch(by_pair, output_path, topk, title=None):
    """Draw the CKNNA-vs-epoch and z-vs-epoch subplots for the given pairs and save to disk."""
    # different model groups can have different checkpoint/epoch labels, so build one
    # consistently (numeric-then-best/last) sorted x-axis shared by all pairs here, instead
    # of relying on matplotlib's categorical axis, which orders ticks by first appearance
    all_epochs = sorted({r["epoch"] for rows in by_pair.values() for r in rows}, key=epoch_sort_key)
    epoch_to_x = {epoch: i for i, epoch in enumerate(all_epochs)}

    fig, (ax_cknna, ax_z) = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(by_pair), 1)))
    for color, (pair_name, pair_rows) in zip(colors, by_pair.items()):
        style = dict(color=color, marker="o", linestyle="-", label=pair_name)
        x = [epoch_to_x[r["epoch"]] for r in pair_rows]
        cknna_vals = [r["cknna"] for r in pair_rows]
        cknna_errs = [r["cknna_std"] for r in pair_rows]
        z_vals = [r["z"] for r in pair_rows]
        ax_cknna.errorbar(x, cknna_vals, yerr=cknna_errs, **style)
        ax_z.plot(x, z_vals, **style)

    for ax in (ax_cknna, ax_z):
        ax.set_xticks(range(len(all_epochs)))
        ax.set_xticklabels([str(e) for e in all_epochs])
        ax.set_xlabel("epoch")

    ax_cknna.set_ylabel(f"CKNNA (topk={topk})")
    ax_z.set_ylabel(f"z (topk={topk})")
    ax_z.axhline(0, color="black", lw=0.8, linestyle=":")

    if title:
        fig.suptitle(title)

    handles, labels = ax_cknna.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0.0), loc="lower center", ncol=3, fontsize=9)
    fig.tight_layout(rect=[0, 0.22, 1, 1])

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_cknna_results(results, output_dir, topk=10, plot_name="cknna_vs_epoch"):
    """Plot CKNNA and z vs epoch, at a fixed topk, in two separate figures.

    One figure holds the real model-vs-model comparisons; the other holds the
    per-group identity/permutation self-comparison baselines.
    """
    rows = [r for r in results if r.get("topk") == topk]
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair"], []).append(r)
    for pair_rows in by_pair.values():
        pair_rows.sort(key=lambda r: epoch_sort_key(r["epoch"]))

    is_baseline = lambda pair_name: any(s in pair_name for s in ("identity", "permutation", "random_seeds"))  # noqa: E731
    comparisons = {name: rows for name, rows in by_pair.items() if not is_baseline(name)}
    baselines = {name: rows for name, rows in by_pair.items() if is_baseline(name)}

    if comparisons:
        _plot_cknna_and_z_vs_epoch(
            comparisons, Path(output_dir) / f"{plot_name}_topk{topk}.png", topk,
            title="Model comparisons (CKNNA between checkpoints)",
        )
    if baselines:
        _plot_cknna_and_z_vs_epoch(
            baselines, Path(output_dir) / f"{plot_name}_baselines_topk{topk}.png", topk,
            title="Baselines (identity & permutation self-comparison)",
        )


if __name__ == "__main__":

    ############### load aoj data through dataloader #################

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir="/home/users/w/wozniak/dev/enhancing-ntp4jets/configs", version_base=None):
        cfg_aoj = compose(
            config_name="train",
            overrides=["experiment=example_experiment_backbone_and_head_allfeat_ntp_classifier_head",
                "feature_dict=feature_dict_all_without_cuts"],
        )
        
    AOJ_FILE = "/srv/beegfs/scratch/groups/rodem/lumea/datasets/aspenJets/RunG_batch0.h5"

    max_batches = 50
    batch_size_train = cfg_aoj.data.batch_size.train
    # +1 batch of margin, since the last batch from a capped file can come up short
    n_jets_per_file = (max_batches + 1) * batch_size_train if max_batches is not None else None

    dataloader_aoj = get_aoj_dataloader(cfg_aoj, AOJ_FILE, n_jets_per_file=n_jets_per_file)

    n_events_aoj = get_n_events_in_aoj_file(AOJ_FILE)
    print(f"Loaded AOJ dataloader with {n_events_aoj} jets from {AOJ_FILE}")

    ###################### load foundation models #######################

    # untrained weights, one checkpoint per epoch (lowest step) and the fine-tuning start/source

    ### pretrain

    # fine-tuning started from the epoch-13 snapshot, so the pretraining curve ends there
    ntp_model_path = "/home/users/w/wozniak/dev/enhancing-ntp4jets/trained/omnijet-backbone-multihead-allfeat-ntp/runs/2026-09-09_10-23-31_gpu017_NonstandardAitchbone"
    ntp_models, ntp_ckpt_paths = load_model_checkpoints(
        ntp_model_path, max_epoch=PRETRAIN_INIT_EPOCH,
        init_ckpt=Path(ntp_model_path) / "checkpoints" / PRETRAIN_INIT_NAMES[0],
    )

    print(f"Loaded {len(ntp_models)} NTP model checkpoints from {ntp_model_path}")

    mpm_model_path = "/home/users/w/wozniak/dev/enhancing-ntp4jets/trained/omnijet-backbone-multihead-allfeat-mpm/runs/2026-09-09_10-23-23_gpu017_OrganicMolluscum"
    mpm_models, mpm_ckpt_paths = load_model_checkpoints(
        mpm_model_path, max_epoch=PRETRAIN_INIT_EPOCH,
        init_ckpt=Path(mpm_model_path) / "checkpoints" / PRETRAIN_INIT_NAMES[1],
    )

    print(f"Loaded {len(mpm_models)} MPM model checkpoints from {mpm_model_path}")

    ### finetune

    ntp_finetuned_model_path = "/home/users/w/wozniak/dev/enhancing-ntp4jets/trained/omnijet-backbone-multihead-allfeat-ntp-classifier-head/runs/2026-09-14_17-32-09_gpu047_NeonatalSulfur"
    # the finetuned NTP run trained 24 epochs, the finetuned MPM run 16 - cut NTP to epochs 0-15 to compare equally
    ntp_finetuned_models, ntp_finetuned_ckpt_paths = load_model_checkpoints(
        ntp_finetuned_model_path, max_epoch=15, init_ckpt=Path(ntp_finetuned_model_path) / INIT_CKPT_NAME, init_first=True
    )

    print(f"Loaded {len(ntp_finetuned_models)} finetuned NTP model checkpoints from {ntp_finetuned_model_path}")

    mpm_finetuned_model_path = "/home/users/w/wozniak/dev/enhancing-ntp4jets/trained/omnijet-backbone-multihead-allfeat-mpm-classifier-head/runs/2026-09-14_17-32-09_gpu024_TaxableSpecialism"
    mpm_finetuned_models, mpm_finetuned_ckpt_paths = load_model_checkpoints(
        mpm_finetuned_model_path, max_epoch=15, init_ckpt=Path(mpm_finetuned_model_path) / INIT_CKPT_NAME, init_first=True
    )

    print(f"Loaded {len(mpm_finetuned_models)} finetuned MPM model checkpoints from {mpm_finetuned_model_path}")

    # epoch labels (int epoch number, or "best"/"last") per checkpoint, used as the x-axis when plotting
    ntp_epochs = [epoch_label_from_ckpt_path(p) for p in ntp_ckpt_paths]
    mpm_epochs = [epoch_label_from_ckpt_path(p) for p in mpm_ckpt_paths]
    ntp_finetuned_epochs = [epoch_label_from_ckpt_path(p) for p in ntp_finetuned_ckpt_paths]
    mpm_finetuned_epochs = [epoch_label_from_ckpt_path(p) for p in mpm_finetuned_ckpt_paths]


    ###################### inference on AOJ #######################

    print('running inference on AOJ dataset...')

    # pass aoj data through each model and get pooled per-jet embeddings
    ntp_latents = [do_inference_on_aoj(dataloader_aoj, model, max_batches=max_batches) for model in ntp_models]

    print('NTP pre-trained inference complete.')

    mpm_latents = [do_inference_on_aoj(dataloader_aoj, model, max_batches=max_batches) for model in mpm_models]
    print('MPM pre-trained inference complete.')

    ntp_finetuned_latents = [
        do_inference_on_aoj(dataloader_aoj, model, max_batches=max_batches) for model in ntp_finetuned_models
    ]
    print('NTP finetuned inference complete.')

    mpm_finetuned_latents = [
        do_inference_on_aoj(dataloader_aoj, model, max_batches=max_batches) for model in mpm_finetuned_models
    ]
    print('MPM finetuned inference complete.')

    # the fine-tuning start must reproduce the pretraining snapshot it loaded, bit for bit
    for name, pre_latents, pre_epochs, ft_latents, ft_epochs in [
        ("ntp", ntp_latents, ntp_epochs, ntp_finetuned_latents, ntp_finetuned_epochs),
        ("mpm", mpm_latents, mpm_epochs, mpm_finetuned_latents, mpm_finetuned_epochs),
    ]:
        diff = (pre_latents[pre_epochs.index("ft_src")] - ft_latents[ft_epochs.index("init")]).abs().max()
        print(f"[check] {name}: max |embedding(pretraining snapshot) - embedding(fine-tuning init)| = {diff:.2e}")
        if diff > 1e-3:
            print(f"[check] WARNING: {name} fine-tuning did not start from the pretraining snapshot")


    ###################### alignment #######################

    ####### compute CKNNA between model pairs

    # for each pair of models, compute cknna between the checkpoints that share an epoch label
    # (the number of epochs can differ between models, so pairing by index would mismatch them)
    model_pairs = [
        ("ntp_vs_mpm", ntp_latents, mpm_latents, ntp_epochs, mpm_epochs),
        ("ntp_finetuned_vs_mpm_finetuned", ntp_finetuned_latents, mpm_finetuned_latents,
         ntp_finetuned_epochs, mpm_finetuned_epochs),
    ]

    print('computing CKNNA between model pairs and baselines...')

    results = []
    for pair_name, latents_a, latents_b, epochs_a, epochs_b in model_pairs:
        idx_b = {epoch: j for j, epoch in enumerate(epochs_b)}
        for i, epoch in enumerate(epochs_a):
            if epoch not in idx_b:
                continue
            j = idx_b[epoch]
            print(f"\n=== {pair_name} | epoch {epoch} ===")
            rows = compute_cknna_between_models(latents_a[i], latents_b[j])
            for row in rows:
                results.append({"pair": pair_name, "checkpoint_idx": i, "epoch": epoch, **row})

    ####### compute self-CKNNA identity / permutation baselines

    # identity / permutation baselines, per model group and checkpoint
    model_groups = [
        ("ntp", ntp_latents, ntp_epochs),
        ("mpm", mpm_latents, mpm_epochs),
        ("ntp_finetuned", ntp_finetuned_latents, ntp_finetuned_epochs),
        ("mpm_finetuned", mpm_finetuned_latents, mpm_finetuned_epochs),
    ]
    for group_name, latents, epochs in model_groups:
        for i, latent in enumerate(latents):
            print(f"\n=== {group_name} | checkpoint {i} baselines ===")
            rows, rows_perm, identity = compute_identity_and_permutation_baselines(latent)
            for row in rows:
                results.append({"pair": f"{group_name}_identity", "checkpoint_idx": i, "epoch": epochs[i], **row})
            for row in rows_perm:
                results.append(
                    {"pair": f"{group_name}_permutation", "checkpoint_idx": i, "epoch": epochs[i], **row}
                )
            results.append(
                {"pair": f"{group_name}_identity_map", "checkpoint_idx": i, "epoch": epochs[i], **identity}
            )

    ####### random-init baseline: CKNNA between independently initialised models, several seeds

    random_pairs = [
        ("ntp_vs_mpm_random_seeds", ntp_ckpt_paths, mpm_ckpt_paths),
        ("ntp_finetuned_vs_mpm_finetuned_random_seeds", ntp_finetuned_ckpt_paths, mpm_finetuned_ckpt_paths),
    ]
    for pair_name, paths_a, paths_b in random_pairs:
        ref_a, ref_b = first_real_ckpt(paths_a), first_real_ckpt(paths_b)
        for seed in range(N_RANDOM_SEEDS):
            print(f"\n=== {pair_name} | seed {seed} ===")
            U = random_init_embeddings(ref_a, seed, dataloader_aoj, max_batches)
            V = random_init_embeddings(ref_b, 1000 + seed, dataloader_aoj, max_batches)
            for row in compute_cknna_between_models(U, V):
                results.append({"pair": pair_name, "checkpoint_idx": seed, "epoch": f"seed{seed}", **row})


    ###################### results #######################
    
    print('saving CKNNA results...')

    # save the values
    batches_label = f"{max_batches}batches" if max_batches is not None else "allbatches"
    output_dir = Path("analysis/cknna") / f"{batches_label}_per-epoch"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cknna_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved CKNNA results to {output_path}")

    # plot the results, one figure per topk
    for topk in (5, 10, 25, 50):
        print(f'plotting CKNNA results for top-{topk}...')
        plot_cknna_results(results, output_dir, topk=topk, plot_name="cknna_vs_epoch")
