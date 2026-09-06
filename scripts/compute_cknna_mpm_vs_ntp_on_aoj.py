
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


def epoch_label_from_ckpt_path(path):
    """Return a checkpoint's epoch number as an int, or "best"/"last" for those files."""
    import re

    name = Path(path).name
    if name.lower() == "best.ckpt":
        return "best"
    if name.lower() == "last.ckpt":
        return "last"
    match = re.search(r"epoch[=_-]?([0-9]+)", name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else name


def epoch_sort_key(epoch):
    """Sort key that orders numeric epochs ascending, then "best", then "last"."""
    if isinstance(epoch, int):
        return (0, epoch)
    if epoch == "best":
        return (1, 0)
    if epoch == "last":
        return (2, 0)
    return (3, 0)


def pick_checkpoints_from_model_dir(model_path, n=3, order="epoch"):
    """Return a selection of checkpoint paths from a model directory.

    The selection is `n` evenly spaced checkpoints (excluding `last.ckpt`/`best.ckpt`)
    plus `best.ckpt`, if present.

    order: "epoch" (default) picks evenly spaced checkpoints by epoch number;
    "loss" picks them by the loss value parsed from the filename instead.
    Checkpoint filenames are expected to look like
    `epoch_000_step_100000_loss_0.63595.ckpt`.

    The function searches under <model_path>/checkpoints first and falls back to
    the model directory itself if needed.
    """
    from pathlib import Path
    import re

    if order not in ("epoch", "loss"):
        raise ValueError(f"order must be 'epoch' or 'loss', got {order!r}")

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

    def epoch_from_name(path):
        match = re.search(r"epoch[=_-]?([0-9]+)", path.name, flags=re.IGNORECASE)
        return int(match.group(1)) if match else None

    def loss_from_name(path):
        match = re.search(r"loss[=_-]?([0-9]+(?:\.[0-9]+)?)", path.name, flags=re.IGNORECASE)
        return float(match.group(1)) if match else None

    def is_named_checkpoint(path, name):
        return path.name.lower() == name

    key_fn = epoch_from_name if order == "epoch" else loss_from_name

    # candidates for the n evenly spaced picks: exclude last.ckpt/best.ckpt and
    # anything the sort key can't be parsed from
    candidates = [
        p for p in unique_ckpts
        if not is_named_checkpoint(p, "last.ckpt")
        and not is_named_checkpoint(p, "best.ckpt")
        and key_fn(p) is not None
    ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoints with a parseable {order} found under {model_dir!s}")
    candidates = sorted(candidates, key=key_fn)

    if len(candidates) <= n:
        selected = list(candidates)
    else:
        idxs = np.linspace(0, len(candidates) - 1, num=n, dtype=int)
        selected = [candidates[i] for i in sorted(set(int(i) for i in idxs))]

    best_ckpt = next((p for p in unique_ckpts if is_named_checkpoint(p, "best.ckpt")), None)
    if best_ckpt is not None:
        selected.append(best_ckpt)

    # Deduplicate again and keep a stable order.
    final = []
    seen = set()
    for p in selected:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            final.append(p)

    return final


def load_n_model_checkpoints(model_path, n=3, order="epoch"):
    """Load n checkpoints from a model directory and return the corresponding models.

    The selection of checkpoints is determined by `pick_checkpoints_from_model_dir`.
    Returns (models, checkpoint_paths).
    """
    checkpoint_paths = pick_checkpoints_from_model_dir(model_path, n=n, order=order)
    models = []
    for ckpt_path in checkpoint_paths:
        model = BackboneMultiHeadLightning.load_from_checkpoint(ckpt_path, strict=False)
        model.eval()
        model.to(device)
        models.append(model)
    return models, checkpoint_paths



def do_inference_on_aoj(dataloader, model, max_batches=None):
    """Run inference on the AOJ dataloader using the given model.

    Returns (latent_representations, padding_masks): the mask marks the valid
    (non-padded) particles per jet, including the prepended start token, so it
    lines up 1:1 with the latent representations' sequence dimension.
    """
    latent_representations = []
    padding_masks = []
    with torch.no_grad():
        for batch in dataloader:
            if max_batches is not None and len(latent_representations) >= max_batches:
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
            latent_representations.append(latent.cpu().numpy())
            padding_masks.append(part_mask.cpu().numpy())
    return np.concatenate(latent_representations, axis=0), np.concatenate(padding_masks, axis=0)


def compute_cknna_between_models(latent1, latent2, mask1, mask2, topks=(5, 10, 25, 50), n_sub=1500, n_draws=5):
    """Compute pooled per-jet CKNNA (with permutation-null baseline) between two models.

    Returns the list of per-topk result dicts from `cknna.sweep_topk`.
    """
    # cknna.pool_jets expects torch Tensors (uses .to()/.device), not numpy arrays
    latent1, mask1 = torch.from_numpy(latent1), torch.from_numpy(mask1)
    latent2, mask2 = torch.from_numpy(latent2), torch.from_numpy(mask2)
    U, V = cknna.pool_jets(latent1, mask1), cknna.pool_jets(latent2, mask2)

    rows = cknna.sweep_topk(U, V, topks=topks, n_sub=n_sub, n_draws=n_draws)
    print("\n--- alignment ---")
    for row in rows:
        print(f"topk={row['topk']:>4}  CKNNA={row['cknna']:.4f}+-{row['cknna_std']:.4f}   "
                f"null={row['null']:.4f}+-{row['null_std']:.4f}   z={row['z']:.1f}")
    return rows


def compute_identity_and_permutation_baselines(latent, mask, topks=(5, 10, 25, 50), n_sub=1500, n_draws=5):
    """Compute pooled per-jet CKNNA baselines for a single model, against itself.

    Returns (rows, rows_perm, identity_baselines): the self-CKNNA sweep, the
    permuted-CKNNA sweep, and the identity-map baseline dict.
    """
    # cknna.pool_jets expects torch Tensors (uses .to()/.device), not numpy arrays
    latent, mask = torch.from_numpy(latent), torch.from_numpy(mask)
    U = cknna.pool_jets(latent, mask)

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

    is_baseline = lambda pair_name: "identity" in pair_name or "permutation" in pair_name  # noqa: E731
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

    max_batches = None
    batch_size_train = cfg_aoj.data.batch_size.train
    # +1 batch of margin, since the last batch from a capped file can come up short
    n_jets_per_file = (max_batches + 1) * batch_size_train if max_batches is not None else None

    dataloader_aoj = get_aoj_dataloader(cfg_aoj, AOJ_FILE, n_jets_per_file=n_jets_per_file)

    n_events_aoj = get_n_events_in_aoj_file(AOJ_FILE)
    print(f"Loaded AOJ dataloader with {n_events_aoj} jets from {AOJ_FILE}")

    ###################### load foundation models #######################

    # "epoch" picks evenly spaced checkpoints by epoch number, "loss" by loss value
    CHECKPOINT_ORDER = "epoch"

    ### pretrain

    # load 3 checkpoints from ntp model
    ntp_model_path = "/home/users/w/wozniak/dev/enhancing-ntp4jets/trained/omnijet-backbone-multihead-allfeat-ntp/runs/2026-08-23_13-26-36_gpu017_LongitudinalDyslectic"
    ntp_models, ntp_ckpt_paths = load_n_model_checkpoints(ntp_model_path, n=3, order=CHECKPOINT_ORDER)

    print(f"Loaded {len(ntp_models)} NTP model checkpoints from {ntp_model_path}")

    # load 3 checkpoints from mpm model
    mpm_model_path = "/home/users/w/wozniak/dev/enhancing-ntp4jets/trained/omnijet-backbone-multihead-allfeat-mpm/runs/2026-08-17_11-46-48_gpu050_RetentiveUnuntrium"
    mpm_models, mpm_ckpt_paths = load_n_model_checkpoints(mpm_model_path, n=3, order=CHECKPOINT_ORDER)

    print(f"Loaded {len(mpm_models)} MPM model checkpoints from {mpm_model_path}")

    ### finetune

    # load 3 checkpoints from finetuned ntp model
    ntp_finetuned_model_path = "/home/users/w/wozniak/dev/enhancing-ntp4jets/trained/omnijet-backbone-multihead-allfeat-ntp-classifier-head/runs/2026-08-29_03-54-46_gpu023_UngussetedHandsaw"
    ntp_finetuned_models, ntp_finetuned_ckpt_paths = load_n_model_checkpoints(
        ntp_finetuned_model_path, n=3, order=CHECKPOINT_ORDER
    )

    print(f"Loaded {len(ntp_finetuned_models)} finetuned NTP model checkpoints from {ntp_finetuned_model_path}")

    # load 3 checkpoints from finetuned mpm model
    mpm_finetuned_model_path = "/home/users/w/wozniak/dev/enhancing-ntp4jets/trained/omnijet-backbone-multihead-allfeat-mpm-classifier-head/runs/2026-08-21_15-06-12_gpu024_TympanicAnkylosis"
    mpm_finetuned_models, mpm_finetuned_ckpt_paths = load_n_model_checkpoints(
        mpm_finetuned_model_path, n=3, order=CHECKPOINT_ORDER
    )

    print(f"Loaded {len(mpm_finetuned_models)} finetuned MPM model checkpoints from {mpm_finetuned_model_path}")

    # epoch labels (int epoch number, or "best"/"last") per checkpoint, used as the x-axis when plotting
    ntp_epochs = [epoch_label_from_ckpt_path(p) for p in ntp_ckpt_paths]
    mpm_epochs = [epoch_label_from_ckpt_path(p) for p in mpm_ckpt_paths]
    ntp_finetuned_epochs = [epoch_label_from_ckpt_path(p) for p in ntp_finetuned_ckpt_paths]
    mpm_finetuned_epochs = [epoch_label_from_ckpt_path(p) for p in mpm_finetuned_ckpt_paths]


    ###################### inference on AOJ #######################

    print('running inference on AOJ dataset...')

    # pass aoj data through each model and get latent representations + padding masks
    ntp_latents, ntp_masks = zip(*(do_inference_on_aoj(dataloader_aoj, model, max_batches=max_batches) for model in ntp_models))

    print('NTP pre-trained inference complete.')

    mpm_latents, mpm_masks = zip(*(do_inference_on_aoj(dataloader_aoj, model, max_batches=max_batches) for model in mpm_models))
    print('MPM pre-trained inference complete.')

    ntp_finetuned_latents, ntp_finetuned_masks = zip(
        *(do_inference_on_aoj(dataloader_aoj, model, max_batches=max_batches) for model in ntp_finetuned_models))
    print('NTP finetuned inference complete.')
    
    mpm_finetuned_latents, mpm_finetuned_masks = zip(
        *(do_inference_on_aoj(dataloader_aoj, model, max_batches=max_batches) for model in mpm_finetuned_models)
    )
    print('MPM finetuned inference complete.')


    ###################### alignment #######################

    ####### compute CKNNA between model pairs

    # for each checkpoint pair, compute cknna and baseline identity / permutation
    # epoch is taken from the first ("a") model in the pair, since that's what's plotted on the x-axis
    model_pairs = [
        ("ntp_vs_mpm", ntp_latents, mpm_latents, ntp_masks, mpm_masks, ntp_epochs),
        ("ntp_finetuned_vs_mpm_finetuned", ntp_finetuned_latents, mpm_finetuned_latents,
         ntp_finetuned_masks, mpm_finetuned_masks, ntp_finetuned_epochs),
    ]

    print('computing CKNNA between model pairs and baselines...')

    results = []
    for pair_name, latents_a, latents_b, masks_a, masks_b, epochs in model_pairs:
        for i, (latent_a, latent_b, mask_a, mask_b) in enumerate(zip(latents_a, latents_b, masks_a, masks_b)):
            print(f"\n=== {pair_name} | checkpoint pair {i} ===")
            rows = compute_cknna_between_models(latent_a, latent_b, mask_a, mask_b)
            for row in rows:
                results.append({"pair": pair_name, "checkpoint_idx": i, "epoch": epochs[i], **row})

    ####### compute self-CKNNA identity / permutation baselines

    # identity / permutation baselines, per model group and checkpoint
    model_groups = [
        ("ntp", ntp_latents, ntp_masks, ntp_epochs),
        ("mpm", mpm_latents, mpm_masks, mpm_epochs),
        ("ntp_finetuned", ntp_finetuned_latents, ntp_finetuned_masks, ntp_finetuned_epochs),
        ("mpm_finetuned", mpm_finetuned_latents, mpm_finetuned_masks, mpm_finetuned_epochs),
    ]
    for group_name, latents, masks, epochs in model_groups:
        for i, (latent, mask) in enumerate(zip(latents, masks)):
            print(f"\n=== {group_name} | checkpoint {i} baselines ===")
            rows, rows_perm, identity = compute_identity_and_permutation_baselines(latent, mask)
            for row in rows:
                results.append({"pair": f"{group_name}_identity", "checkpoint_idx": i, "epoch": epochs[i], **row})
            for row in rows_perm:
                results.append(
                    {"pair": f"{group_name}_permutation", "checkpoint_idx": i, "epoch": epochs[i], **row}
                )
            results.append(
                {"pair": f"{group_name}_identity_map", "checkpoint_idx": i, "epoch": epochs[i], **identity}
            )


    ###################### results #######################
    
    print('saving CKNNA results...')

    # save the values
    batches_label = f"{max_batches}batches" if max_batches is not None else "allbatches"
    output_dir = Path("analysis/cknna") / f"{batches_label}_order-{CHECKPOINT_ORDER}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cknna_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved CKNNA results to {output_path}")

    # plot the results, one figure per topk
    for topk in (5, 10, 25, 50):
        print(f'plotting CKNNA results for top-{topk}...')
        plot_cknna_results(results, output_dir, topk=topk, plot_name="cknna_vs_epoch")
