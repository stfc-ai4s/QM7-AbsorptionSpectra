#!/usr/bin/env python3
"""
Create all manuscript plots from already-generated data.

The script makes:
  - normalization prediction + error histogram
  - Figure 3: three example spectra
  - Figure 4: comparison of two similar molecules
All figures are saved in one output directory.

Usage
-----
python make_figures_3_4_5.py
python make_figures_3_4_5.py --output-dir /path/to/paper_plots

Edit the paths in the SETTINGS section below if your data locations change.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

from pathlib import Path


# Repository root: two directories above scripts/plotting/
REPO_ROOT = Path(__file__).resolve().parents[2]

# -----------------------------------------------------------------------------
# SETTINGS
# -----------------------------------------------------------------------------

NORMALIZATION_NPZ = Path(
    "/home/ubuntu/projects/densitySpec/density_normconst_runs/exp_sum900_v3/"
    "val_preds_true_raw.npz"
)

SPECTRUM_FILES = [
    REPO_ROOT / "predictions/density_M481/spectrum_prediction_M481.dat",
    REPO_ROOT / "predictions/density_M3935/spectrum_prediction_M3935.dat",
    REPO_ROOT / "predictions/density_M4170/spectrum_prediction_M4170.dat",
    REPO_ROOT / "predictions/density_M972/spectrum_prediction_M972.dat",
]


CHEMISTRY_DATA_DIR = Path(
    "/home/ubuntu/projects/densitySpec/size_scaling_exp004_v2/cnn_exp003/"
    "chemistry_spectra_error_analysis"
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "figures"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def molecule_image(smiles: str, size=(400, 220)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    rdDepictor.Compute2DCoords(mol)
    image = Draw.MolToImage(mol, size=size, kekulize=True, fitImage=True).convert("RGBA")

    arr = np.asarray(image).copy()
    white = np.all(arr[:, :, :3] > 245, axis=2)
    arr[white, 3] = 0
    return arr


def load_spectra():
    files = [np.loadtxt(path) for path in SPECTRUM_FILES]
    true = [f[:, 1] for f in files]
    pred = [f[:, 2] for f in files]
    return true, pred


# -----------------------------------------------------------------------------
# Plot 1: normalization constant
# -----------------------------------------------------------------------------

def plot_normalization(output_dir: Path, dpi: int):
    z = np.load(NORMALIZATION_NPZ)
    pred = z["pred"].astype(float).reshape(-1)
    true = z["true"].astype(float).reshape(-1)

    error = pred - true
    r2 = r2_score(true, pred)
    mn = min(true.min(), pred.min())
    mx = max(true.max(), pred.max())

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    axes[0].scatter(true, pred, s=20, alpha=0.6, label=f"R² = {r2:.4f}")
    axes[0].plot([mn, mx], [mn, mx], label="y=x reference")
    axes[0].set_xlabel("True", fontsize=16)
    axes[0].set_ylabel("Predicted", fontsize=16)
    axes[0].set_title("Validation: Predicted vs True Normalization Constant", fontsize=18)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(error, bins=20, alpha=0.8, histtype="step", lw=4)
    axes[1].set_xlabel("Error (pred - true)", fontsize=16)
    axes[1].set_ylabel("Count", fontsize=16)
    axes[1].set_title("Validation Error Histogram", fontsize=18)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / "figure_5.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# Figure 3: three example spectra
# -----------------------------------------------------------------------------

def plot_figure_3(true, pred, output_dir: Path, dpi: int):
    examples = [0, 1, 2]
    labels = ["a", "b", "c"]
    smiles = [
        "C#CC#CCOC",
        "CNC[C@@H](C)CO",
        "C=C(CC)NC=[N]",
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9.2), sharex=True)

    for ax, idx, label, smi in zip(axes, examples, labels, smiles):
        x = np.linspace(0, 0.45, len(true[idx]))
        corr = np.corrcoef(true[idx], pred[idx])[0, 1]

        ax.plot(x, pred[idx], label="Predicted", linewidth=1.8)
        ax.plot(x, true[idx], label="Reference", linewidth=1.2)
        ax.margins(y=0.05)

        ax.text(0.01, 0.83, f"{label}) {smi}", transform=ax.transAxes,
                ha="left", va="top", fontsize=11)
        ax.text(0.01, 0.72, f"Corr = {corr:.4g}", transform=ax.transAxes,
                ha="left", va="top", fontsize=11)

        inset = ax.inset_axes([0.08, 0.34, 0.29, 0.40])
        inset.imshow(molecule_image(smi))
        inset.axis("off")

        ax.set_ylabel("Normalized Intensity")
        ax.grid(True, alpha=0.15)

    axes[-1].set_xlabel("Frequency (a.u.)")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, loc="upper center",
               bbox_to_anchor=(0.85, 0.95), ncols=2)

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = output_dir / "figure_3.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# Figure 4: two similar molecules
# -----------------------------------------------------------------------------

def plot_figure_4(true, pred, output_dir: Path, dpi: int):
    smiles_a = "C#CC#CCCO"
    smiles_b = "C#CC#CCOC"

    i1, i2 = 0, 3
    t1, t2 = true[i1], true[i2]
    p1, p2 = pred[i1], pred[i2]
    x = np.linspace(0, 0.45, len(t1))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True)

    axes[0].plot(x, t1, linewidth=1.6, label=smiles_a)
    axes[0].plot(x, t2, linewidth=1.6, label=smiles_b)
    axes[0].set_ylabel("Normalized Intensity")
    axes[0].set_yticks([])
    axes[0].set_title("True spectra comparison")
    axes[0].legend(frameon=False, ncols=2, loc="upper left")
    axes[0].grid(True, alpha=0.15)

    axes[1].plot(x, p1, linewidth=1.6, label=smiles_a)
    axes[1].plot(x, p2, linewidth=1.6, label=smiles_b)
    axes[1].set_xlabel("Frequency (a.u.)")
    axes[1].set_ylabel("Normalized Intensity")
    axes[1].set_yticks([])
    axes[1].set_title("Predicted spectra comparison")
    axes[1].legend(frameon=False, ncols=2, loc="upper left")
    axes[1].grid(True, alpha=0.15)

    inset_a = axes[0].inset_axes([0.10, 0.48, 0.14, 0.34])
    inset_a.imshow(molecule_image(smiles_a, size=(350, 220)))
    inset_a.axis("off")

    inset_b = axes[0].inset_axes([0.26, 0.48, 0.14, 0.34])
    inset_b.imshow(molecule_image(smiles_b, size=(350, 220)))
    inset_b.axis("off")

    axes[0].text(0.17, 0.45, "A", transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=10)
    axes[0].text(0.33, 0.45, "B", transform=axes[0].transAxes,
                 ha="center", va="top", fontsize=10)

    fig.tight_layout()
    out = output_dir / "figure_4.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Create all plots from existing data.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    saved.append(plot_normalization(output_dir, args.dpi))

    true, pred = load_spectra()
    saved.append(plot_figure_3(true, pred, output_dir, args.dpi))
    saved.append(plot_figure_4(true, pred, output_dir, args.dpi))


    print("\nSaved plots:")
    for path in saved:
        print(" ", path)


if __name__ == "__main__":
    main()
