"""
inspect_dataset.py
==================
Mirrors the exact loading logic from run_ensemble (merge_and_filter_datasets)
but reports full statistics for both left-eye and right-eye directories,
plus produces diagnostic plots — all without needing to train anything.

Edit the CONFIG block below, then run:
    python inspect_dataset.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  ← edit these to match your local paths
# ─────────────────────────────────────────────────────────────────────────────

CSV_PATH      = "/kaggle/input/datasets/junaidgpu/imagehb/dataset/dataset/merge_excel_1.xlsx"
LEFT_DIR      = "/kaggle/input/datasets/junaidgpu/imagehb/dataset/dataset/left_eye"
RIGHT_DIR     = "/kaggle/input/datasets/junaidgpu/imagehb/dataset/dataset/right_eye"  # set "" to skip

IMAGE_COL     = "Patient ID"
HB_COL        = "Haemoglobin (gm/dL)"
HB_THRESH     = 12.0     # classification boundary (anemic < threshold)
HB_FILTER_MIN = 7.0      # outlier floor
HB_FILTER_MAX = 15.0     # outlier ceiling
AUG_BINS      = 10

SAVE_PLOT     = True
PLOT_PATH     = "dataset_inspection.png"   # saved next to this script

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ""]

def find_image_path(pid: str, img_dir: str):
    for ext in IMG_EXTS:
        p = os.path.join(img_dir, str(pid) + ext)
        if os.path.exists(p):
            return p
    return None


def count_images_in_dir(img_dir: str) -> int:
    if not img_dir or not os.path.isdir(img_dir):
        return 0
    total = 0
    for f in os.listdir(img_dir):
        if any(f.lower().endswith(e) for e in [".jpg", ".jpeg", ".png", ".bmp"]):
            total += 1
    return total


def analyse_side(label: str, df_raw: pd.DataFrame, img_dir: str) -> dict:
    """
    Run the same filtering logic as merge_and_filter_datasets for one eye side.
    Returns a dict of statistics plus the final clean DataFrame.
    """
    print(f"\n{'═'*60}")
    print(f"  {label.upper()} EYE  —  {img_dir or '(not configured)'}")
    print(f"{'═'*60}")

    if not img_dir:
        print("  Skipped (no directory set).")
        return None

    if not os.path.isdir(img_dir):
        print(f"  ✗ Directory not found: {img_dir}")
        return None

    # ── Physical images on disk ───────────────────────────────────────────
    n_disk = count_images_in_dir(img_dir)
    print(f"  Images on disk             : {n_disk}")

    # ── Work on a clean copy of the CSV ──────────────────────────────────
    df = df_raw.copy()
    n_csv_rows = len(df)

    # Drop non-numeric HB
    df[HB_COL] = pd.to_numeric(df[HB_COL], errors="coerce")
    n_bad_hb   = int(df[HB_COL].isna().sum())
    df         = df[df[HB_COL].notna()].copy()
    print(f"  CSV rows total             : {n_csv_rows}")
    print(f"  Rows with non-numeric HB   : {n_bad_hb}  (dropped)")
    print(f"  Rows with valid HB         : {len(df)}")

    # ── Match images ─────────────────────────────────────────────────────
    df["_img_path"] = df[IMAGE_COL].apply(lambda pid: find_image_path(str(pid), img_dir))
    n_matched  = int(df["_img_path"].notna().sum())
    n_missing  = int(df["_img_path"].isna().sum())

    missing_ids = df.loc[df["_img_path"].isna(), IMAGE_COL].tolist()
    print(f"\n  Patients matched to image  : {n_matched}")
    print(f"  Patients with no image     : {n_missing}  (dropped)")
    if missing_ids[:5]:
        print(f"    Examples: {missing_ids[:5]}")

    df = df[df["_img_path"].notna()].copy()

    # ── HB outlier filter ────────────────────────────────────────────────
    hb_all       = df[HB_COL].copy()
    n_before_flt = len(df)
    df_flt       = df[(df[HB_COL] >= HB_FILTER_MIN) & (df[HB_COL] <= HB_FILTER_MAX)].copy()
    n_outliers   = n_before_flt - len(df_flt)

    print(f"\n  HB filter [{HB_FILTER_MIN}, {HB_FILTER_MAX}] g/dL:")
    print(f"    Before filter              : {n_before_flt}")
    print(f"    Dropped as outliers        : {n_outliers}")
    print(f"    Kept (final usable)        : {len(df_flt)}")

    # ── HB statistics (filtered) ─────────────────────────────────────────
    hb = df_flt[HB_COL]
    n_anemic = int((hb < HB_THRESH).sum())
    n_normal = int((hb >= HB_THRESH).sum())
    print(f"\n  HB statistics (filtered):")
    print(f"    mean   : {hb.mean():.2f} g/dL")
    print(f"    std    : {hb.std():.2f} g/dL")
    print(f"    median : {hb.median():.2f} g/dL")
    print(f"    min    : {hb.min():.2f} g/dL")
    print(f"    max    : {hb.max():.2f} g/dL")
    print(f"\n  Class balance (threshold = {HB_THRESH} g/dL):")
    print(f"    Anemic  (HB < {HB_THRESH}) : {n_anemic:4d}  ({n_anemic/len(hb)*100:.1f}%)")
    print(f"    Normal  (HB ≥ {HB_THRESH}) : {n_normal:4d}  ({n_normal/len(hb)*100:.1f}%)")

    # ── Bin breakdown ─────────────────────────────────────────────────────
    bin_edges  = np.linspace(HB_FILTER_MIN, HB_FILTER_MAX, AUG_BINS + 1)
    counts, _  = np.histogram(hb.values, bins=bin_edges)
    mean_cnt   = counts.mean()
    print(f"\n  Bin breakdown ({AUG_BINS} bins, {HB_FILTER_MIN}–{HB_FILTER_MAX} g/dL):")
    for lo, hi, cnt in zip(bin_edges[:-1], bin_edges[1:], counts):
        flag = "  ← underrepresented" if cnt < mean_cnt * 0.7 else ""
        print(f"    [{lo:.1f}–{hi:.1f}): {cnt:5d}{flag}")

    return {
        "label"         : label,
        "n_disk"        : n_disk,
        "n_csv"         : n_csv_rows,
        "n_bad_hb"      : n_bad_hb,
        "n_matched"     : n_matched,
        "n_missing_img" : n_missing,
        "n_outliers"    : n_outliers,
        "n_final"       : len(df_flt),
        "n_anemic"      : n_anemic,
        "n_normal"      : n_normal,
        "hb_series"     : hb.values,       # filtered
        "hb_series_all" : hb_all.values,   # pre-filter (for plotting raw)
        "bin_edges"     : bin_edges,
        "bin_counts"    : counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_side(axes_row, stats: dict):
    """Fill one row of subplots for a single eye side."""
    hb_raw = stats["hb_series_all"]
    hb_f   = stats["hb_series"]
    label  = stats["label"]

    # ── Panel A: Histogram + KDE ─────────────────────────────────────────
    ax = axes_row[0]
    bins_hist = np.linspace(
        min(hb_raw.min(), HB_FILTER_MIN) - 0.5,
        max(hb_raw.max(), HB_FILTER_MAX) + 0.5,
        40,
    )
    ax.hist(hb_raw, bins=bins_hist, density=True, alpha=0.30,
            color="steelblue", label="All data")
    ax.hist(hb_f,   bins=bins_hist, density=True, alpha=0.45,
            color="orange",   label=f"Filtered [{HB_FILTER_MIN}–{HB_FILTER_MAX}]")
    if HAS_SCIPY and len(hb_raw) > 2:
        x = np.linspace(hb_raw.min(), hb_raw.max(), 300)
        ax.plot(x, gaussian_kde(hb_raw)(x), "b-",        lw=1.5, label="KDE all")
        ax.plot(x, gaussian_kde(hb_f)(x),   "darkorange", lw=1.5,
                linestyle="--", label="KDE filtered")
    ax.axvline(HB_THRESH,      color="red",   linestyle="--", lw=1.5,
               label=f"Threshold ({HB_THRESH})")
    ax.axvline(hb_f.mean(),    color="green", linestyle=":",  lw=1.5,
               label=f"Mean ({hb_f.mean():.1f})")
    ax.axvspan(HB_FILTER_MIN, HB_FILTER_MAX, alpha=0.07, color="limegreen")
    ax.set_xlabel("HB (g/dL)")
    ax.set_title(f"{label} Eye — Distribution")
    ax.legend(fontsize=7)

    # ── Panel B: Bin counts ───────────────────────────────────────────────
    ax = axes_row[1]
    centres   = (stats["bin_edges"][:-1] + stats["bin_edges"][1:]) / 2
    bwidth    = (stats["bin_edges"][1] - stats["bin_edges"][0]) * 0.85
    mean_cnt  = stats["bin_counts"].mean()
    bar_cols  = ["salmon" if c < mean_cnt * 0.7 else "steelblue"
                 for c in stats["bin_counts"]]
    ax.bar(centres, stats["bin_counts"], width=bwidth,
           color=bar_cols, edgecolor="white")
    ax.axhline(mean_cnt,  color="red",    linestyle="--", lw=1.5,
               label=f"Mean ({mean_cnt:.0f})")
    ax.axvline(HB_THRESH, color="orange", linestyle="--", lw=1.2,
               label=f"Threshold ({HB_THRESH})")
    ax.set_xlabel("HB (g/dL)")
    ax.set_ylabel("Count")
    ax.set_title(f"{label} Eye — Samples per bin\n(red = underrepresented)")
    ax.legend(fontsize=7)

    # ── Panel C: Class balance ────────────────────────────────────────────
    ax = axes_row[2]
    vals  = [stats["n_anemic"], stats["n_normal"]]
    total = sum(vals)
    xlbls = [f"Anemic\n(<{HB_THRESH:.0f})", f"Normal\n(≥{HB_THRESH:.0f})"]
    bars  = ax.bar(xlbls, vals, color=["salmon", "steelblue"], edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.01,
                f"{v}\n({v/total*100:.1f}%)", ha="center", fontsize=9)
    ax.set_title(f"{label} Eye — Class Balance")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(vals) * 1.30)

    # ── Panel D: Drop funnel ──────────────────────────────────────────────
    ax = axes_row[3]
    stages = [
        ("CSV rows",     stats["n_csv"]),
        ("Valid HB",     stats["n_csv"] - stats["n_bad_hb"]),
        ("Image found",  stats["n_matched"]),
        ("After filter", stats["n_final"]),
    ]
    stage_lbls = [s[0] for s in stages]
    stage_vals = [s[1] for s in stages]
    palette    = ["#4C9BE8", "#5CB85C", "#F0AD4E", "#D9534F"]
    ax.barh(stage_lbls[::-1], stage_vals[::-1], color=palette[::-1],
            edgecolor="white")
    for i, v in enumerate(stage_vals[::-1]):
        ax.text(v + max(stage_vals) * 0.01, i, str(v),
                va="center", fontsize=9)
    ax.set_xlabel("Count")
    ax.set_title(f"{label} Eye — Sample Funnel\n(rows→valid HB→image found→kept)")
    ax.set_xlim(0, max(stage_vals) * 1.20)


def plot_combined_summary(left: dict, right: dict):
    """
    Side-by-side summary bar chart if both sides have data.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    metrics = ["n_disk", "n_matched", "n_outliers", "n_final",
               "n_anemic", "n_normal"]
    labels  = ["Images\non disk", "Matched\nto HB", "Outliers\ndropped",
               "Final\nusable", "Anemic", "Normal"]
    x      = np.arange(len(metrics))
    width  = 0.35

    left_vals  = [left[m]  if left  else 0 for m in metrics]
    right_vals = [right[m] if right else 0 for m in metrics]

    ax.bar(x - width/2, left_vals,  width, label="Left eye",  color="#4C9BE8", edgecolor="white")
    ax.bar(x + width/2, right_vals, width, label="Right eye", color="#F0AD4E", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Left vs Right Eye — Dataset Summary")
    ax.legend()
    for i, (lv, rv) in enumerate(zip(left_vals, right_vals)):
        if lv: ax.text(i - width/2, lv + max(max(left_vals), max(right_vals))*0.01,
                       str(lv), ha="center", fontsize=8)
        if rv: ax.text(i + width/2, rv + max(max(left_vals), max(right_vals))*0.01,
                       str(rv), ha="center", fontsize=8)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*60)
    print("  ImageHB DATASET INSPECTOR")
    print("  (mirrors run_ensemble load_data logic)")
    print("═"*60)

    # ── Load spreadsheet ──────────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        print(f"\n✗ CSV not found: {CSV_PATH}")
        print("  → Update CSV_PATH in the CONFIG block at the top of this file.")
        sys.exit(1)

    print(f"\n  Loading CSV: {CSV_PATH}")
    df_raw = (pd.read_excel(CSV_PATH)
              if CSV_PATH.endswith((".xlsx", ".xls"))
              else pd.read_csv(CSV_PATH))
    print(f"  Columns     : {list(df_raw.columns)}")
    print(f"  Total rows  : {len(df_raw)}")

    # ── Analyse each side ─────────────────────────────────────────────────
    left_stats  = analyse_side("Left",  df_raw, LEFT_DIR)
    right_stats = analyse_side("Right", df_raw, RIGHT_DIR)

    # ── Overall summary ───────────────────────────────────────────────────
    active = [s for s in [left_stats, right_stats] if s is not None]
    if active:
        total_final = sum(s["n_final"] for s in active)
        total_out   = sum(s["n_outliers"] for s in active)
        print(f"\n{'═'*60}")
        print(f"  COMBINED SUMMARY")
        print(f"{'═'*60}")
        for s in active:
            print(f"  {s['label']:5s} — images on disk: {s['n_disk']:4d} | "
                  f"matched to HB: {s['n_matched']:4d} | "
                  f"outliers dropped: {s['n_outliers']:3d} | "
                  f"final usable: {s['n_final']:4d}")
        print(f"  Total final samples (all sides): {total_final}")
        print(f"  Total outliers dropped          : {total_out}")

    # ── Plots ─────────────────────────────────────────────────────────────
    n_rows = len(active)
    if n_rows == 0:
        print("\n  No data to plot.")
        return

    fig = plt.figure(figsize=(20, 5 * n_rows))
    fig.suptitle(
        "ImageHB Dataset Inspection\n"
        f"(HB filter [{HB_FILTER_MIN}–{HB_FILTER_MAX}] g/dL,  "
        f"threshold {HB_THRESH} g/dL)",
        fontsize=13, y=1.01,
    )
    gs = gridspec.GridSpec(n_rows, 4, figure=fig, hspace=0.55, wspace=0.35)

    for row_idx, stats in enumerate(active):
        axes_row = [fig.add_subplot(gs[row_idx, col]) for col in range(4)]
        plot_side(axes_row, stats)

    plt.tight_layout()
    if SAVE_PLOT:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), PLOT_PATH)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"\n  Plot saved → {out_path}")
    plt.show()

    # Combined left-vs-right bar chart (only if both sides ran)
    if left_stats and right_stats:
        fig2 = plot_combined_summary(left_stats, right_stats)
        if SAVE_PLOT:
            p2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "dataset_left_vs_right.png")
            fig2.savefig(p2, dpi=120, bbox_inches="tight")
            print(f"  Combined plot saved → {p2}")
        plt.show()


if __name__ == "__main__":
    main()
