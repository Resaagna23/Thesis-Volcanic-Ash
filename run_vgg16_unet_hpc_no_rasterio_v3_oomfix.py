#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HPC-ready VGG16 + U-Net workflow for the Mount Lewotobi thesis.

Method preserved from:
Tunning_VGG16_UNet_5Fold_CV_DrNattapon_Updated.ipynb

Workflow:
1. Match Himawari-9 Ash RGB images and binary labels by timestamp.
2. Lock Phases 14-16 as the independent test set.
3. Use phase-grouped 5-fold CV on Phases 1-13.
4. Evaluate an equal tuning budget of VGG16+U-Net configurations.
5. Keep BatchNormalization OFF because batch size is 1.
6. Calibrate a validation-only segmentation threshold in every fold.
7. Report Dice/F1 and AUPRC.
8. Re-evaluate the best configuration with two-stage transfer learning:
      Stage 1: frozen VGG16 encoder
      Stage 2: fine-tune VGG16 Block 5 only
9. Select final epoch counts and threshold from the medians of the five folds.
10. Train on all Phase 1-13 images.
11. Evaluate once on locked Phases 14-16.

Designed for Linux/HPC batch execution, including Lenovo LiCO TensorFlow jobs.\nV3 adds GPU OOM mitigation using cuda_malloc_async and a clean model/session\nrebuild between Stage 1 and Stage 2 fine-tuning.
"""

import os

# Set these BEFORE importing TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

# OOM FIX V3:
# cuda_malloc_async can reduce GPU-memory fragmentation during the
# Stage-1 -> Stage-2 transfer-learning transition.
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import argparse
import gc
import json
import math
import random
import re
import socket
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)

from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ReduceLROnPlateau,
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dropout,
    Input,
    UpSampling2D,
)
from tensorflow.keras.optimizers import Adam


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase-grouped 5-fold VGG16+U-Net volcanic-cloud segmentation."
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help=(
            "Project data directory containing image/, label/, and manual_split.csv. "
            "Example: /home/user/volcanic_ash/data"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: <base-dir>/vgg16_unet_5fold_cv_drnattapon_output"
        ),
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Equal-budget hyperparameter candidates. Thesis default: 30.",
    )

    parser.add_argument(
        "--cv-tuning-max-epochs",
        type=int,
        default=20,
        help="Maximum epochs for each tuning fold. Thesis default: 20.",
    )

    parser.add_argument(
        "--stage1-max-epochs",
        type=int,
        default=40,
        help="Maximum frozen-encoder epochs in best-hyperparameter CV.",
    )

    parser.add_argument(
        "--stage2-max-epochs",
        type=int,
        default=20,
        help="Maximum Block-5 fine-tuning epochs in best-hyperparameter CV.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size. Thesis design uses 1.",
    )

    parser.add_argument(
        "--translation-pixels",
        type=int,
        default=16,
        help="Maximum spatial translation augmentation in pixels.",
    )

    parser.add_argument(
        "--vgg16-weights",
        type=str,
        default="imagenet",
        help=(
            "Use 'imagenet' if the runtime has cached weights/internet, "
            "or provide an absolute path to a local VGG16 no-top weights file."
        ),
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Quick pipeline test: 1 tuning candidate, 1 tuning epoch, "
            "2/1 epochs in the two-stage CV. Do NOT use these results in the thesis."
        ),
    )

    return parser.parse_args()


ARGS = parse_args()

# ============================================================
# HPC / RUNTIME CONFIGURATION
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = ARGS.base_dir.expanduser().resolve()
IMAGE_DIR = BASE_DIR / "image"
LABEL_DIR = BASE_DIR / "label"
MANUAL_SPLIT_PATH = BASE_DIR / "manual_split.csv"

OUTPUT_DIR = (
    ARGS.output_dir.expanduser().resolve()
    if ARGS.output_dir is not None
    else BASE_DIR / "vgg16_unet_5fold_cv_drnattapon_output"
)

DATA_LIST_DIR = OUTPUT_DIR / "data_lists"
CV_DIR = OUTPUT_DIR / "cross_validation"
TUNING_DIR = OUTPUT_DIR / "hyperparameter_tuning"
MODEL_DIR = OUTPUT_DIR / "models"
PREDICTION_DIR = OUTPUT_DIR / "test_predictions"
ERROR_MAP_DIR = OUTPUT_DIR / "test_error_maps"
PLOT_DIR = OUTPUT_DIR / "plots"
LOG_DIR = OUTPUT_DIR / "logs"

for folder in [
    OUTPUT_DIR,
    DATA_LIST_DIR,
    CV_DIR,
    TUNING_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
    ERROR_MAP_DIR,
    PLOT_DIR,
    LOG_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

TARGET_HEIGHT = 256
TARGET_WIDTH = 256
INPUT_CHANNELS = 3

CV_BATCH_SIZE = int(ARGS.batch_size)
FINAL_BATCH_SIZE = int(ARGS.batch_size)

N_CV_TRIALS = int(ARGS.trials)
CV_TUNING_MAX_EPOCHS = int(ARGS.cv_tuning_max_epochs)
CV_TUNING_PATIENCE = 5

BEST_HP_STAGE1_MAX_EPOCHS = int(ARGS.stage1_max_epochs)
BEST_HP_STAGE2_MAX_EPOCHS = int(ARGS.stage2_max_epochs)
BEST_HP_PATIENCE_STAGE1 = 8
BEST_HP_PATIENCE_STAGE2 = 6

if ARGS.smoke_test:
    N_CV_TRIALS = 1
    CV_TUNING_MAX_EPOCHS = 1
    BEST_HP_STAGE1_MAX_EPOCHS = 2
    BEST_HP_STAGE2_MAX_EPOCHS = 1
    BEST_HP_PATIENCE_STAGE1 = 1
    BEST_HP_PATIENCE_STAGE2 = 1

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.025), 3)
FINAL_PREDICTION_THRESHOLD = None

AUGMENT_TRANSLATION_PIXELS = int(ARGS.translation_pixels)

# Conservative defaults are stable on shared HPC filesystems.
DATA_PARALLEL_CALLS = 1
PREFETCH_BUFFER = 1

FOLD_TO_PHASES = {
    1: [1, 13],
    2: [2, 12],
    3: [3, 7, 9],
    4: [4, 5, 8, 11],
    5: [6, 10],
}

PHASE_TO_FOLD = {
    phase: fold
    for fold, phases in FOLD_TO_PHASES.items()
    for phase in phases
}

EXPECTED_PHASE_COUNTS = {
    1: 26, 2: 37, 3: 17, 4: 14, 5: 10, 6: 9, 7: 11,
    8: 18, 9: 18, 10: 42, 11: 7, 12: 10, 13: 26,
    14: 29, 15: 6, 16: 13,
}

# Keras accepts either "imagenet" or a local weights filepath.
if ARGS.vgg16_weights.lower() == "imagenet":
    VGG16_WEIGHTS = "imagenet"
else:
    weights_path = Path(ARGS.vgg16_weights).expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Local VGG16 weights not found: {weights_path}")
    VGG16_WEIGHTS = str(weights_path)

# Avoid TensorFlow grabbing all GPU memory at process start.
physical_gpus = tf.config.list_physical_devices("GPU")
for gpu in physical_gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

if TARGET_HEIGHT % 32 != 0 or TARGET_WIDTH % 32 != 0:
    raise ValueError("TARGET_HEIGHT and TARGET_WIDTH must be divisible by 32.")

for required_path in [IMAGE_DIR, LABEL_DIR]:
    if not required_path.exists():
        raise FileNotFoundError(f"Directory not found: {required_path}")

if not MANUAL_SPLIT_PATH.exists():
    raise FileNotFoundError(f"manual_split.csv not found: {MANUAL_SPLIT_PATH}")

print("=" * 100)
print("HPC VGG16 + U-NET / PHASE-GROUPED 5-FOLD CV")
print("TIFF reader           : Pillow (rasterio not required)")
print("HPC script version    : v3-oomfix")
print("GPU allocator         :", os.environ.get("TF_GPU_ALLOCATOR", "default"))
print("=" * 100)
print("Hostname              :", socket.gethostname())
print("Python                :", sys.version.replace("\n", " "))
print("TensorFlow            :", tf.__version__)
print("Visible GPU(s)        :", physical_gpus)
print("Base directory        :", BASE_DIR)
print("Image directory       :", IMAGE_DIR)
print("Label directory       :", LABEL_DIR)
print("Manual split          :", MANUAL_SPLIT_PATH)
print("Output directory      :", OUTPUT_DIR)
print("Target size           :", TARGET_HEIGHT, "x", TARGET_WIDTH)
print("Batch size            :", CV_BATCH_SIZE)
print("CV tuning trials      :", N_CV_TRIALS)
print("CV tuning max epochs  :", CV_TUNING_MAX_EPOCHS)
print("VGG16 weights         :", VGG16_WEIGHTS)
print("Smoke test            :", ARGS.smoke_test)
print("=" * 100)

if not physical_gpus:
    print(
        "WARNING: No GPU is visible to TensorFlow. The workflow can run on CPU, "
        "but VGG16+U-Net tuning will be very slow."
    )


# ============================================================
# CELL 4 - TIMESTAMP AND PHASE FUNCTIONS
# ============================================================

def extract_timestamp(filename: str):
    """
    Extract timestamp YYYYMMDD_HHMM from filename.
    
    Example:
    AshRGB_Lewotobi_20250618_0000.tif
    -> 20250618_0000
    """
    
    match = re.search(r"(\d{8}_\d{4})", filename)
    
    return match.group(1) if match else None


def timestamp_to_phase(timestamp: str):
    """
    Assign each timestamp to eruption Phase 1-16.
    
    IMPORTANT:
    20250618_0000 belongs to Phase 14
    20250802_0000 belongs to Phase 16
    """
    
    dt = pd.to_datetime(
        timestamp,
        format="%Y%m%d_%H%M"
    )

    ranges = [

        # ====================================================
        # TRAINING DATA
        # ====================================================

        (
            1,
            "2024-11-08 21:30",
            "2024-11-09 10:00"
        ),

        (
            2,
            "2024-11-09 10:30",
            "2024-11-10 04:30"
        ),

        (
            3,
            "2024-11-10 05:00",
            "2024-11-10 13:00"
        ),

        (
            4,
            "2024-11-10 13:30",
            "2024-11-10 20:00"
        ),

        (
            5,
            "2024-11-11 03:00",
            "2024-11-11 07:30"
        ),

        (
            6,
            "2024-11-11 08:00",
            "2024-11-11 12:00"
        ),

        (
            7,
            "2024-11-11 12:30",
            "2024-11-11 17:30"
        ),

        (
            8,
            "2024-11-11 18:00",
            "2024-11-12 02:30"
        ),

        (
            9,
            "2024-11-12 03:00",
            "2024-11-12 11:30"
        ),

        (
            10,
            "2024-11-12 12:00",
            "2024-11-13 08:30"
        ),

        (
            11,
            "2024-11-13 09:00",
            "2024-11-13 12:00"
        ),

        (
            12,
            "2024-11-13 14:30",
            "2024-11-13 19:00"
        ),

        (
            13,
            "2024-11-13 19:30",
            "2024-11-14 08:00"
        ),


        # ====================================================
        # TEST DATA
        # ====================================================

        # Phase 14
        # 17 Jun 2025 10:00
        # through
        # 18 Jun 2025 00:00
        (
            14,
            "2025-06-17 10:00",
            "2025-06-18 00:00"
        ),

        # Phase 15
        (
            15,
            "2025-08-01 15:00",
            "2025-08-01 17:30"
        ),

        # Phase 16
        # 1 Aug 2025 18:00
        # through
        # 2 Aug 2025 00:00
        (
            16,
            "2025-08-01 18:00",
            "2025-08-02 00:00"
        ),
    ]


    # ========================================================
    # FIND MATCHING PHASE
    # ========================================================

    for phase, start, end in ranges:

        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        if start_dt <= dt <= end_dt:

            return phase


    # ========================================================
    # NOT ASSIGNED
    # ========================================================

    return None

# ============================================================
# CELL 5 - FIND, MATCH, AND SPLIT IMAGE-LABEL PAIRS
# ============================================================

image_files = sorted(
    [p.resolve() for p in IMAGE_DIR.iterdir()
     if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}],
    key=lambda p: p.name.lower(),
)

label_files = sorted(
    [p.resolve() for p in LABEL_DIR.iterdir()
     if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".png"}],
    key=lambda p: p.name.lower(),
)

if not image_files:
    raise FileNotFoundError(f"No TIFF images found in {IMAGE_DIR}")
if not label_files:
    raise FileNotFoundError(f"No label files found in {LABEL_DIR}")

image_groups = defaultdict(list)
for p in image_files:
    ts = extract_timestamp(p.name)
    if ts:
        image_groups[ts].append(p)

label_groups = defaultdict(list)
for p in label_files:
    ts = extract_timestamp(p.name)
    if ts:
        label_groups[ts].append(p)

image_dictionary = {ts: sorted(paths, key=lambda p: p.name.lower())[0]
                    for ts, paths in image_groups.items()}
label_dictionary = {ts: sorted(paths, key=lambda p: p.name.lower())[0]
                    for ts, paths in label_groups.items()}

matched_timestamps = sorted(set(image_dictionary) & set(label_dictionary))
if not matched_timestamps:
    raise ValueError("No matching image-label pairs were found.")

pairs_df = pd.DataFrame([
    {
        "timestamp": ts,
        "image_path": str(image_dictionary[ts]),
        "label_path": str(label_dictionary[ts]),
    }
    for ts in matched_timestamps
])

manual_split_df = pd.read_csv(
    MANUAL_SPLIT_PATH,
    dtype={"timestamp": str, "split": str},
)

required_columns = {"timestamp", "split"}
missing_columns = required_columns - set(manual_split_df.columns)
if missing_columns:
    raise ValueError(
        "manual_split.csv is missing columns: " + ", ".join(sorted(missing_columns))
    )

manual_split_df["timestamp"] = (
    manual_split_df["timestamp"].astype(str).str.strip()
    .str.replace(r"\.0$", "", regex=True)
)
manual_split_df["split"] = manual_split_df["split"].astype(str).str.strip().str.lower()
manual_split_df["split"] = manual_split_df["split"].map({
    "train": "train", "training": "train", "tr": "train",
    "test": "test", "testing": "test", "te": "test",
})

if manual_split_df["split"].isna().any():
    raise ValueError("manual_split.csv split must contain only train/test values.")

if manual_split_df["timestamp"].duplicated().any():
    raise ValueError("Duplicate timestamps found in manual_split.csv.")

split_df = pairs_df.merge(
    manual_split_df[["timestamp", "split"]],
    on="timestamp",
    how="left",
    validate="one_to_one",
)

if split_df["split"].isna().any():
    bad = split_df.loc[split_df["split"].isna(), ["timestamp", "image_path"]]
    print(bad)
    raise ValueError("Some matched pairs have no train/test assignment.")

split_df["phase"] = split_df["timestamp"].apply(timestamp_to_phase)

if split_df["phase"].isna().any():
    bad = split_df.loc[split_df["phase"].isna(), ["timestamp", "split"]]
    print(bad)
    raise ValueError("Some images could not be assigned to Phase 1-16.")

split_df["phase"] = split_df["phase"].astype(int)
split_df["phase_name"] = "Phase " + split_df["phase"].astype(str)

# Strict leakage checks.
bad_train = split_df[(split_df["split"] == "train") & (~split_df["phase"].between(1, 13))]
bad_test = split_df[(split_df["split"] == "test") & (~split_df["phase"].between(14, 16))]

if not bad_train.empty or not bad_test.empty:
    print("Unexpected train rows:")
    print(bad_train[["timestamp", "phase", "split"]])
    print("Unexpected test rows:")
    print(bad_test[["timestamp", "phase", "split"]])
    raise ValueError(
        "Train must contain only Phases 1-13 and test only Phases 14-16."
    )

split_df["fold"] = pd.Series(pd.NA, index=split_df.index, dtype="Int64")
train_mask = split_df["split"] == "train"
split_df.loc[train_mask, "fold"] = (
    split_df.loc[train_mask, "phase"].map(PHASE_TO_FOLD).astype("Int64")
)

if split_df.loc[train_mask, "fold"].isna().any():
    raise ValueError("At least one training phase has no fold assignment.")

train_df = split_df[split_df["split"] == "train"].copy().reset_index(drop=True)
test_df = split_df[split_df["split"] == "test"].copy().reset_index(drop=True)

split_df.to_csv(DATA_LIST_DIR / "all_pairs_with_phase_and_fold.csv", index=False)
train_df.to_csv(DATA_LIST_DIR / "training_files_with_folds.csv", index=False)
test_df.to_csv(DATA_LIST_DIR / "testing_files_with_phases.csv", index=False)

print("=" * 80)
print("TRAIN / TEST SPLIT")
print("=" * 80)
print("Training images:", len(train_df))
print("Testing images :", len(test_df))
print("Total images   :", len(split_df))

# ============================================================
# CELL 6 - READ IMAGE AND BINARY LABEL
# ============================================================


def _decode_tf_path(path_value):
    """
    Convert a path received from tf.numpy_function / tf.py_function
    into a normal Python string.

    TensorFlow commonly passes tf.string values to NumPy/Python as:
      - bytes
      - numpy.bytes_
      - 0-D numpy.ndarray containing bytes
    """
    if isinstance(path_value, np.ndarray):
        if path_value.shape == ():
            path_value = path_value.item()
        elif path_value.size == 1:
            path_value = path_value.reshape(()).item()

    if isinstance(path_value, np.bytes_):
        path_value = bytes(path_value)

    if isinstance(path_value, bytes):
        path_value = path_value.decode("utf-8")

    return str(path_value)


def read_tiff_rgb(image_path):
    """
    Read a 3-channel TIFF image without rasterio.

    Uses Pillow first, which is commonly available in TensorFlow containers.
    The output is float32 with shape (H, W, 3).

    If the TIFF contains:
    - 1 channel: it is repeated to 3 channels.
    - >3 channels: only the first 3 channels are used.
    """

    image_path = Path(_decode_tf_path(image_path))

    with Image.open(image_path) as img:
        image = np.array(img)

    if image.ndim == 2:
        image = np.stack(
            [image, image, image],
            axis=-1,
        )

    elif image.ndim == 3:
        if image.shape[-1] == 1:
            image = np.repeat(
                image,
                3,
                axis=-1,
            )
        elif image.shape[-1] > 3:
            image = image[:, :, :3]

    else:
        raise ValueError(
            f"Unexpected image shape for {image_path}: {image.shape}"
        )

    image = image.astype(np.float32)

    # Clean invalid values if any
    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )

    # If image values look normalized to 0-1, convert to 0-255
    finite_max = float(np.max(image)) if image.size else 0.0

    if finite_max <= 1.5:
        image = image * 255.0

    image = np.clip(
        image,
        0.0,
        255.0,
    )

    return image


def read_binary_label(label_path) -> np.ndarray:
    """
    Return H x W x 1 float32 binary mask without rasterio:
      0 = non-volcanic cloud
      1 = volcanic cloud

    Supported label forms:
      - single-band 0/1
      - single-band 0/255
      - RGB red=positive / blue=negative
    """

    label_path = Path(_decode_tf_path(label_path))

    with Image.open(label_path) as img:
        label_array = np.array(img)

    if label_array.ndim == 3:
        if label_array.shape[-1] == 1:
            arr = label_array[:, :, 0].astype(np.float32)

            max_value = (
                float(np.nanmax(arr))
                if arr.size
                else 0.0
            )

            if max_value <= 1.0:
                binary_label = (arr >= 0.5).astype(np.float32)
            else:
                binary_label = (arr >= 128.0).astype(np.float32)

        else:
            rgb = label_array[:, :, :3].astype(np.float32)

            red = rgb[:, :, 0]
            green = rgb[:, :, 1]
            blue = rgb[:, :, 2]

            # Red-dominant pixels are the positive volcanic-cloud class.
            binary_label = (
                (red > green)
                & (red > blue)
                & (red >= 128)
            ).astype(np.float32)

    elif label_array.ndim == 2:
        arr = label_array.astype(np.float32)

        max_value = (
            float(np.nanmax(arr))
            if arr.size
            else 0.0
        )

        if max_value <= 1.0:
            binary_label = (arr >= 0.5).astype(np.float32)
        else:
            binary_label = (arr >= 128.0).astype(np.float32)

    else:
        raise ValueError(
            f"Unexpected label shape for {label_path}: {label_array.shape}"
        )

    return binary_label[..., np.newaxis].astype(np.float32)


def label_pixel_counts(label_path):
    mask = read_binary_label(label_path)[:, :, 0]
    ash = int(np.sum(mask == 1))
    non_ash = int(np.sum(mask == 0))
    total = ash + non_ash
    return ash, non_ash, total


pixel_stats = []
for i, row in split_df.iterrows():
    ash, non_ash, total = label_pixel_counts(row["label_path"])
    pixel_stats.append((ash, non_ash, total))

split_df[["ash_pixels", "non_ash_pixels", "total_pixels"]] = pd.DataFrame(
    pixel_stats, index=split_df.index
)

split_df["ash_percent"] = (
    split_df["ash_pixels"] / split_df["total_pixels"].clip(lower=1) * 100.0
)

train_df = split_df[split_df["split"] == "train"].copy().reset_index(drop=True)
test_df = split_df[split_df["split"] == "test"].copy().reset_index(drop=True)

phase_summary = (
    split_df.groupby(["phase", "phase_name", "split"], as_index=False)
    .agg(
        number_of_images=("timestamp", "count"),
        total_pixels=("total_pixels", "sum"),
        ash_pixels=("ash_pixels", "sum"),
        non_ash_pixels=("non_ash_pixels", "sum"),
        mean_image_ash_percent=("ash_percent", "mean"),
        std_image_ash_percent=("ash_percent", "std"),
    )
)
phase_summary["phase_ash_percent"] = (
    phase_summary["ash_pixels"] / phase_summary["total_pixels"].clip(lower=1) * 100
)
phase_summary["phase_non_ash_percent"] = 100 - phase_summary["phase_ash_percent"]
phase_summary.to_csv(DATA_LIST_DIR / "phase_summary.csv", index=False)

fold_rows = []
for fold, phases in FOLD_TO_PHASES.items():
    fold_df = train_df[train_df["fold"] == fold]
    total = int(fold_df["total_pixels"].sum())
    ash = int(fold_df["ash_pixels"].sum())
    fold_rows.append({
        "fold": fold,
        "validation_phases": ", ".join(f"Phase {p}" for p in phases),
        "validation_images": len(fold_df),
        "training_images": len(train_df) - len(fold_df),
        "validation_ash_percent": ash / max(total, 1) * 100,
        "validation_non_ash_percent": 100 - ash / max(total, 1) * 100,
    })

fold_summary = pd.DataFrame(fold_rows)
fold_summary.to_csv(DATA_LIST_DIR / "fold_summary.csv", index=False)

print("\nPHASE SUMMARY")
print(phase_summary[[
    "phase_name", "split", "number_of_images",
    "phase_ash_percent", "phase_non_ash_percent"
]].round(3).to_string(index=False))

print("\nFOLD SUMMARY")
print(fold_summary.round(3).to_string(index=False))

print("\nCOUNT CHECK")
actual_counts = split_df.groupby("phase").size().to_dict()
for phase in range(1, 17):
    actual = actual_counts.get(phase, 0)
    expected = EXPECTED_PHASE_COUNTS.get(phase)
    status = "OK" if actual == expected else "CHECK"
    print(f"Phase {phase:2d}: actual={actual:3d} expected={expected:3d}  {status}")

# ============================================================
# CELL 8 - TF.DATA PIPELINE
# ============================================================

def load_image_and_label(image_path, label_path):
    image = tf.numpy_function(read_tiff_rgb, [image_path], tf.float32)
    label = tf.numpy_function(read_binary_label, [label_path], tf.float32)

    image.set_shape([None, None, INPUT_CHANNELS])
    label.set_shape([None, None, 1])

    image = tf.image.resize(
        image,
        [TARGET_HEIGHT, TARGET_WIDTH],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=True,
    )
    label = tf.image.resize(
        label,
        [TARGET_HEIGHT, TARGET_WIDTH],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    )

    label = tf.where(label >= 0.5, 1.0, 0.0)

    image = tf.ensure_shape(image, [TARGET_HEIGHT, TARGET_WIDTH, INPUT_CHANNELS])
    label = tf.ensure_shape(label, [TARGET_HEIGHT, TARGET_WIDTH, 1])

    return image, label


def augment_image_and_label(image, label):
    """
    Spatial augmentation only.

    1) Horizontal flip is applied jointly to image and label.
    2) Random translation is implemented with padding + a matched crop.

    No independent brightness perturbation is used here because the comparison
    should preserve the same Ash RGB source values across model families.
    """
    flip_horizontal = tf.random.uniform([]) > 0.5
    image = tf.cond(
        flip_horizontal,
        lambda: tf.image.flip_left_right(image),
        lambda: image,
    )
    label = tf.cond(
        flip_horizontal,
        lambda: tf.image.flip_left_right(label),
        lambda: label,
    )

    pad = int(AUGMENT_TRANSLATION_PIXELS)
    if pad > 0:
        image_padded = tf.pad(
            image,
            paddings=[[pad, pad], [pad, pad], [0, 0]],
            mode="REFLECT",
        )
        label_padded = tf.pad(
            label,
            paddings=[[pad, pad], [pad, pad], [0, 0]],
            mode="CONSTANT",
            constant_values=0.0,
        )

        max_offset = 2 * pad
        offset_y = tf.random.uniform(
            [], minval=0, maxval=max_offset + 1, dtype=tf.int32
        )
        offset_x = tf.random.uniform(
            [], minval=0, maxval=max_offset + 1, dtype=tf.int32
        )

        image = tf.slice(
            image_padded,
            [offset_y, offset_x, 0],
            [TARGET_HEIGHT, TARGET_WIDTH, INPUT_CHANNELS],
        )
        label = tf.slice(
            label_padded,
            [offset_y, offset_x, 0],
            [TARGET_HEIGHT, TARGET_WIDTH, 1],
        )

    image = tf.ensure_shape(image, [TARGET_HEIGHT, TARGET_WIDTH, INPUT_CHANNELS])
    label = tf.ensure_shape(label, [TARGET_HEIGHT, TARGET_WIDTH, 1])
    return image, label


def apply_vgg16_preprocessing(image, label):
    # Model-specific normalization required by ImageNet-pretrained VGG16.
    # The underlying Ash RGB source image remains the same.
    return preprocess_input(image), label


def create_dataset(dataframe, training, batch_size):
    image_paths = dataframe["image_path"].astype(str).values
    label_paths = dataframe["label_path"].astype(str).values

    ds = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))

    if training:
        ds = ds.shuffle(
            buffer_size=max(len(dataframe), 1),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(load_image_and_label, num_parallel_calls=DATA_PARALLEL_CALLS)

    if training:
        ds = ds.map(augment_image_and_label, num_parallel_calls=DATA_PARALLEL_CALLS)

    ds = ds.map(apply_vgg16_preprocessing, num_parallel_calls=DATA_PARALLEL_CALLS)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(PREFETCH_BUFFER)
    return ds

# ============================================================
# CELL 9 - LOSS AND TRAINING METRICS
# ============================================================

SMOOTH = 1e-6

def dice_coefficient(y_true, y_pred):
    """Soft Dice used only as the training/early-stopping metric."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    denominator = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)
    return (2.0 * intersection + SMOOTH) / (denominator + SMOOTH)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


binary_crossentropy_loss = tf.keras.losses.BinaryCrossentropy()


def combined_bce_dice_loss(y_true, y_pred):
    return binary_crossentropy_loss(y_true, y_pred) + dice_loss(y_true, y_pred)


def compile_model(model, learning_rate):
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=combined_bce_dice_loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            dice_coefficient,
        ],
    )
    return model

# ============================================================
# CELL 10 - VGG16 + U-NET MODEL
# ============================================================

def decoder_conv_block(inputs, filters, dropout_rate, use_batch_norm=False):
    x = Conv2D(
        filters, 3, padding="same", activation="relu",
        kernel_initializer="he_normal"
    )(inputs)
    if use_batch_norm:
        x = BatchNormalization()(x)

    x = Conv2D(
        filters, 3, padding="same", activation="relu",
        kernel_initializer="he_normal"
    )(x)
    if use_batch_norm:
        x = BatchNormalization()(x)

    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    return x


def build_vgg16_unet(params):
    tf.keras.backend.clear_session()

    f0 = int(params["decoder_initial_filters"])
    dropout_shallow = float(params["dropout_shallow"])
    dropout_deep = float(params["dropout_deep"])
    dropout_bottleneck = float(params["dropout_bottleneck"])
    # Batch normalization is fixed OFF because CV_BATCH_SIZE = 1.
    use_batch_norm = False
    interpolation = str(params["upsampling_interpolation"])
    learning_rate = float(params["learning_rate"])

    inputs = Input(
        shape=(TARGET_HEIGHT, TARGET_WIDTH, INPUT_CHANNELS),
        name="input_image"
    )

    encoder = VGG16(
        include_top=False,
        weights=VGG16_WEIGHTS,
        input_tensor=inputs,
    )
    encoder.trainable = False

    skip_1 = encoder.get_layer("block1_conv2").output
    skip_2 = encoder.get_layer("block2_conv2").output
    skip_3 = encoder.get_layer("block3_conv3").output
    skip_4 = encoder.get_layer("block4_conv3").output
    bottleneck_input = encoder.get_layer("block5_conv3").output

    bottleneck = decoder_conv_block(
        bottleneck_input, f0 * 8, dropout_bottleneck, use_batch_norm
    )

    d4 = UpSampling2D((2, 2), interpolation=interpolation)(bottleneck)
    d4 = Conv2D(f0 * 8, 2, padding="same", activation="relu")(d4)
    d4 = Concatenate()([d4, skip_4])
    d4 = decoder_conv_block(d4, f0 * 8, dropout_deep, use_batch_norm)

    d3 = UpSampling2D((2, 2), interpolation=interpolation)(d4)
    d3 = Conv2D(f0 * 4, 2, padding="same", activation="relu")(d3)
    d3 = Concatenate()([d3, skip_3])
    d3 = decoder_conv_block(d3, f0 * 4, dropout_deep, use_batch_norm)

    d2 = UpSampling2D((2, 2), interpolation=interpolation)(d3)
    d2 = Conv2D(f0 * 2, 2, padding="same", activation="relu")(d2)
    d2 = Concatenate()([d2, skip_2])
    d2 = decoder_conv_block(d2, f0 * 2, dropout_shallow, use_batch_norm)

    d1 = UpSampling2D((2, 2), interpolation=interpolation)(d2)
    d1 = Conv2D(f0, 2, padding="same", activation="relu")(d1)
    d1 = Concatenate()([d1, skip_1])
    d1 = decoder_conv_block(d1, f0, dropout_shallow, use_batch_norm)

    outputs = Conv2D(
        1, 1, activation="sigmoid", name="volcanic_ash_probability"
    )(d1)

    model = Model(inputs, outputs, name="VGG16_UNet")
    return compile_model(model, learning_rate)


def unfreeze_vgg16_block5(model):
    """Unfreeze only block5_* VGG16 layers; keep blocks 1-4 frozen."""
    for layer in model.layers:
        if layer.name.startswith("block5"):
            layer.trainable = True
        elif layer.name.startswith("block"):
            layer.trainable = False

# ============================================================
# CELL 11 - THRESHOLD CALIBRATION, AUPRC, AND SEGMENTATION METRICS
# ============================================================

def metrics_from_counts(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall_pod = tp / max(tp + fn, 1)
    f1_dice = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    far = fp / max(tp + fp, 1)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall_pod": float(recall_pod),
        "f1_dice": float(f1_dice),
        "iou": float(iou),
        "false_alarm_ratio": float(far),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


def confusion_counts(actual, predicted):
    actual = np.asarray(actual).astype(np.uint8).reshape(-1)
    predicted = np.asarray(predicted).astype(np.uint8).reshape(-1)

    tp = int(np.sum((actual == 1) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    return tp, tn, fp, fn


def collect_probability_scores(model, dataframe, batch_size=1):
    """Pool labels and model probabilities over an entire dataframe."""
    ds = create_dataset(dataframe, training=False, batch_size=batch_size)
    y_true_parts = []
    y_score_parts = []

    for images, labels in ds:
        probs = model.predict_on_batch(images)
        if hasattr(probs, "numpy"):
            probs = probs.numpy()
        if hasattr(labels, "numpy"):
            labels = labels.numpy()

        y_score_parts.append(np.asarray(probs, dtype=np.float32).reshape(-1))
        y_true_parts.append((np.asarray(labels).reshape(-1) >= 0.5).astype(np.uint8))

    if not y_true_parts:
        return np.array([], dtype=np.uint8), np.array([], dtype=np.float32)

    return np.concatenate(y_true_parts), np.concatenate(y_score_parts)


def safe_auprc(y_true, y_score):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return np.nan
    return float(average_precision_score(y_true, y_score))


def evaluate_scores_at_threshold(y_true, y_score, threshold):
    y_true = np.asarray(y_true).astype(np.uint8).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float32).reshape(-1)
    predicted = (y_score >= float(threshold)).astype(np.uint8)
    tp, tn, fp, fn = confusion_counts(y_true, predicted)
    result = metrics_from_counts(tp, tn, fp, fn)
    result["threshold"] = float(threshold)
    result["auprc"] = safe_auprc(y_true, y_score)
    return result


def calibrate_threshold(y_true, y_score, threshold_grid=THRESHOLD_GRID):
    """
    Select threshold using validation data only.

    Primary criterion: highest F1/Dice.
    Ties: choose the threshold closest to 0.50 (IoU is not used as a
    tiebreaker because binary IoU is a monotonic transform of F1/Dice).
    """
    records = []
    for threshold in threshold_grid:
        m = evaluate_scores_at_threshold(y_true, y_score, float(threshold))
        records.append({
            "threshold": float(threshold),
            "f1_dice": m["f1_dice"],
            "precision": m["precision"],
            "recall_pod": m["recall_pod"],
            "false_alarm_ratio": m["false_alarm_ratio"],
        })

    curve_df = pd.DataFrame(records)
    best_f1 = curve_df["f1_dice"].max()
    tied = curve_df[np.isclose(curve_df["f1_dice"], best_f1)].copy()
    tied["distance_from_0_5"] = np.abs(tied["threshold"] - 0.5)
    best_threshold = float(
        tied.sort_values(["distance_from_0_5", "threshold"]).iloc[0]["threshold"]
    )
    return best_threshold, curve_df


def evaluate_model_with_calibrated_threshold(model, dataframe):
    y_true, y_score = collect_probability_scores(model, dataframe, batch_size=1)
    threshold, threshold_curve = calibrate_threshold(y_true, y_score)
    metrics = evaluate_scores_at_threshold(y_true, y_score, threshold)
    return metrics, threshold_curve, y_true, y_score

# ============================================================
# CELL 12 - GENERATE REPRODUCIBLE HYPERPARAMETER CANDIDATES
# ============================================================

def generate_hyperparameter_candidates(n_trials, seed=SEED):
    """
    Equal-budget random search for the deep model.

    Batch normalization is intentionally NOT tuned because batch size = 1.
    The first candidate is a reproducible reference configuration.
    """
    rng = np.random.default_rng(seed)

    candidates = [{
        "trial_id": 1,
        "decoder_initial_filters": 32,
        "dropout_shallow": 0.1,
        "dropout_deep": 0.2,
        "dropout_bottleneck": 0.4,
        "upsampling_interpolation": "bilinear",
        "learning_rate": 1e-4,
    }]

    seen = {
        tuple((k, v) for k, v in candidates[0].items() if k != "trial_id")
    }

    while len(candidates) < n_trials:
        candidate = {
            "trial_id": len(candidates) + 1,
            "decoder_initial_filters": int(rng.choice([32, 48, 64])),
            "dropout_shallow": float(rng.choice([0.0, 0.1, 0.2, 0.3])),
            "dropout_deep": float(rng.choice([0.1, 0.2, 0.3, 0.4, 0.5])),
            "dropout_bottleneck": float(rng.choice([0.2, 0.3, 0.4, 0.5, 0.6])),
            "upsampling_interpolation": str(rng.choice(["nearest", "bilinear"])),
            "learning_rate": float(
                10 ** rng.uniform(np.log10(1e-5), np.log10(5e-4))
            ),
        }
        key = tuple((k, v) for k, v in candidate.items() if k != "trial_id")
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)

    return pd.DataFrame(candidates)


candidate_df = generate_hyperparameter_candidates(N_CV_TRIALS)
candidate_path = TUNING_DIR / "cv_hyperparameter_candidates.csv"
candidate_df.to_csv(candidate_path, index=False)

print(candidate_df.to_string(index=False))
print()
print("Saved candidates to:", candidate_path)

# ============================================================
# CELL 13 - 5-FOLD PHASE-GROUPED HYPERPARAMETER TUNING
# ============================================================

def best_epoch_from_history(history, metric_name="val_dice_coefficient"):
    values = history.history.get(metric_name, [])
    if not values:
        return len(history.epoch)
    return int(np.nanargmax(values) + 1)


cv_tuning_records = []

for _, candidate in candidate_df.iterrows():
    trial_id = int(candidate["trial_id"])
    params = candidate.to_dict()
    params.pop("trial_id", None)

    print()
    print("=" * 90)
    print(f"HYPERPARAMETER TRIAL {trial_id}/{len(candidate_df)}")
    print("=" * 90)
    print(params)

    for fold in sorted(FOLD_TO_PHASES):
        fold_train_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
        fold_val_df = train_df[train_df["fold"] == fold].reset_index(drop=True)

        print()
        print("-" * 80)
        print(f"Trial {trial_id} | Fold {fold}")
        print("Train images:", len(fold_train_df))
        print("Val images  :", len(fold_val_df))
        print("Val phases  :", FOLD_TO_PHASES[fold])

        tf.keras.backend.clear_session()
        gc.collect()

        model = build_vgg16_unet(params)
        train_ds = create_dataset(
            fold_train_df, training=True, batch_size=CV_BATCH_SIZE
        )
        val_ds = create_dataset(
            fold_val_df, training=False, batch_size=CV_BATCH_SIZE
        )

        callbacks = [
            EarlyStopping(
                monitor="val_dice_coefficient",
                mode="max",
                patience=CV_TUNING_PATIENCE,
                restore_best_weights=True,
                verbose=0,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=0,
            ),
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=CV_TUNING_MAX_EPOCHS,
            callbacks=callbacks,
            verbose=1,
        )

        y_true, y_score = collect_probability_scores(model, fold_val_df, batch_size=1)
        optimal_threshold, threshold_curve = calibrate_threshold(y_true, y_score)
        fold_metrics = evaluate_scores_at_threshold(
            y_true, y_score, optimal_threshold
        )

        threshold_curve.to_csv(
            TUNING_DIR / f"trial_{trial_id:02d}_fold_{fold}_threshold_curve.csv",
            index=False,
        )

        record = {
            "trial_id": trial_id,
            "fold": fold,
            "validation_phases": ",".join(map(str, FOLD_TO_PHASES[fold])),
            "n_train_images": len(fold_train_df),
            "n_validation_images": len(fold_val_df),
            "best_epoch": best_epoch_from_history(history),
            "optimal_threshold": optimal_threshold,
            **params,
            **fold_metrics,
        }
        cv_tuning_records.append(record)

        pd.DataFrame(cv_tuning_records).to_csv(
            TUNING_DIR / "cv_tuning_fold_results.csv",
            index=False,
        )

        del model, train_ds, val_ds, history, y_true, y_score, threshold_curve
        tf.keras.backend.clear_session()
        gc.collect()

cv_tuning_fold_df = pd.DataFrame(cv_tuning_records)

cv_trial_summary = (
    cv_tuning_fold_df.groupby("trial_id", as_index=False)
    .agg(
        mean_dice=("f1_dice", "mean"),
        std_dice=("f1_dice", "std"),
        mean_auprc=("auprc", "mean"),
        std_auprc=("auprc", "std"),
        mean_precision=("precision", "mean"),
        mean_recall_pod=("recall_pod", "mean"),
        mean_far=("false_alarm_ratio", "mean"),
        mean_accuracy=("accuracy", "mean"),
        mean_optimal_threshold=("optimal_threshold", "mean"),
        mean_best_epoch=("best_epoch", "mean"),
    )
    .sort_values(["mean_dice", "mean_auprc"], ascending=False)
    .reset_index(drop=True)
)

cv_trial_summary.to_csv(
    TUNING_DIR / "cv_tuning_trial_summary.csv",
    index=False,
)

print()
print("=" * 90)
print("CROSS-VALIDATED HYPERPARAMETER RESULTS")
print("=" * 90)
print(cv_trial_summary.round(5).to_string(index=False))

# ============================================================
# CELL 14 - SELECT AND SAVE BEST HYPERPARAMETERS
# ============================================================
# Primary selection criterion: mean validation Dice/F1.
# Secondary criterion: mean validation AUPRC (not IoU, because IoU and F1 are
# mathematically linked for binary segmentation).

best_trial_id = int(cv_trial_summary.iloc[0]["trial_id"])
best_row = candidate_df[candidate_df["trial_id"] == best_trial_id].iloc[0]

best_params = {
    "decoder_initial_filters": int(best_row["decoder_initial_filters"]),
    "dropout_shallow": float(best_row["dropout_shallow"]),
    "dropout_deep": float(best_row["dropout_deep"]),
    "dropout_bottleneck": float(best_row["dropout_bottleneck"]),
    "upsampling_interpolation": str(best_row["upsampling_interpolation"]),
    "learning_rate": float(best_row["learning_rate"]),
}

with open(
    TUNING_DIR / "best_hyperparameters.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(best_params, f, indent=4)

pd.DataFrame([{"best_trial_id": best_trial_id, **best_params}]).to_csv(
    TUNING_DIR / "best_hyperparameters.csv",
    index=False,
)

print("=" * 80)
print("BEST CROSS-VALIDATED HYPERPARAMETERS")
print("=" * 80)
print("Best trial:", best_trial_id)
for name, value in best_params.items():
    print(f"{name:30s}: {value}")
print("use_batch_norm                : False (fixed; batch size = 1)")

# ============================================================
# CELL 15 - TWO-STAGE 5-FOLD CV USING THE BEST HYPERPARAMETERS
# ============================================================
# Stage 1: VGG16 encoder frozen.
# Stage 2: only VGG16 Block 5 fine-tuned.
# Each validation fold calibrates its own threshold and reports AUPRC.
# The independent test set remains untouched.

best_hp_cv_records = []

for fold in sorted(FOLD_TO_PHASES):
    fold_train_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
    fold_val_df = train_df[train_df["fold"] == fold].reset_index(drop=True)

    print()
    print("=" * 90)
    print(f"BEST-HYPERPARAMETER TWO-STAGE CV | FOLD {fold}")
    print("=" * 90)

    tf.keras.backend.clear_session()
    gc.collect()

    model = build_vgg16_unet(best_params)
    train_ds = create_dataset(fold_train_df, training=True, batch_size=CV_BATCH_SIZE)
    val_ds = create_dataset(fold_val_df, training=False, batch_size=CV_BATCH_SIZE)

    stage1_callbacks = [
        EarlyStopping(
            monitor="val_dice_coefficient",
            mode="max",
            patience=BEST_HP_PATIENCE_STAGE1,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    stage1_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=BEST_HP_STAGE1_MAX_EPOCHS,
        callbacks=stage1_callbacks,
        verbose=1,
    )
    stage1_best_epoch = best_epoch_from_history(stage1_history)

    # ========================================================
    # V3 OOM FIX: CLEAN STAGE-1 -> STAGE-2 TRANSITION
    # ========================================================
    # Keep the Stage-1 history on disk before deleting the model.
    stage1_history_df = pd.DataFrame(stage1_history.history)
    stage1_history_df.to_csv(
        CV_DIR / f"best_hp_fold_{fold}_stage1_history.csv",
        index=False,
    )

    # Save Stage-1 weights, then fully release the Stage-1 graph,
    # optimizer state, and Python references before fine-tuning.
    stage1_temp_weights = (
        CV_DIR / f"_temp_best_hp_fold_{fold}_stage1.weights.h5"
    )
    model.save_weights(stage1_temp_weights)

    print()
    print("V3 OOM FIX: Stage-1 weights saved:", stage1_temp_weights)
    print("V3 OOM FIX: clearing TensorFlow session before Stage 2...")

    del model
    del stage1_history
    del stage1_callbacks
    gc.collect()
    tf.keras.backend.clear_session()
    gc.collect()

    # Rebuild the identical architecture with a fresh TensorFlow graph,
    # then restore the exact best Stage-1 weights.
    model = build_vgg16_unet(best_params)
    model.load_weights(stage1_temp_weights)

    # Fine-tune VGG16 Block 5 only.
    unfreeze_vgg16_block5(model)
    fine_tuning_lr = min(best_params["learning_rate"] * 0.1, 1e-5)
    compile_model(model, fine_tuning_lr)

    print("V3 OOM FIX: fresh model rebuilt and Stage-1 weights restored.")
    print("V3 OOM FIX: starting Stage-2 Block-5 fine-tuning.")
    print()

    stage2_callbacks = [
        EarlyStopping(
            monitor="val_dice_coefficient",
            mode="max",
            patience=BEST_HP_PATIENCE_STAGE2,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-8,
            verbose=1,
        ),
    ]

    stage2_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=BEST_HP_STAGE2_MAX_EPOCHS,
        callbacks=stage2_callbacks,
        verbose=1,
    )
    stage2_best_epoch = best_epoch_from_history(stage2_history)

    y_true, y_score = collect_probability_scores(model, fold_val_df, batch_size=1)
    optimal_threshold, threshold_curve = calibrate_threshold(y_true, y_score)
    fold_metrics = evaluate_scores_at_threshold(
        y_true, y_score, optimal_threshold
    )

    threshold_curve.to_csv(
        CV_DIR / f"best_hp_fold_{fold}_threshold_curve.csv",
        index=False,
    )

    record = {
        "fold": fold,
        "validation_phases": ",".join(map(str, FOLD_TO_PHASES[fold])),
        "n_train_images": len(fold_train_df),
        "n_validation_images": len(fold_val_df),
        "stage1_best_epoch": stage1_best_epoch,
        "stage2_best_epoch": stage2_best_epoch,
        "fine_tuning_learning_rate": fine_tuning_lr,
        "optimal_threshold": optimal_threshold,
        **fold_metrics,
    }
    best_hp_cv_records.append(record)

    pd.DataFrame(stage2_history.history).to_csv(
        CV_DIR / f"best_hp_fold_{fold}_stage2_history.csv", index=False
    )

    # Temporary Stage-1 weights are no longer needed after Stage 2.
    try:
        stage1_temp_weights.unlink(missing_ok=True)
    except Exception as exc:
        print("WARNING: could not remove temporary Stage-1 weights:", exc)
    pd.DataFrame(best_hp_cv_records).to_csv(
        CV_DIR / "best_hyperparameters_5fold_results.csv", index=False
    )

    del model, train_ds, val_ds, stage2_history, stage1_history_df
    del y_true, y_score, threshold_curve
    tf.keras.backend.clear_session()
    gc.collect()

best_hp_cv_df = pd.DataFrame(best_hp_cv_records)

cv_metric_columns = [
    "accuracy", "precision", "recall_pod",
    "f1_dice", "iou", "false_alarm_ratio", "auprc"
]

cv_summary_rows = []
for metric in cv_metric_columns:
    cv_summary_rows.append({
        "metric": metric,
        "mean": best_hp_cv_df[metric].mean(),
        "std": best_hp_cv_df[metric].std(ddof=1),
        "min": best_hp_cv_df[metric].min(),
        "max": best_hp_cv_df[metric].max(),
    })

best_hp_cv_summary = pd.DataFrame(cv_summary_rows)
best_hp_cv_summary.to_csv(
    CV_DIR / "best_hyperparameters_5fold_summary.csv", index=False
)

print()
print("=" * 90)
print("BEST MODEL - 5-FOLD INTERNAL CV RESULTS")
print("=" * 90)
print(best_hp_cv_df[[
    "fold", "validation_phases", "n_validation_images",
    "optimal_threshold", "accuracy", "precision", "recall_pod",
    "f1_dice", "iou", "false_alarm_ratio", "auprc"
]].round(5).to_string(index=False))

print()
print("MEAN +/- SD")
for _, r in best_hp_cv_summary.iterrows():
    print(f"{r['metric']:20s}: {r['mean']:.5f} +/- {r['std']:.5f}")

# ============================================================
# CELL 16 - CHOOSE FINAL EPOCH COUNTS AND FINAL THRESHOLD FROM CV
# ============================================================
# The independent test set is NOT used to choose epochs or threshold.
# Median across the five phase-grouped validation folds is used for robustness.

FINAL_STAGE1_EPOCHS = max(
    1, int(round(best_hp_cv_df["stage1_best_epoch"].median()))
)
FINAL_STAGE2_EPOCHS = max(
    1, int(round(best_hp_cv_df["stage2_best_epoch"].median()))
)
FINAL_PREDICTION_THRESHOLD = float(
    best_hp_cv_df["optimal_threshold"].median()
)

epoch_threshold_selection_df = pd.DataFrame([{
    "final_stage1_epochs": FINAL_STAGE1_EPOCHS,
    "final_stage2_epochs": FINAL_STAGE2_EPOCHS,
    "final_prediction_threshold": FINAL_PREDICTION_THRESHOLD,
    "epoch_method": "median best epoch across 5 CV folds",
    "threshold_method": "median validation-optimal F1/Dice threshold across 5 CV folds",
}])
epoch_threshold_selection_df.to_csv(
    CV_DIR / "final_epoch_and_threshold_selection.csv",
    index=False,
)

with open(
    CV_DIR / "final_prediction_threshold.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        {"final_prediction_threshold": FINAL_PREDICTION_THRESHOLD},
        f,
        indent=4,
    )

print("Final Stage-1 epochs       :", FINAL_STAGE1_EPOCHS)
print("Final Stage-2 epochs       :", FINAL_STAGE2_EPOCHS)
print("Final prediction threshold :", FINAL_PREDICTION_THRESHOLD)

# ============================================================
# CELL 17 - FINAL TRAINING ON ALL PHASE 1-13 IMAGES
# ============================================================
#
# No validation split is used here, because the epoch counts and
# hyperparameters, threshold, and epoch counts were already selected using cross-validation.
# Phases 14-16 remain completely locked.

tf.keras.backend.clear_session()
gc.collect()

final_model = build_vgg16_unet(best_params)
all_train_ds = create_dataset(
    train_df,
    training=True,
    batch_size=FINAL_BATCH_SIZE,
)

stage1_log_path = LOG_DIR / "final_stage1_all_training.csv"
stage1_history = final_model.fit(
    all_train_ds,
    epochs=FINAL_STAGE1_EPOCHS,
    callbacks=[CSVLogger(str(stage1_log_path))],
    verbose=1,
)

# ============================================================
# V3 OOM FIX: CLEAN FINAL STAGE-1 -> STAGE-2 TRANSITION
# ============================================================
final_stage1_temp_weights = (
    MODEL_DIR / "_temp_final_stage1.weights.h5"
)
final_model.save_weights(final_stage1_temp_weights)

print()
print("V3 OOM FIX: final Stage-1 weights saved:", final_stage1_temp_weights)
print("V3 OOM FIX: clearing TensorFlow session before final Stage 2...")

del final_model
del stage1_history
gc.collect()
tf.keras.backend.clear_session()
gc.collect()

# Rebuild exactly the same model, restore Stage-1 weights,
# then unfreeze VGG16 Block 5 only.
final_model = build_vgg16_unet(best_params)
final_model.load_weights(final_stage1_temp_weights)

unfreeze_vgg16_block5(final_model)
FINAL_FINE_TUNING_LR = min(best_params["learning_rate"] * 0.1, 1e-5)
compile_model(final_model, FINAL_FINE_TUNING_LR)

print("V3 OOM FIX: final model rebuilt and Stage-1 weights restored.")
print("V3 OOM FIX: starting final Stage-2 Block-5 fine-tuning.")
print()

stage2_log_path = LOG_DIR / "final_stage2_all_training.csv"
stage2_history = final_model.fit(
    all_train_ds,
    epochs=FINAL_STAGE2_EPOCHS,
    callbacks=[CSVLogger(str(stage2_log_path))],
    verbose=1,
)

FINAL_MODEL_PATH = MODEL_DIR / "best_cv_tuned_vgg16_unet_all_train.keras"
final_model.save(FINAL_MODEL_PATH)

try:
    final_stage1_temp_weights.unlink(missing_ok=True)
except Exception as exc:
    print("WARNING: could not remove temporary final Stage-1 weights:", exc)

print("Final model saved to:", FINAL_MODEL_PATH)

# ============================================================
# CELL 18 - FINAL TEST PREDICTIONS + PER-IMAGE METRICS
# UPDATED VERSION
# ============================================================

def create_red_blue_mask(binary_mask):
    """
    Create visualization mask:
      Blue = Non-ash
      Red  = Ash
    """
    color = np.zeros(
        (*binary_mask.shape, 3),
        dtype=np.uint8
    )

    color[binary_mask == 0] = [0, 0, 255]      # Non-ash
    color[binary_mask == 1] = [255, 0, 0]      # Ash

    return color


def create_error_map(actual, predicted):
    """
    Error-map classes:

      TN = black
      TP = green
      FP = red
      FN = yellow
    """

    error_map = np.zeros(
        (*actual.shape, 3),
        dtype=np.uint8
    )

    tn = (
        (actual == 0) &
        (predicted == 0)
    )

    tp = (
        (actual == 1) &
        (predicted == 1)
    )

    fp = (
        (actual == 0) &
        (predicted == 1)
    )

    fn = (
        (actual == 1) &
        (predicted == 0)
    )

    error_map[tn] = [0, 0, 0]          # Black
    error_map[tp] = [0, 180, 0]        # Green
    error_map[fp] = [255, 0, 0]        # Red
    error_map[fn] = [255, 255, 0]      # Yellow

    return error_map


# ============================================================
# HELPER FUNCTION
# Convert TensorFlow Tensor OR NumPy array safely
# ============================================================

def to_numpy(x):

    if hasattr(x, "numpy"):
        return x.numpy()

    return np.asarray(x)


# ============================================================
# CREATE TEST DATASET
# ============================================================

test_dataset = create_dataset(
    test_df,
    training=False,
    batch_size=1,
)


# ============================================================
# STORAGE FOR PER-IMAGE RESULTS
# ============================================================

prediction_records = []

# Store probability scores for pooled AUPRC calculations.
test_y_true_parts = []
test_y_score_parts = []
phase_probability_store = defaultdict(lambda: {"y_true": [], "y_score": []})


print("=" * 100)
print("FINAL INDEPENDENT TEST PREDICTIONS")
print("=" * 100)

print(
    f"Number of test images: {len(test_df)}"
)

print(
    f"Prediction threshold : {FINAL_PREDICTION_THRESHOLD}"
)

print("=" * 100)


# ============================================================
# LOOP THROUGH EACH TEST IMAGE
# ============================================================

for image_index, (row, batch) in enumerate(
    zip(
        test_df.itertuples(index=False),
        test_dataset
    ),
    start=1
):

    images, labels = batch


    # ========================================================
    # 1. MODEL PREDICTION
    # ========================================================

    probability = final_model.predict_on_batch(
        images
    )

    # TensorFlow Tensor OR NumPy array
    probability = to_numpy(
        probability
    )

    # Remove:
    # batch dimension + channel dimension
    #
    # (1, H, W, 1)
    # ->
    # (H, W)
    probability = probability[
        0, :, :, 0
    ]


    # ========================================================
    # 2. GROUND TRUTH
    # ========================================================

    actual = to_numpy(
        labels
    )

    actual = actual[
        0, :, :, 0
    ]


    # Convert ground truth safely to binary
    actual = (
        actual >= 0.5
    ).astype(
        np.uint8
    )


    # ========================================================
    # 3. BINARY MODEL PREDICTION
    # ========================================================

    predicted = (
        probability
        >= FINAL_PREDICTION_THRESHOLD
    ).astype(
        np.uint8
    )


    # ========================================================
    # 4. CONFUSION MATRIX COUNTS
    # ========================================================

    tp, tn, fp, fn = confusion_counts(
        actual,
        predicted
    )


    # ========================================================
    # 5. METRICS
    # ========================================================

    metrics = metrics_from_counts(
        tp,
        tn,
        fp,
        fn
    )

    image_auprc = safe_auprc(
        actual.reshape(-1),
        probability.reshape(-1),
    )

    test_y_true_parts.append(actual.reshape(-1).astype(np.uint8))
    test_y_score_parts.append(probability.reshape(-1).astype(np.float32))
    phase_probability_store[int(row.phase)]["y_true"].append(
        actual.reshape(-1).astype(np.uint8)
    )
    phase_probability_store[int(row.phase)]["y_score"].append(
        probability.reshape(-1).astype(np.float32)
    )


    # ========================================================
    # 6. ASH PERCENTAGE
    # ========================================================

    total_pixels = max(
        tp + tn + fp + fn,
        1
    )


    ground_truth_ash_percent = (
        (tp + fn)
        /
        total_pixels
        *
        100
    )


    predicted_ash_percent = (
        (tp + fp)
        /
        total_pixels
        *
        100
    )


    # ========================================================
    # 7. SAVE METRICS RECORD
    # ========================================================

    prediction_records.append({

        "timestamp":
            row.timestamp,

        "phase":
            int(row.phase),

        "phase_name":
            row.phase_name,

        "ground_truth_ash_percent":
            ground_truth_ash_percent,

        "predicted_ash_percent":
            predicted_ash_percent,

        "auprc":
            image_auprc,

        **metrics,
    })


    # ========================================================
    # PRINT PROGRESS
    # ========================================================

    print(
        f"[{image_index:02d}/{len(test_df)}] "
        f"{row.timestamp} | "
        f"Phase {int(row.phase)} | "
        f"IoU={metrics['iou']:.4f} | "
        f"Dice={metrics['f1_dice']:.4f} | "
        f"POD={metrics['recall_pod']:.4f} | "
        f"FAR={metrics['false_alarm_ratio']:.4f} | "
        f"AUPRC={image_auprc:.4f}"
    )


    # ========================================================
    # 8. SAVE BINARY PREDICTION
    # ========================================================

    binary_prediction_path = (

        PREDICTION_DIR
        /
        (
            f"{row.timestamp}_"
            f"phase{int(row.phase)}_"
            f"prediction_binary.png"
        )
    )


    Image.fromarray(
        predicted * 255
    ).save(
        binary_prediction_path
    )


    # ========================================================
    # 9. CREATE COLOR MASKS
    # ========================================================

    prediction_color = create_red_blue_mask(
        predicted
    )

    actual_color = create_red_blue_mask(
        actual
    )


    # ========================================================
    # 10. SAVE COLOR PREDICTION
    # ========================================================

    color_prediction_path = (

        PREDICTION_DIR
        /
        (
            f"{row.timestamp}_"
            f"phase{int(row.phase)}_"
            f"prediction_color.png"
        )
    )


    Image.fromarray(
        prediction_color
    ).save(
        color_prediction_path
    )


    # ========================================================
    # 11. CREATE ERROR MAP
    # ========================================================

    error_map = create_error_map(
        actual,
        predicted
    )


    # ========================================================
    # 12. SAVE STANDALONE ERROR MAP
    # ========================================================

    error_map_path = (

        ERROR_MAP_DIR
        /
        (
            f"{row.timestamp}_"
            f"phase{int(row.phase)}_"
            f"error_map.png"
        )
    )


    Image.fromarray(
        error_map
    ).save(
        error_map_path
    )


    # ========================================================
    # 13. READ ORIGINAL ASH RGB IMAGE
    # ========================================================

    raw_image = read_tiff_rgb(
        row.image_path
    )


    raw_image = np.asarray(

        Image.fromarray(

            np.clip(
                raw_image,
                0,
                255
            ).astype(
                np.uint8
            )

        ).resize(

            (
                TARGET_WIDTH,
                TARGET_HEIGHT
            ),

            resample=
                Image.Resampling.BILINEAR,
        )
    )


    # ========================================================
    # 14. CREATE 4-PANEL FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )


    # --------------------------------------------------------
    # PANEL 1: ORIGINAL HIMAWARI ASH RGB
    # --------------------------------------------------------

    axes[0].imshow(
        raw_image
    )

    axes[0].set_title(
        f"Himawari-9 Ash RGB\n"
        f"{row.timestamp} | "
        f"Phase {int(row.phase)}"
    )

    axes[0].axis(
        "off"
    )


    # --------------------------------------------------------
    # PANEL 2: GROUND TRUTH
    # --------------------------------------------------------

    axes[1].imshow(
        actual_color
    )

    axes[1].set_title(
        "Ground Truth\n"
        "Red = Ash, Blue = Non-ash"
    )

    axes[1].axis(
        "off"
    )


    # --------------------------------------------------------
    # PANEL 3: MODEL PREDICTION
    # --------------------------------------------------------

    axes[2].imshow(
        prediction_color
    )

    axes[2].set_title(

        "VGG16 + U-Net Prediction\n"

        f"IoU={metrics['iou']:.3f}, "

        f"Dice={metrics['f1_dice']:.3f}"
    )

    axes[2].axis(
        "off"
    )


    # --------------------------------------------------------
    # PANEL 4: ERROR MAP
    # --------------------------------------------------------

    axes[3].imshow(
        error_map
    )

    axes[3].set_title(
        "Error Map"
    )

    axes[3].axis(
        "off"
    )


    # ========================================================
    # ERROR MAP LEGEND
    # ========================================================

    axes[3].legend(

        handles=[

            Patch(
                facecolor="black",
                label="TN"
            ),

            Patch(
                facecolor="green",
                label="TP"
            ),

            Patch(
                facecolor="red",
                label="FP"
            ),

            Patch(
                facecolor="yellow",
                label="FN"
            ),

        ],

        loc="lower center",

        bbox_to_anchor=(
            0.5,
            -0.18
        ),

        ncol=4,

        frameon=False,
    )


    # ========================================================
    # 15. SAVE COMPARISON FIGURE
    # ========================================================

    plt.tight_layout()


    comparison_path = (

        ERROR_MAP_DIR
        /
        (
            f"{row.timestamp}_"
            f"phase{int(row.phase)}_"
            f"comparison_error_map.png"
        )
    )


    plt.savefig(

        comparison_path,

        dpi=250,

        bbox_inches="tight",
    )


    plt.close(
        fig
    )


# ============================================================
# POOLED TEST LABELS AND PROBABILITIES FOR AUPRC
# ============================================================

test_y_true_all = np.concatenate(test_y_true_parts)
test_y_score_all = np.concatenate(test_y_score_parts)

# ============================================================
# 16. CREATE PER-IMAGE METRICS DATAFRAME
# ============================================================

test_per_image_df = pd.DataFrame(
    prediction_records
)


# ============================================================
# 17. SAVE PER-IMAGE METRICS
# ============================================================

test_metrics_path = (

    OUTPUT_DIR
    /
    "test_per_image_metrics.csv"
)


test_per_image_df.to_csv(

    test_metrics_path,

    index=False,
)


# ============================================================
# 18. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 100)
print("TEST PER-IMAGE RESULTS")
print("=" * 100)


print(

    test_per_image_df
    .round(5)
    .to_string(
        index=False
    )
)


print("\n")
print("=" * 100)

print(
    "Per-image metrics saved to:"
)

print(
    test_metrics_path
)

print(
    "\nPrediction images saved to:"
)

print(
    PREDICTION_DIR
)

print(
    "\nError maps saved to:"
)

print(
    ERROR_MAP_DIR
)

print("=" * 100)

# ============================================================
# CELL 19 - TEST METRICS BY PHASE 14, 15, AND 16
# ============================================================
# Confusion-matrix metrics are pooled across pixels in each phase.
# AUPRC is calculated from pooled probability scores, not from thresholded masks.

phase_test_records = []

for phase in [14, 15, 16]:
    phase_df = test_per_image_df[test_per_image_df["phase"] == phase].copy()

    if phase_df.empty:
        print(f"WARNING: No test images found for Phase {phase}")
        continue

    tp = int(phase_df["true_positive"].sum())
    tn = int(phase_df["true_negative"].sum())
    fp = int(phase_df["false_positive"].sum())
    fn = int(phase_df["false_negative"].sum())
    pooled = metrics_from_counts(tp, tn, fp, fn)

    phase_y_true = np.concatenate(phase_probability_store[phase]["y_true"])
    phase_y_score = np.concatenate(phase_probability_store[phase]["y_score"])
    phase_auprc = safe_auprc(phase_y_true, phase_y_score)

    phase_test_records.append({
        "phase": phase,
        "phase_name": f"Phase {phase}",
        "number_of_images": len(phase_df),
        "prediction_threshold": FINAL_PREDICTION_THRESHOLD,
        "ground_truth_ash_percent": (tp + fn) / max(tp + tn + fp + fn, 1) * 100,
        "predicted_ash_percent": (tp + fp) / max(tp + tn + fp + fn, 1) * 100,
        "mean_image_dice": phase_df["f1_dice"].mean(),
        "std_image_dice": phase_df["f1_dice"].std(ddof=1),
        "mean_image_auprc": phase_df["auprc"].mean(),
        "pooled_auprc": phase_auprc,
        **pooled,
    })


test_by_phase_df = pd.DataFrame(phase_test_records)
test_by_phase_df.to_csv(
    OUTPUT_DIR / "test_metrics_by_phase_14_15_16.csv",
    index=False,
)

print()
print("=" * 100)
print("INDEPENDENT TEST RESULTS BY ERUPTION PHASE")
print("=" * 100)
print(test_by_phase_df[[
    "phase_name", "number_of_images", "prediction_threshold",
    "accuracy", "precision", "recall_pod", "f1_dice", "iou",
    "false_alarm_ratio", "pooled_auprc",
    "ground_truth_ash_percent", "predicted_ash_percent"
]].round(5).to_string(index=False))

# ============================================================
# CELL 20 - OVERALL INDEPENDENT TEST METRICS + AUPRC
# ============================================================

tp = int(test_per_image_df["true_positive"].sum())
tn = int(test_per_image_df["true_negative"].sum())
fp = int(test_per_image_df["false_positive"].sum())
fn = int(test_per_image_df["false_negative"].sum())

overall_test_metrics = metrics_from_counts(tp, tn, fp, fn)
overall_test_auprc = safe_auprc(test_y_true_all, test_y_score_all)

overall_test_df = pd.DataFrame([{
    "number_of_images": len(test_per_image_df),
    "phases": "14,15,16",
    "prediction_threshold": FINAL_PREDICTION_THRESHOLD,
    "auprc": overall_test_auprc,
    "ground_truth_ash_percent": (tp + fn) / max(tp + tn + fp + fn, 1) * 100,
    "predicted_ash_percent": (tp + fp) / max(tp + tn + fp + fn, 1) * 100,
    **overall_test_metrics,
}])

overall_test_df.to_csv(
    OUTPUT_DIR / "test_metrics_overall.csv",
    index=False,
)

# Save the full precision-recall curve for the locked independent test set.
pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
    test_y_true_all, test_y_score_all
)
pr_curve_df = pd.DataFrame({
    "precision": pr_precision,
    "recall": pr_recall,
    "threshold": np.append(pr_thresholds, np.nan),
})
pr_curve_df.to_csv(
    OUTPUT_DIR / "test_precision_recall_curve.csv",
    index=False,
)

print()
print("=" * 100)
print("OVERALL LOCKED TEST RESULT")
print("=" * 100)
print(overall_test_df.round(5).to_string(index=False))
print()
print("Precision-recall curve saved to:")
print(OUTPUT_DIR / "test_precision_recall_curve.csv")

# ============================================================
# CELL 21 - SAVE A COMPACT CV VS TEST SUMMARY
# ============================================================

cv_summary_wide = {
    f"cv_{row['metric']}_mean": row["mean"]
    for _, row in best_hp_cv_summary.iterrows()
}
cv_summary_wide.update({
    f"cv_{row['metric']}_std": row["std"]
    for _, row in best_hp_cv_summary.iterrows()
})

final_summary = pd.DataFrame([{
    "best_trial_id": best_trial_id,
    "tuning_budget_candidates": N_CV_TRIALS,
    "batch_normalization": False,
    "final_stage1_epochs": FINAL_STAGE1_EPOCHS,
    "final_stage2_epochs": FINAL_STAGE2_EPOCHS,
    "final_prediction_threshold": FINAL_PREDICTION_THRESHOLD,
    **cv_summary_wide,
    "test_auprc": overall_test_auprc,
    **{f"test_{k}": v for k, v in overall_test_metrics.items()
       if k not in {"true_positive", "true_negative", "false_positive", "false_negative"}},
}])

final_summary.to_csv(
    OUTPUT_DIR / "cv_and_independent_test_summary.csv",
    index=False,
)

print()
print("All outputs saved under:")
print(OUTPUT_DIR)
print()
print("Important files:")
print(" - hyperparameter_tuning/cv_hyperparameter_candidates.csv")
print(" - hyperparameter_tuning/cv_tuning_trial_summary.csv")
print(" - hyperparameter_tuning/best_hyperparameters.csv")
print(" - cross_validation/best_hyperparameters_5fold_results.csv")
print(" - cross_validation/best_hyperparameters_5fold_summary.csv")
print(" - cross_validation/final_epoch_and_threshold_selection.csv")
print(" - test_per_image_metrics.csv")
print(" - test_metrics_by_phase_14_15_16.csv")
print(" - test_metrics_overall.csv")
print(" - test_precision_recall_curve.csv")
print(" - test_error_maps/*.png")
