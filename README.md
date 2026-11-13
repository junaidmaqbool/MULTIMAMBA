# MultiHb — multimodal hemoglobin estimation from eye images and bedside anthropometry

Code for the paper *MultiHb: An AI-Enabled Multimodal Fusion Network for Non-Invasive
Hemoglobin Estimation on the HemoWLab Platform*.

MultiHb converts an image-only hemoglobin (Hb) regressor into a multimodal one. A single
wrapper, `MultiModalWrapper`, reads the feature width off a backbone's terminal linear
head, registers a forward pre-hook that captures the pooled image feature, and fuses that
feature with an encoding of three bedside clinical variables before a fresh regression
head produces the Hb value. No backbone file is modified.

---

## What this repository reproduces

| | |
|---|---|
| Task | Continuous Hb regression, in g/dL |
| Image input | One left palpebral conjunctiva photograph per subject |
| Clinical inputs | Height, weight, body-mass index (BMI) |
| Backbones | 14 total: 9 Mamba variants, 5 ImageNet-pretrained CNN/ViT from `timm` |
| Fusion operators | Concatenation, FiLM, channel gating, token-level cross-attention |
| Protocol | Locked 10% test set, stratified 5-fold CV, per-fold tabular standardisation, Hb-bin-balanced sampling |
| Proposed model | ViT-B/16 + cross-attention over image, height, weight, BMI |

The regression head is the head trained and reported. A classification head exists in the
code so users can run the categorical target, but no classification result appears in the
paper.

Every split is at the **subject level**: each subject contributes exactly one left-eye
image, so a row is a subject and splitting rows is splitting subjects. No child appears in
more than one of the training, validation and test sets.

---

## Configuration

Everything is set in the first cell of `MultiHb.ipynb`. These values reproduce the
numbers in the paper.

```python
IMAGE_DIR   = "<path>/left_eye"       # one image per subject
CSV_PATH    = "<path>/merge_excel_1.xlsx"
IMAGE_DIR_2 = ""                      # second image slot unused
CSV_PATH_2  = ""

TASK        = "regression"
USE_TABULAR = True
NUMERIC_FEATURES = ["Height", "Weight", "BMI"]   # BMI derived from height and weight

HB_FILTER_MIN, HB_FILTER_MAX = 7.0, 15.0         # inclusion filter
REG_CONFIG = {"normalize_hb": True,
              "hb_min": 7.0, "hb_max": 15.0,     # target scaling
              "loss_fn": "huber", "huber_delta": 1.0}

K           = 5        # stratified folds on the train pool
TEST_SPLIT  = 0.10     # locked test set
AUG_BINS    = 10       # Hb-bin-balanced sampler
SEED        = 42
FUSION      = "attn"   # concat | film | gate | attn
EPOCHS      = 40
BATCH_SIZE  = 32
LR          = 1e-4
IMG_SIZE    = 224
FREEZE_BACKBONE = True
```

Other fixed settings: AdamW, mixed precision, 3-epoch linear warmup then cosine
annealing, early stopping with patience 8, tabular MLP width 64 with dropout 0.2,
attention tokens of dimension 128 over 4 heads, Huber delta 1.0.

A single Hb cut at 12.0 g/dL is used **only** to stratify the splits and to describe the
target distribution. It is not a clinical definition of anemia for this cohort, it is
never a model output, and it enters neither the loss nor any reported metric.

---

## Dataset

`ImageHb`, collected on the HemoWLab handheld anemometer from 4,430 pediatric subjects
aged 1 to 201 months. The full collection holds 27,259 images across four pallor sites
(eye conjunctiva, palm, tongue, nailbed) with anthropometry and a paired venous complete
blood count per subject. This study uses the 3,320 left-eye images that remain after
pairing with a numeric Hb value and applying the inclusion filter.

The dataset is shared with researchers through <https://bdtd.github.io/ImageHB> under the
data-sharing terms of the ethics approval (Institutional Ethics Committee, PGIMER
Chandigarh, letter no. PGI/IEC/2022/EIC000190, 18 February 2022). Trained model weights
are available on the same terms.

**The raw clinical spreadsheet contains direct identifiers and is not distributed here.**
Only the de-identified release through the portal above should be used.

---

## Three-stage study

1. **Backbones** — compare all 14 feature extractors.
2. **Fusion** — fix the winning backbone, sweep the 4 fusion operators.
3. **Clinical variables** — fix backbone and fusion, add height, weight and BMI singly and
   together.

Each stage reports the test metric as a five-fold mean ± standard deviation.

---

## Repository layout

```
MultiHb.ipynb             reference implementation of the paper (run this one)
multimamba.ipynb          original exploratory notebook, kept for provenance
run_all_mamba.ipynb       batch runs over the backbone registry
run_ensemble.ipynb        ensembling experiments (not reported in the paper)
new_mamba_model.ipynb     custom backbone development
inspect_dataset.py        dataset coverage and column validation
smoke_test_all_models.py  instantiates every registry backbone
MAMBA_MODELS/             backbone implementations and shared utilities
requirements_kaggle.txt   dependencies
```

Outputs written by `MultiHb.ipynb`:

```
multihb_stage1_backbones.csv   Stage 1, fourteen backbones
multihb_stage2_fusion.csv      Stage 2, fusion sweep
multihb_stage3_variables.csv   Stage 3, per-variable ablation
multihb_all_results.csv        the three stages combined
multihb_stage*.png             the corresponding bar charts
```

---

## Install and run

```bash
pip install mamba-ssm causal-conv1d einops timm pandas openpyxl scikit-learn tqdm matplotlib
```

Open `MultiHb.ipynb`, set `IMAGE_DIR` and `CSV_PATH` in the first cell, and run top to
bottom. Stages 1 to 3 run in order and each writes its own CSV and chart.

---

## Citing

Citation details will be added once the article is published.

## License

See `LICENSE`.
