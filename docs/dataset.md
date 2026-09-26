# Dataset: NEU Surface Defects — Classification Split

## Provenance and license

| Field | Value |
|---|---|
| Original dataset | NEU Surface Defects Database (NEU-DET), Northeastern University (Song & Yan, 2013) |
| Original channel | IEEE DataPort (account required; no explicit redistribution license found) |
| Mirror used by this project | [Vania43/neu_det_caption](https://huggingface.co/datasets/Vania43/neu_det_caption) on Hugging Face |
| Mirror license | Not explicitly stated on the mirror page. The upstream NEU database is published for academic/research use. This project uses the data **for educational, non-commercial portfolio purposes** and does not redistribute it in this repository. |
| Original paper | Song, K., & Yan, Y. (2013). "A noise robust method based on completed behaviors of primitive model for surface defect detection." *Applied Surface Science*. |

**What we verified ourselves (2026-09-02):** the mirror hosts a single
1,440-row parquet file containing 200×200 grayscale images, filenames of the
form `<class>_<n>.jpg`, and string labels for exactly 6 classes (240 images
each — the classification split of NEU-DET, which also has a 300-image/class
detection variant). No other dataset statistics are claimed from the mirror.

If you require the official distribution (including bounding boxes for
detection), obtain it from IEEE DataPort or the authors directly.

## Classes (6, balanced: 240 images each)

| Label | Description |
|---|---|
| crazing | Network of fine surface cracks from uneven cooling / residual stress |
| inclusion | Foreign material (non-metallic particles) rolled into the strip |
| patches | Localized areas of irregular texture/color from uneven processing |
| pitted_surface | Dense small pits from localized corrosion |
| rolled-in_scale | Oxide scale rolled into the surface as dark streaks/particles |
| scratches | Mechanical abrasion marks |

## Splits (deterministic, class-stratified, seed 42)

| Split | Images | Per class |
|---|---|---|
| train | 1080 | 180 |
| val | 216 | 36 |
| test | 144 | 24 |

Splitting is done in `ml/data/prepare.py`: images are grouped by class,
ordered by content hash (filesystem-independent), shuffled with a seeded RNG,
then sliced 10% test / 15% val / 75% train. The same seed reproduces the same
split on any machine.

## Preprocessing

1. Download parquet from the mirror (recorded in the dataset manifest).
2. Decode each image, convert to grayscale `L`, re-encode as JPEG
   (quality 95) under `data/processed/<split>/<class>/`.
3. Training-time transforms (`ml/data/dataset.py`):
   - resize to 224×224 (antialias)
   - train only: horizontal/vertical flips (p=0.5), rotation ±15°, translate ±5%
   - normalize with dataset statistics below

Photometric augmentation (brightness/contrast/color) is intentionally
avoided: defects are grayscale texture phenomena and photometric distortion
can destroy the defect signature or create fake "inclusion" evidence.

## Normalization statistics

Computed 2026-09-02 by a direct two-pass over all 1080 training images in
[0,1] space (reproduce with `ml/preprocessing/stats.py` and the training
loader without augmentation):

- mean = **0.5088**
- std = **0.2095**

These constants live in `ml/data/dataset.py` (MEAN/STD).

## Known limitations

- **Every image contains a defect.** NEU-DET's classification split contains
  no defect-free surface images, so this platform cannot detect "no defect"
  (OK/NG) surfaces — it classifies *which* defect a defective surface shows.
  The UI states this clearly; a production system would need an additional
  OK-class dataset (e.g., DAGM texture patches or factory OK images).
- 200×200 crops from a single hot-rolling line; lighting/texture may not
  transfer to other mills without domain adaptation.
- Class balance (240/240/…) is convenient but unrealistic; real defect
  distributions are heavily imbalanced, so reported metrics are optimistic
  relative to a production prior.
- The mirror lacks bounding boxes; the detection variant of NEU-DET (with
  annotations) is not used — see `docs/explainability.md` for why Grad-CAM
  localization is presented as *evidence*, not detection.
- Model confidence is a softmax output, not a calibrated probability. Do not
  interpret 0.9 as "90% chance of being right" without calibration analysis.
