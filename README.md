# 🛰️ Satellite Image-Based Poverty Prediction

An end-to-end machine learning project that predicts an area's poverty level
(LOW / MEDIUM / HIGH) from a satellite-style image, based on visible
buildings, roads, and vegetation.

## ⚠️ About the data (read this first)

Real satellite imagery (Sentinel-2, Maxar) and real poverty ground-truth
labels (DHS surveys, World Bank LSMS) require external registration and
multi-GB downloads. To give you a **complete, runnable project today**,
`data_generator.py` procedurally generates synthetic satellite-style images
whose building density, road count, and vegetation cover are controlled to
match each poverty class — the same visual signals real researchers use.

**This means:** the pipeline, model, and app are 100% real and functional.
Only the training images are synthetic. See "Using real data" below for how
to swap in the real thing without changing any other code.

## Project structure

```
poverty-prediction/
├── data_generator.py   # generates the synthetic image dataset
├── features.py         # extracts numeric features from an image
├── train.py             # trains the classifier + saves plots
├── predict.py           # CLI: predict a single image
├── app.py                # Streamlit web app (upload & predict)
├── requirements.txt
├── data/                 # generated images (train/test x low/medium/high)
└── models/               # trained model + evaluation plots
```

## Setup (run once)

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt
```

## How to run — step by step

### Step 1 — Generate the dataset
```bash
python data_generator.py
```
Creates 600 training images + 120 test images under `data/`.

### Step 2 — Train the model
```bash
python train.py
```
Prints accuracy + a classification report, and saves:
- `models/poverty_model.joblib` (the trained model)
- `models/confusion_matrix.png`
- `models/feature_importance.png`

### Step 3 — See it work! Two ways to check output:

**Option A — Command line (fastest way to see a prediction):**
```bash
python predict.py data/test/high/high_0000.png
python predict.py data/test/low/low_0000.png
```

**Option B — Web app (recommended, visual):**
```bash
streamlit run app.py
```
This opens a browser tab at `http://localhost:8501` where you can:
- Upload your own image and get a prediction, or
- Click "Generate random area" to instantly test the model on a new
  synthetic example
- See the confidence bar chart and the extracted features table

## How it works

1. **`data_generator.py`** draws images with controlled building density,
   road count, and vegetation — mimicking developed vs. underdeveloped areas.
2. **`features.py`** extracts 6 numeric signals from any image:
   vegetation ratio, built-up ratio, edge density (roads/building outlines),
   color texture variance, and brightness stats.
3. **`train.py`** trains a Random Forest classifier on these features against
   the known class labels, and evaluates it on a held-out test set.
4. **`app.py` / `predict.py`** load the trained model and run it on new images.

## Using real data instead of synthetic data

1. Get satellite tiles: [Sentinel-2 via Copernicus](https://dataspace.copernicus.eu/)
   (free, 10m resolution) or a paid provider (Maxar/Planet) for higher detail.
2. Get poverty ground truth: [DHS Program surveys](https://dhsprogram.com/)
   (has GPS-tagged wealth index scores) or World Bank LSMS data.
3. For each survey GPS point, crop a satellite tile centered on it, bucket
   its wealth index into low/medium/high, and save it into
   `data/train/<class>/` and `data/test/<class>/` — same folder structure
   this project already uses.
4. Re-run `python train.py`. No other file needs to change.
5. For higher accuracy on real imagery, swap the Random Forest in `train.py`
   for a CNN (e.g., a pretrained ResNet fine-tuned with PyTorch/TensorFlow) —
   `features.py` would be replaced by the CNN's own learned features.

## Notes

- Because the synthetic classes are cleanly separated by design, you'll see
  very high (often ~100%) test accuracy. This is expected for synthetic
  data — real satellite imagery is noisier and typically yields
  60-85% accuracy in published research, depending on region and resolution.
- To regenerate a fresh/larger dataset, edit `n_per_class_train` /
  `n_per_class_test` in `data_generator.py`'s `build_dataset()` call.
