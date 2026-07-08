# Credit Card Fraud Detection — Reproduction & Critical Evaluation

Final project for *Data Science in Cyber* (Dr. Uri Itai). Reproduces and critically
evaluates a published autoencoder-based credit card fraud detection tutorial, adds two
supervised baselines, and quantifies a test-set threshold-selection leakage issue in the
original source's evaluation methodology.

## Source being evaluated

- Blog post: [Credit Card Fraud Detection using Autoencoders in Keras — TensorFlow for Hackers (Part VII)](https://curiousily.com/posts/credit-card-fraud-detection-using-autoencoders-in-keras/) by Venelin Valkov
- Original GitHub repository: [curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras](https://github.com/curiousily/Credit-Card-Fraud-Detection-using-Autoencoders-in-Keras)

## Dataset

Kaggle *Credit Card Fraud Detection* dataset (`mlg-ulb/creditcardfraud`): 284,807 European
cardholder transactions from September 2013, 492 of which are fraudulent (0.172%).
Features `V1`-`V28` are PCA components (anonymized for privacy); `Time` and `Amount` are
the only untransformed features.

The raw CSV is not committed to this repository (it is a ~100 MB third-party dataset).
Run `python data/download_data.py` to fetch it — this pulls the identical file from a
public GitHub mirror (avoids requiring a Kaggle account/API key):
`https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv`

## Repository contents

```
fraud-detection-cyber-project/
  data/download_data.py          # fetches creditcard.csv
  src/data_utils.py               # data loading / inspection / split / preprocessing helpers
  src/models.py                   # shared, leakage-free model pipeline factories (LR, RF)
  src/eval_utils.py               # metric computation / plotting helpers
  notebooks/fraud_detection_analysis.ipynb   # full analysis (EDA, feature engineering,
                                              # 3 models, evaluation, error analysis,
                                              # cross-validation, chronological split)
  run_experiment.py                # standalone script: runs the core experiment outside Jupyter
  report/report.pdf               # full written report (8 required sections)
  report/figures/                 # figures extracted from the executed notebook
  outputs/                        # metrics CSV produced by run_experiment.py
  requirements.txt
```

## Execution instructions

```bash
pip install -r requirements.txt
python data/download_data.py

# Option A: full exploratory notebook (EDA, all 3 models, CV, chronological split)
jupyter notebook notebooks/fraud_detection_analysis.ipynb   # or nbconvert --execute

# Option B: standalone script, no Jupyter required
python run_experiment.py                              # Logistic Regression + Random Forest only
python run_experiment.py --with-autoencoder --epochs 20  # also trains the autoencoder (slower)
```

The notebook has already been executed end-to-end (outputs are saved in the `.ipynb`
file) and takes roughly 10-15 minutes to re-run in full, primarily due to Random Forest,
cross-validation, and autoencoder training. `run_experiment.py` shares its preprocessing
and model-building code with the notebook (`src/data_utils.py`, `src/models.py`), so the
two can never silently drift apart on hyperparameters or preprocessing, and either one can
be used to independently verify the other's results.

## Summary of findings

- The source's autoencoder (14-7-7-29 architecture, trained only on normal transactions)
  reproduces closely on ROC-AUC (~0.947 here vs. 0.94 reported).
- The source selects its reconstruction-error threshold by inspecting the test set's
  error distribution directly — a form of test-set leakage. We quantified this by
  comparing that method against a leakage-free, validation-based threshold; in this
  reproduction the two converged on the same threshold, but the practice remains unsound
  in general.
- The source reports only ROC-AUC despite 0.17% class imbalance; adding PR-AUC (~0.43-0.44)
  and MCC (~0.49-0.51) paints a less flattering picture.
- A simple Random Forest baseline (never tried in the source) substantially outperforms
  the autoencoder on every threshold-dependent metric (F1 0.822 vs. 0.490, MCC 0.833 vs.
  0.490), confirmed stable via 5-fold cross-validation and a chronological
  (train-on-past/test-on-future) split.
- The chronological split also revealed that Logistic Regression's precision is notably
  sensitive to split methodology (0.057 on a random split vs. 0.024 chronologically),
  while Random Forest is essentially unaffected — a model-specific fragility the random
  split alone would not have surfaced.
- Our own first-pass Amount scaling was itself leakage-prone (fit on the full dataset
  before splitting); this was caught and fixed by moving scaling into a per-split sklearn
  Pipeline (`src/models.py`), used consistently across the random split, the chronological
  split, and every cross-validation fold.

See `report/report.pdf` for the full critical evaluation, feature engineering analysis,
reproducibility analysis, experimental results, and executive summary.
