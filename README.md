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
  src/data_utils.py               # data loading / inspection / split helpers
  src/eval_utils.py               # metric computation / plotting helpers
  notebooks/fraud_detection_analysis.ipynb   # full analysis (EDA, feature engineering,
                                              # 3 models, evaluation, error analysis)
  report/report.pdf               # full written report (8 required sections)
  report/figures/                 # figures extracted from the executed notebook
  requirements.txt
```

## Execution instructions

```bash
pip install -r requirements.txt
python data/download_data.py
jupyter notebook notebooks/fraud_detection_analysis.ipynb   # or nbconvert --execute
```

The notebook has already been executed end-to-end (outputs are saved in the `.ipynb`
file) and takes roughly 5-10 minutes to re-run in full, primarily due to Random Forest
and autoencoder training.

## Summary of findings

- The source's autoencoder (14-7-7-29 architecture, trained only on normal transactions)
  reproduces closely on ROC-AUC (~0.948 here vs. 0.94 reported).
- The source selects its reconstruction-error threshold by inspecting the test set's
  error distribution directly — a form of test-set leakage. We quantified this by
  comparing that method against a leakage-free, validation-based threshold; in this
  reproduction the effect was small, but the practice remains unsound in general.
- The source reports only ROC-AUC despite 0.17% class imbalance; adding PR-AUC (~0.40)
  and MCC (~0.46-0.47) paints a less flattering picture.
- A simple Random Forest baseline (never tried in the source) substantially outperforms
  the autoencoder on every threshold-dependent metric (F1 0.822 vs. 0.455, MCC 0.833 vs.
  0.462).

See `report/report.pdf` for the full critical evaluation, feature engineering analysis,
reproducibility analysis, experimental results, and executive summary.
