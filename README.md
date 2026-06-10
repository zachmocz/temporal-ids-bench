# temporal-ids-bench

Code for the paper **"Do Transformers Actually Help Intrusion Detection? A Temporal Sequence Evaluation on CIC-IDS2017"** by Zach Moczkodan and Hany Ragab (Royal Military College of Canada). See the paper for the methodology, experimental setup, and results; this repository covers the implementation only.

**Preprint:** [arXiv:2606.11098](https://arxiv.org/abs/2606.11098)

> The code will be released here upon acceptance.

## Overview

A single self-contained notebook that turns the flow-level CIC-IDS2017 dataset into a sequence-classification benchmark and runs every experiment end to end. From the raw flow table it:

1. groups flows by their five-tuple and builds length-`T` sliding windows per conversation, under two padding schemes (repeat-last, and zero-pad with an attention mask);
2. builds the train/test protocols — random 80/20, time-ordered, and a leakage-free group-by-five-tuple split in which windows are constructed *after* the partition;
3. trains nine models (Random Forest, linear SVM, MLP, image-CNN, 1D-CNN, GRU, LSTM, a CNN+Transformer encoder, and a smaller Transformer) over three seeds, with per-class weighting and early stopping;
4. runs the sequence-length sweep and an inference-latency benchmark;
5. writes every per-seed metric, summary CSV, and figure to `figures/`.

## Repository

| Path | Description |
|---|---|
| `Transformer_IDS_Final.ipynb` | The full pipeline above. Runs top to bottom on a single GPU. |
| `figures/` | Generated figures (PDF + PNG) and the per-seed / summary CSVs behind them, including `master_summary.csv`. |

## Running it

Dependencies: Python ≥ 3.10, TensorFlow ≥ 2.20, scikit-learn, numpy, pandas, matplotlib, seaborn. The CIC-IDS2017 Network-Flows data is pulled through the `nids-datasets` package.

Open the notebook, set the data and output paths (`XXXX_DATA_PATH` / `XXXX_FIG_DIR`, or edit the config cell), and run all cells. A full three-seed run takes roughly five to eight hours on an NVIDIA A100. Each experiment checkpoints its own CSVs, so individual blocks can be re-run without repeating the rest.

## Citation

```bibtex
@misc{moczkodan2026transformers,
  author        = {Moczkodan, Zach and Ragab, Hany},
  title         = {Do Transformers Actually Help Intrusion Detection?
                   A Temporal Sequence Evaluation on CIC-IDS2017},
  year          = {2026},
  eprint        = {2606.11098},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  doi           = {10.48550/arXiv.2606.11098}
}
```

## Contact

zach.moczkodan@gmail.com · zachery.moczkodan@forces.gc.ca · hany.ragab@{queensu, rmc-cmr}.ca

A license will accompany the code release.
