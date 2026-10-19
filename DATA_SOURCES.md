# Dataset source and redistribution evidence

Checked on 2026-10-19.

- Publisher account: Ammok on Hugging Face (not asserted to be the original collector).
- Repository: https://huggingface.co/datasets/Ammok/laptop_price_prediction
- CSV revision: https://huggingface.co/datasets/Ammok/laptop_price_prediction/blob/fcc3679105da35bc6874349f6372389476c5907a/laptops_test.csv
- Dataset card: https://huggingface.co/datasets/Ammok/laptop_price_prediction/blob/b7e3598/README.md
- Declared license: Apache-2.0. A copy of the card and license are in `licenses/`.
- Local SHA-256: `b6635750672b8febe4e846bcd529f2ef001fa27f64450c49b4bc88b867189b7e`.
- Upstream SHA-256: `019d1b6091337c8a0e5be885a30c7b1bace57035f3fa7ff289d0aea683deff72`.
- The files are byte-identical after CRLF-to-LF normalization and equal when parsed as CSV. Local bytes are preserved.

Redistribution relies on this matching upstream publication's declared license; the declaration is not an independent audit of the uploader's ownership. No original collection date, country, currency or collection methodology was established. No currency is inferred from the values.

The source CSV is preserved. Derived processed CSVs/reports remove exact duplicates, parse numeric fields and add engineered features as documented in `src/data_cleaning.py`. Raw data are not changed by the pipeline. Attribution applies to the dataset and its derived records, not automatically to the independently maintained application code. This project does not assign a new license to the owner's original source code.
