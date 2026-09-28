## Basetable construction scripts

This directory contains the scripts used to construct the different **basetable variants** explored during model development to evaluate which feature representations improved performance.

The scripts located directly in this folder (i.e. not inside a subdirectory) define the baseline basetable used for all models. These scripts must be executed **in the given numerical order** and produce the standard train, validation, and test sets.

The subdirectories contain **alternative basetable constructions** tailored to specific modeling hypotheses:
- **`without_svd/`**: builds train and test sets without SVD-reduced text features, mainly used for tree-based models such as XGBoost.
- **`adding extra features/`**: extends the baseline basetable with additional engineered features, computed on the training data only to avoid leakage, including:
  - brand-level mean log-price (target encoding),
  - category-level mean log-prices (three hierarchy levels),
  - sentiment score of the item description.

The extended basetable constructed in `adding_extra_features/` was evaluated during development but was not used in the final models, as it did not lead to improved validation performance.
