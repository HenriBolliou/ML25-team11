# Src/Models

This directory contains all model-building scripts used to train, tune, and evaluate the algorithms. Each script or subdirectory implements the full training and validation pipeline for a specific model.

Standalone `.R` scripts correspond to individual model families (e.g. linear models, tree-based models, GAM, BART, SVR), all trained on the basetable from the goldlayer.

The Lasso Feature Selection script is used to reduce dimensionality and optimize model performance. The resulting 300 features were consistently used as input for the tree-based models (Decision tree, Bagging, Random forest, BART).

For XGBoost and MLP, two separate subdirectories are present (`XgBoost` / `xgboost2` and `MLP` / `MLP2`). These reflect different feature representations and modeling choices: XGBoost was additionally trained without SVD components and with sample weights for high-priced items, while MLP was trained on a higher-dimensional SVD representation (k = 150 instead of 100).

Each subdirectory contains the tested hyperparameter configurations and the final selected model to ensure reproducibility.
