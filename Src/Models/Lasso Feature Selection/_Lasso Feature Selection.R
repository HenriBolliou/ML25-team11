# --- FEATURE SELECTION PIPELINE ---
# Executes the LASSO cross-validation once to find and save the top 300 
# feature indices for use in all subsequent models (BART, RF, etc.).

library(glmnet)
library(Matrix)
library(tidyverse)

# --- CONFIGURATION ---
TOP_N_FEATURES <- 300 
# SAMPLE_SIZE_LASSO constant is removed to use the full dataset.

# Load necessary data objects (Assumed to be in the environment or path)
X_tr_original <- readRDS("Data/Gold/Features_TFIDF/X_tr_final.rds")
X_val_original <- readRDS("Data/Gold/Features_TFIDF/X_val_final.rds")
y_tr <- readRDS("Data/Gold/Features_TFIDF/y_tr.rds")
y_val <- readRDS("Data/Gold/Features_TFIDF/y_val.rds")

# 1. Combine Data (FULL DATASET)
X_all_sparse <- rbind(X_tr_original, X_val_original)
y_all <- c(y_tr, y_val)

# 2. Assign the full dataset for LASSO CV
X_lasso_sample <- X_all_sparse # Full sparse matrix
y_lasso_sample <- y_all # Full response vector

cat(sprintf("Starting LASSO CV on %d samples (FULL DATASET) to find top %d features...\n", nrow(X_lasso_sample), TOP_N_FEATURES))

# 3. Execute LASSO CV
# NOTE: This step is computationally expensive and will take longer than the sampled version.
lasso_cv <- glmnet::cv.glmnet(
  x = X_lasso_sample, 
  y = y_lasso_sample, 
  alpha = 1, 
  nfolds = 5, 
  standardize = TRUE
)

# 4. Select top feature indices
lasso_coef <- as.matrix(coef(lasso_cv, s = "lambda.min"))
feature_importance <- abs(lasso_coef[-1, ]) 
selected_features_indices <- order(feature_importance, decreasing = TRUE)[1:TOP_N_FEATURES]
selected_features_indices <- selected_features_indices[!is.na(selected_features_indices)] 

# 5. Save the indices list (This is the crucial step)
dir.create("Data/Processed", recursive = TRUE, showWarnings = FALSE)
saveRDS(selected_features_indices, "Src/Models/Lasso Feature Selection/top_300_feature_indices.rds")

cat(sprintf(" Feature indices saved successfully to 'Src/Models/Lasso Feature Selection/top_300_feature_indices.rds'\n"))
