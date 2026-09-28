# --- BART MODEL SCRIPT (R) ---
# Executes all steps: Feature Selection (Lasso), Hyperparameter Tuning, 
# Final Model Training, and Final Validation Check in one efficient flow.
# NOTE: Feature Selection is loaded from the saved RDS file.

# --- 0. Setup and Data Loading ---
library(dbarts)
library(tidyverse)
library(glmnet) # Required for Lasso functionality (though not calculation)
library(Matrix) # Required for sparse matrix handling

set.seed(123)
# Load the original (likely sparse) data
X_tr_original <- readRDS("Data/Gold/features_TFIDF/X_tr_final.rds")
X_val_original <- readRDS("Data/Gold/features_TFIDF/X_val_final.rds")
X_test_original <- readRDS("Data/Gold/features_TFIDF/X_test_final.rds")
y_tr <- readRDS("Data/Gold/features_TFIDF/y_tr.rds")
y_val <- readRDS("Data/Gold/features_TFIDF/y_val.rds")

# --- 1. Evaluation Metric Function ---
# Custom RMSE function to evaluate in the original price scale (not log-scale)
rmse_price <- function(pred_log, true_log) {
  # Inverse transform the predictions and true values
  pred_price <- exp(pred_log) - 1
  true_price <- exp(true_log) - 1
  
  # Calculate the Root Mean Squared Error (RMSE)
  # FIX: Compares price scale against price scale (true_price)
  sqrt(mean((pred_price - true_price)^2))
}

# -----------------------------------------------------------------------
# 2. FEATURE SELECTION (Lasso) & DATA PREPARATION - LOAD FROM RDS
# Loads the top 300 indices calculated on the FULL dataset previously.
# -----------------------------------------------------------------------

cat("\n[1/3] Loading pre-calculated LASSO Features (TOP 300)...\n")

tryCatch({
  selected_features_indices <- readRDS("Src/Models/Lasso Feature Selection/top_300_feature_indices.rds")
}, error = function(e) {
  cat("ERROR: Could not load feature indices. Please run the separate Feature Selection Pipeline script first.\n")
  stop("Feature loading failed.")
})

# Combine all data (sparse)
X_all_sparse <- rbind(X_tr_original, X_val_original)
y_all <- c(y_tr, y_val)

# Apply selection and convert data to DENSE matrices
X_tr_dense <- as.matrix(X_tr_original[, selected_features_indices])
X_val_dense <- as.matrix(X_val_original[, selected_features_indices])
X_test_dense <- as.matrix(X_test_original[, selected_features_indices])
X_full_dense <- as.matrix(X_all_sparse[, selected_features_indices]) # Combined Train + Val
y_full_log <- y_all # Define combined response vector

cat(sprintf("-> Feature selection loaded successfully. Training will use %d features.\n", ncol(X_tr_dense)))

# -----------------------------------------------------------------------
# 3. HYPERPARAMETER TUNING (Find best ntree and k)
# Determines the optimal structural parameters on a subset.
# -----------------------------------------------------------------------

cat("\n[2/3] Starting BART Hyperparameter Tuning (n=30k)...\n")

MAX_TUNE_ROWS <- 30000
set.seed(42)
tune_idx <- sample(1:nrow(X_tr_dense), min(MAX_TUNE_ROWS, nrow(X_tr_dense)))
X_tune <- X_tr_dense[tune_idx, ]
y_tune <- y_tr[tune_idx]

bart_grid <- expand.grid(
  ntree = c(50, 200, 500), # Number of trees
  k = c(2, 3, 5), # Regularization prior
  nskip = 200, # Fixed for tuning efficiency
  ndpost = 1000 # Fixed for tuning efficiency
)

best_rmse <- Inf
best_cfg <- NULL

for (i in seq_len(nrow(bart_grid))) {
  cfg <- bart_grid[i, ]
  # --- BART fit for this configuration ---
  bart_fit <- dbarts::bart(x.train = X_tune, y.train = y_tune, x.test = X_val_dense, 
                           ntree = cfg$ntree, nskip = cfg$nskip, ndpost = cfg$ndpost, k = cfg$k, verbose = FALSE)
  
  current_rmse <- rmse_price(colMeans(bart_fit$yhat.test), y_val)
  
  # FIX: Add verbose output for each configuration
  cat(sprintf("\n===== Config %d/%d: ntree=%d, k=%.1f => RMSE: %.5f", 
              i, nrow(bart_grid), cfg$ntree, cfg$k, current_rmse))
  
  if (current_rmse < best_rmse) {
    best_rmse <- current_rmse
    best_cfg = cfg
    cat(" (NEW BEST)")
  }
  cat("\n")
}

# FINAL OPTIMAL PARAMETERS from tuning
FINAL_NTREE <- best_cfg$ntree[[1]]
FINAL_K <- best_cfg$k[[1]]
FINAL_NSKIP <- 500 # Increased for final fit stability
FINAL_NDPOST <- 2000 # Increased for final fit accuracy

cat(sprintf("-> Tuning Complete. Best Config: ntree=%d, k=%.1f (RMSE: %.5f)\n", 
            FINAL_NTREE, FINAL_K, best_rmse))

# -----------------------------------------------------------------------
# 4. FINAL TRAINING, PREDICTION, AND RMSE CHECK
# Trains on the full dataset with the optimal architecture.
# -----------------------------------------------------------------------

cat("\n[3/3] Starting FINAL Model Training and Prediction...\n")

# Train the final model on the full data (Train + Val)
set.seed(1234) 
final_bart_fit <- dbarts::bart(
  x.train = X_full_dense, 
  y.train = y_all, # Use the correctly defined y_all vector
  x.test = X_test_dense, # Predictions on Test Set generated during training
  ntree = FINAL_NTREE, 
  nskip = FINAL_NSKIP, 
  ndpost = FINAL_NDPOST, 
  k = FINAL_K, 
  verbose = TRUE,
  keeptrees = TRUE # CRITICAL for later prediction checks
)

# --- A. Final Prediction and Submission ---
pred_test_log_mean <- colMeans(final_bart_fit$yhat.test)
pred_test <- exp(pred_test_log_mean) - 1
test_df <- read_csv("Data/Silver/test_clean_joined.csv")

submission <- tibble(index = test_df$index, price = pred_test)
dir.create("Data/Submissions", recursive = TRUE, showWarnings = FALSE)
write_csv(submission, "Data/Submissions/submission_BART_final_for_real.csv")
cat("-> Submission saved to 'submission_BART_final.csv'\n")

# --- B. Final Validation Check (Generalization Error) ---
val_preds_fit <- predict(final_bart_fit, X_val_dense)

if (is.matrix(val_preds_fit)) {
  val_preds_log <- colMeans(val_preds_fit)
} else {
  val_preds_log <- val_preds_fit
}

final_val_rmse <- rmse_price(val_preds_log, y_val)
cat(sprintf(" FINAL MODEL COMPLETE. Estimated Validation RMSE: %.5f\n", final_val_rmse))
