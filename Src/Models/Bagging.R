# --- BAGGING FINAL EXECUTION ON SUBSET (25K samples - Consolidated) ---
# Executes the complete workflow: Feature Selection, Dynamic Tuning,
# and Final Model training on the representative 25K sample set.
# NOTE: Feature selection indices are loaded from the RDS file.

# --- 0. Setup and Data Loading ---
library(randomForest) # Single-threaded RF implementation (used for Bagging)
library(glmnet) # Required for Lasso functionality (though not calculation)
library(tidyverse)
library(Matrix)

set.seed(123)

# Load the original (likely sparse) data
X_tr_original <- readRDS("Data/Gold/features_TFIDF/X_tr_final.rds")
X_val_original <- readRDS("Data/Gold/features_TFIDF/X_val_final.rds")
X_test_original <- readRDS("Data/Gold/features_TFIDF/X_test_final.rds")
y_tr <- readRDS("Data/Gold/features_TFIDF/y_tr.rds")
y_val <- readRDS("Data/Gold/features_TFIDF/y_val.rds")
test_df <- read_csv("Data/Silver/test_clean_joined.csv")

# --- 1. Evaluation Metric Function ---
# Custom RMSE function to evaluate in the original price scale (not log-scale)
rmse_price <- function(pred_log, true_log) {
  # Inverse transform the predictions and true values
  pred_price <- exp(pred_log) - 1
  true_price <- exp(true_log) - 1
  
  # Calculate the RMSE based on the PRICE values (pred_price - true_price)
  sqrt(mean((pred_price - true_price)^2))
}

# -----------------------------------------------------------------------
# 2. FEATURE SELECTION (Lasso) & DATA PREPARATION - LOAD FROM RDS
# Loads the top 300 indices calculated on the FULL dataset previously.
# -----------------------------------------------------------------------

cat("\n[1/4] Loading pre-calculated LASSO Features (TOP 300)...\n")

tryCatch({
  selected_features_indices <- readRDS("Src/Models/Lasso Feature Selection/top_300_feature_indices.rds")
}, error = function(e) {
  cat("ERROR: Could not load feature indices. Please run the separate Feature Selection Pipeline script first.\n")
  stop("Feature loading failed.")
})

TOP_N_FEATURES <- 300
X_all_sparse <- rbind(X_tr_original, X_val_original)
y_all <- c(y_tr, y_val) # Full response vector for final fit

# Apply selection and convert data to DENSE matrices
X_tr_dense <- as.matrix(X_tr_original[, selected_features_indices])
X_val_dense <- as.matrix(X_val_original[, selected_features_indices])
X_test_dense <- as.matrix(X_test_original[, selected_features_indices])
X_full_dense <- as.matrix(X_all_sparse[, selected_features_indices]) # Combined Train + Val

cat(sprintf("-> Feature selection loaded successfully. Training will use %d features.\n", ncol(X_tr_dense)))

# -----------------------------------------------------------------------
# 3. HYPERPARAMETER TUNING (Find best ntree/nodesize on 5K sample)
# -----------------------------------------------------------------------

cat("\n[2/4] Starting Bagging Hyperparameter Tuning (n=5K sample)...\n")

# Tuning parameters (OPTIMIZED FOR SPEED, 9 configurations)
BAG_MTY <- TOP_N_FEATURES # Bagging condition: mtry = 300
MAX_TUNE_ROWS <- 5000
N_FOLD_CV <- 3

bagging_grid <- expand.grid(
  ntree = c(50, 100, 200),
  nodesize = c(20, 50, 100)
)
G <- nrow(bagging_grid)

# Define 5K SUBSET FOR TUNING
set.seed(42)
tune_idx <- sample(1:nrow(X_tr_original), MAX_TUNE_ROWS)
X_tune <- as.matrix(X_tr_original[tune_idx, selected_features_indices])
y_tune <- y_tr[tune_idx]
n_tune <- nrow(X_tune)
folds <- sample(rep(1:N_FOLD_CV, length.out = n_tune))

best_rmse <- Inf
best_cfg <- NULL

for (i in seq_len(G)) {
  cfg <- bagging_grid[i, ]
  fold_rmses <- numeric(N_FOLD_CV)
  
  cat(sprintf("\n===== Config %d/%d: ntree=%d, nodesize=%d =====\n",
              i, G, cfg$ntree, cfg$nodesize))
  
  for (k in 1:N_FOLD_CV) {
    train_idx <- which(folds != k)
    val_idx <- which(folds == k)
    
    fit <- randomForest::randomForest(
      x = X_tune[train_idx, ], y = y_tune[train_idx],
      ntree = cfg$ntree, mtry = BAG_MTY, nodesize = cfg$nodesize, importance = FALSE
    )
    
    pred_k <- predict(fit, newdata = X_tune[val_idx, ])
    fold_rmses[k] <- rmse_price(pred_k, y_tune[val_idx])
  }
  
  rmse_mean <- mean(fold_rmses)
  
  cat(sprintf(" -> CV Mean RMSE: %.5f", rmse_mean))
  
  if (rmse_mean < best_rmse) {
    best_rmse <- rmse_mean
    best_cfg <- cfg
    cat(" (NEW BEST)")
  }
  cat("\n")
}

cat("\n==========================================================\n")
cat(sprintf("Bagging Tuning Complete. Best CV RMSE: %.5f\n", best_rmse))
print(best_cfg)


# -----------------------------------------------------------------------
# 4. FINAL TRAINING ON 25K SUBSET (Using best parameters)
# This is the target model for the final submission.
# -----------------------------------------------------------------------

# 4.1 Define Final Hyperparameters from tuning
FINAL_NTREE <- best_cfg$ntree[[1]]
FINAL_NODESIZE <- best_cfg$nodesize[[1]]
FINAL_SAMPLE_SIZE <- 25000
BAG_MTY_FINAL <- TOP_N_FEATURES

# 4.2 Define 25K SUBSET
set.seed(999)
subset_idx_final <- sample(1:nrow(X_tr_original), FINAL_SAMPLE_SIZE)

# Create the final Train subset matrix (with 300 features)
X_train_subset_final <- as.matrix(X_tr_original[subset_idx_final, selected_features_indices])
y_train_subset_final <- y_tr[subset_idx_final]


cat(sprintf("\n[3/4] STARTING FINAL BAGGING TRAINING on %d rows...\n", FINAL_SAMPLE_SIZE))
cat(sprintf(" Parameters: ntree=%d, nodesize=%d, mtry=%d (Single-threaded)\n",
            FINAL_NTREE, FINAL_NODESIZE, BAG_MTY_FINAL))


# 4.3 TRAIN THE FINAL MODEL ON THE SUBSET
start_time <- Sys.time()
final_bagging_subset <- randomForest::randomForest(
  x = X_train_subset_final,
  y = y_train_subset_final,
  ntree = FINAL_NTREE,
  mtry = BAG_MTY_FINAL,
  nodesize = FINAL_NODESIZE,
  importance = FALSE
)
training_time <- round(difftime(Sys.time(), start_time, units = "mins"), 1)


# -----------------------------------------------------------------------
# 5. EVALUATION AND SUBMISSION 
# -----------------------------------------------------------------------

cat("\n[4/4] Evaluation and Submission Generation...\n")

# Predict on the full X_val_dense (filtered 300 features)
val_preds_log <- predict(final_bagging_subset, newdata = X_val_dense)
final_val_rmse <- rmse_price(val_preds_log, y_val)


# Predict on Test Set (X_test_dense)
y_pred_test_log <- predict(final_bagging_subset, newdata = X_test_dense)
pred_kaggle <- exp(y_pred_test_log) - 1 # Inverse transform to price scale

# Submission
submission <- data.frame(
  index = test_df$index,
  price = pred_kaggle
)

output_filename_submission <- "submission_bagging_subset_final.csv"
write.csv(submission, output_filename_submission, row.names = FALSE)


cat("====================================================\n")
cat(sprintf(" FINAL BAGGING RESULT (%dK Subset):\n", FINAL_SAMPLE_SIZE / 1000))
cat(sprintf("Estimated Validation RMSE: %.5f\n", final_val_rmse))
cat(sprintf("Training Time: %.1f minutes\n", training_time))
cat(sprintf("Submission saved: '%s'\n", output_filename_submission))
cat("====================================================\n")

results_summary <- tibble(
  model = "Bagging (RF Subset)",
  final_rmse = final_val_rmse,
  sample_size_k = FINAL_SAMPLE_SIZE / 1000,
  training_time_min = training_time,
  ntree = FINAL_NTREE,
  nodesize = FINAL_NODESIZE,
  mtry = BAG_MTY_FINAL
)

output_filename_metrics <- "Data/Submissions/bagging_results_25K_subset.csv"
dir.create("Data/Submissions", recursive = TRUE, showWarnings = FALSE)
write_csv(results_summary, output_filename_metrics)

cat(sprintf("\n Metrics saved to: %s\n", output_filename_metrics))
