# --- RANDOM FOREST TUNING AND FITTING SCRIPT (RANGER) ---
# Executes Random Forest (RF) tuning using the multi-threaded RANGER package.
# RF tunes num.trees, min.node.size, and mtry across 27 configurations.

# --- 0. Setup and Data Loading ---
library(ranger)        # FAST multi-threaded Random Forest implementation (CRUCIAL for speed)
library(glmnet)        # Package for Lasso functionality (not calculation)
library(tidyverse)
library(Matrix)
library(parallel)      # For detecting cores for num.threads

set.seed(123)

# Load the original (likely sparse) data
X_tr_original <- readRDS("Data/Gold/features_TFIDF/X_tr_final.rds")
X_val_original <- readRDS("Data/Gold/features_TFIDF/X_val_final.rds")
X_test_original <- readRDS("Data/Gold/features_TFIDF/X_test_final.rds")
y_tr <- readRDS("Data/Gold/features_TFIDF/y_tr.rds")
y_val <- readRDS("Data/Gold/features_TFIDF/y_val.rds")
test_df <- read_csv("Data/Silver/test_clean_joined.csv")

# --- 1. Evaluation Metric Function ---
rmse_price <- function(pred_log, true_log) {
  # Inverse transform the predictions and true values
  pred_price <- exp(pred_log) - 1
  true_price <- exp(true_log) - 1
  
  # Calculate the RMSE based on the PRICE values (pred_price - true_price)
  sqrt(mean((pred_price - true_price)^2))
}

# -----------------------------------------------------------------------
# 2. FEATURE SELECTION (Lasso) & DATA PREPARATION - LOAD FROM RDS
# Loads the top 300 indices calculated on the FULL dataset previously (Efficiency).
# -----------------------------------------------------------------------

cat("\n[1/4] Loading pre-calculated LASSO Features (TOP 300)...\n")

tryCatch({
  selected_features_indices <- readRDS("Src/Models/Lasso Feature Selection/top_300_feature_indices.rds")
}, error = function(e) {
  cat("ERROR: Could not load feature indices. Please run the separate Feature Selection Pipeline script first.\n")
  stop("Feature loading failed.")
})

# Combine all data (sparse)
X_all_sparse <- rbind(X_tr_original, X_val_original)
y_all <- c(y_tr, y_val) # Full response vector for final fit

# Apply selection and convert data to DENSE matrices
X_tr_dense <- as.matrix(X_tr_original[, selected_features_indices])
X_val_dense <- as.matrix(X_val_original[, selected_features_indices])
X_test_dense <- as.matrix(X_test_original[, selected_features_indices])
X_full_dense <- as.matrix(X_all_sparse[, selected_features_indices]) 

# Create DataFrames for RANGER Input
df_tr_tune <- data.frame(y = y_tr, X_tr_dense)
df_full_fit <- data.frame(y = y_all, X_full_dense)
df_val_dense <- data.frame(X_val_dense)
df_test_dense <- data.frame(X_test_dense)
TOTAL_FEATURES <- ncol(X_tr_dense) # p = 300


cat(sprintf("-> Feature selection loaded successfully. Training will use %d features.\n", TOTAL_FEATURES))

# -----------------------------------------------------------------------
# 3. HYPERPARAMETER TUNING (Random Forest Specific with RANGER)
# Tunes: num.trees, min.node.size, and mtry.
# -----------------------------------------------------------------------

cat("\n[2/4] Starting Random Forest Hyperparameter Tuning (RANGER)...\n")
cat("NOTE: Tuning mtry around sqrt(300) ~ 17. Testing 3x3x3 = 27 configurations.\n")

MAX_TUNE_ROWS <- 30000 # Use 30k rows for better confidence in RF tuning
N_FOLD_CV <- 3 

# Define the tuning grid: 3 parameters to tune (3 values each)
rf_grid <- expand.grid(
  ntree = c(100, 300, 500), 
  nodesize = c(5, 10, 20), 
  mtry = c(10, 17, 25) # Number of predictors to sample at each split
)
G <- nrow(rf_grid)

# Subsample the training data for tuning
set.seed(42)
n_tr <- nrow(df_tr_tune)
tune_idx <- sample(1:n_tr, min(MAX_TUNE_ROWS, n_tr))

df_tune <- df_tr_tune[tune_idx, ]
n_tune <- nrow(df_tune)

# Divide the subsample into K-folds
folds <- sample(rep(1:N_FOLD_CV, length.out = n_tune))

best_rmse <- Inf
best_cfg <- NULL

for (i in seq_len(G)) {
  cfg <- rf_grid[i, ]
  fold_rmses <- numeric(N_FOLD_CV)
  
  cat(sprintf("\n===== Config %d/%d: ntree=%d, nodesize=%d, mtry=%d =====\n",
              i, G, cfg$ntree, cfg$nodesize, cfg$mtry))
  
  # K-Fold CV Loop on the subsample
  for (k in 1:N_FOLD_CV) {
    train_idx <- which(folds != k)
    val_idx <- which(folds == k)
    
    # Use RANGER: fast, multi-threaded
    fit <- ranger::ranger(
      formula = y ~ ., 
      data = df_tune[train_idx, ],
      num.trees = cfg$ntree,
      mtry = cfg$mtry, 
      min.node.size = cfg$nodesize,
      seed = 123,
      num.threads = max(1, parallel::detectCores() - 1) 
    )
    
    pred_obj <- predict(fit, data = df_tune[val_idx, ])
    fold_rmses[k] <- rmse_price(pred_obj$predictions, df_tune$y[val_idx])
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
cat(sprintf("RF Tuning Complete (RANGER). Best CV RMSE: %.5f\n", best_rmse))
print(best_cfg)


# -----------------------------------------------------------------------
# 4. FINAL MODEL TRAINING (on full Train + Val data)
# -----------------------------------------------------------------------

cat("\n[3/4] Starting Final Random Forest Fit on Full Dataset (RANGER) ---\n")

FINAL_NTREE <- best_cfg$ntree[[1]]
FINAL_NODESIZE <- best_cfg$nodesize[[1]]
FINAL_MTY <- best_cfg$mtry[[1]]
FINAL_NTREE_MAX <- 1500 # Maximize trees for final production fit

cat(sprintf("Final RF Parameters: num.trees=%d, min.node.size=%d, mtry=%d\n",
            FINAL_NTREE_MAX, FINAL_NODESIZE, FINAL_MTY))
cat(" RANGER: Multi-threaded training commencing. Expected completion time is minutes/low hours. ✅\n")

# Use the full dataset (df_full_fit) for the final fit
set.seed(123) 
final_rf_fit <- ranger::ranger(
  formula = y ~ ., 
  data = df_full_fit,
  num.trees = FINAL_NTREE_MAX, 
  mtry = FINAL_MTY, 
  min.node.size = FINAL_NODESIZE, 
  importance = 'none',
  seed = 123,
  num.threads = max(1, parallel::detectCores() - 1)
)

# -----------------------------------------------------------------------
# 5. PREDICTION & DEFINITIVE RMSE CALCULATION
# -----------------------------------------------------------------------

cat("\n[4/4] Generating Predictions and Calculating Final RMSE ---\n")

# --- 5a. Evaluation on Validation Set (RMSE) ---
val_preds_obj <- predict(final_rf_fit, data = df_val_dense)
final_val_rmse <- rmse_price(val_preds_obj$predictions, y_val)

cat(sprintf(" Final Estimated RMSE on Validation Set: %.5f\n", final_val_rmse))

# --- 5b. Prediction on Test Set (Submission) ---
pred_test_obj <- predict(final_rf_fit, data = df_test_dense)
pred_test_log <- pred_test_obj$predictions

pred_test <- exp(pred_test_log) - 1

submission <- tibble(
  index = test_df$index,
  price = pred_test
)

dir.create("Data/Submissions", recursive = TRUE, showWarnings = FALSE)
write_csv(submission, "Data/Submissions/submission_RandomForest_RANGER_FINAL.csv")

cat("\nRandom Forest Model Complete (RANGER). Submission saved.\n")
