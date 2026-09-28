# --- DECISION TREE (RPART) FINAL EXECUTION ---
# This script integrates tuning results to train the definitive model
# on the full combined dataset, following best practices.
# NOTE: Feature selection indices are now loaded from the RDS file.

library(rpart)
library(Matrix)
library(tidyverse) # Ensure tidyverse is loaded for pipe operators and read_csv
library(glmnet)    # Required for LASSO Feature Selection (though not calculation)

# Load your data
X_tr_original <- readRDS("Data/Gold/Features_TFIDF/X_tr_final.rds")
y_tr <- readRDS("Data/Gold/Features_TFIDF/y_tr.rds")
X_val_original <- readRDS("Data/Gold/Features_TFIDF/X_val_final.rds")
y_val <- readRDS("Data/Gold/Features_TFIDF/y_val.rds")
X_test_original <- readRDS("Data/Gold/Features_TFIDF/X_test_final.rds") 
test_df <- read_csv("Data/Silver/test_clean_joined.csv")

# --- Evaluation Metric Function ---
# Custom RMSE function to evaluate in the original price scale (not log-scale)
rmse_price <- function(pred_log, true_log) {
  pred_price <- exp(pred_log) - 1
  true_price <- exp(true_log) - 1
  
  sqrt(mean((pred_price - true_price)^2))
}

# -----------------------------------------------------------------------
# 2. DATA PREPARATION & FEATURE SELECTION (CRUCIAL) - LOAD FROM RDS
# Loads the top 300 indices calculated on the FULL dataset previously (Efficiency).
# -----------------------------------------------------------------------

cat("\n[1/3] Loading pre-calculated LASSO Features (TOP 300)...\n")
X_combined_full <- rbind(X_tr_original, X_val_original)
y_combined_full <- c(y_tr, y_val)

TOP_N_FEATURES <- 300 

tryCatch({
  selected_features_indices <- readRDS("Src/Models/Lasso Feature Selection/top_300_feature_indices.rds")
}, error = function(e) {
  cat("ERROR: Could not load feature indices. Please run the separate Feature Selection Pipeline script first.\n")
  stop("Feature loading failed.")
})

# Define the sample index (Still needed for the tuning data split)
SAMPLE_FOR_TUNING <- 50000 
set.seed(123) 
# Note: The tuning sample is still drawn from the original X_tr_original
sample_idx <- sample(1:nrow(X_tr_original), min(SAMPLE_FOR_TUNING, nrow(X_tr_original)))

# --- Apply selection and convert to DENSE DataFrames ---
X_tr_sample_filtered <- as.matrix(X_tr_original[sample_idx, selected_features_indices])
X_combined_full_filtered <- as.matrix(X_combined_full[, selected_features_indices])
X_val_filtered <- as.matrix(X_val_original[, selected_features_indices])
X_test_filtered <- as.matrix(X_test_original[, selected_features_indices])

# Convert the combined sparse matrix to a dense dataframe for rpart
# This reduces the memory burden by using only 300 columns.
train_df_full <- as.data.frame(X_combined_full_filtered)
train_df_full$price_log <- y_combined_full

# Create sample dataframe for tuning
train_df_sample <- as.data.frame(X_tr_sample_filtered)
train_df_sample$price_log <- y_tr[sample_idx] # y_tr must be sampled too

# Prepare the validation set
val_df <- as.data.frame(X_val_filtered)

cat(sprintf("-> Feature selection loaded successfully. Training will use %d features.\n", ncol(X_tr_sample_filtered)))


# --- 1. Decision Tree Hyperparameter Tuning (Max Depth/Terminal Nodes) ---
cat("\n=== Decision Tree Hyperparameter Tuning (Max Depth) ===\n")

# --- TUNING SETUP ---
# Tune Max Depth (direct constraint on tree structure/terminal nodes)
# Set CP very low to allow max growth within the depth constraint
fixed_cp <- 0.00001 
maxdepth_values <- c(10, 15, 20, 25) # Test structure sizes
rmse_results <- numeric(length(maxdepth_values))

for(i in seq_along(maxdepth_values)) {
  current_depth <- maxdepth_values[i]
  # ... (Tuning loop executed, finds best_cp)
  model <- rpart(
    price_log ~ .,
    data = train_df_sample, # Use filtered sample for tuning speed
    method = "anova",
    control = rpart.control(
      cp = fixed_cp, # Allow maximum growth
      maxdepth = current_depth, # Tuning variable
      minsplit = 20,
      minbucket = 7
    )
  )
  y_pred <- predict(model, val_df)
  rmse_results[i] <- rmse_price(y_pred, y_val) 
  cat(sprintf("Testing maxdepth = %d ... RMSE: %.4f\n", current_depth, rmse_results[i]))
}

best_idx <- which.min(rmse_results)
best_maxdepth <- maxdepth_values[best_idx]
best_rmse_tuning <- rmse_results[best_idx]

cat("\n=== Tuning Results ===\n")
cat("Best maxdepth (for terminal nodes control):", best_maxdepth, "\n")
cat("Best Tuning RMSE (on price scale):", round(best_rmse_tuning, 4), "\n")


# --- 2. Train Final Model on FULL Data ---
cat("\n=== Final Model Training (on FULL dataset) ===\n")

final_model_full <- rpart(
  price_log ~ .,
  data = train_df_full, 
  method = "anova",
  control = rpart.control(
    cp = fixed_cp, # Use low CP to maintain structural control via maxdepth
    maxdepth = best_maxdepth, # Use the best depth found
    minsplit = 20,
    minbucket = 7
    # xval is now unnecessary since we use external validation
  )
)

# --- 3. Final Evaluation and Submission ---

# Predict on Validation Set (to get true generalization error)
y_pred_final <- predict(final_model_full, val_df)
final_rmse_dollars <- rmse_price(y_pred_final, y_val)

# Predict on Test Set
test_df_dense <- as.data.frame(X_test_filtered)
y_pred_test <- predict(final_model_full, test_df_dense)
pred_kaggle <- exp(y_pred_test) - 1

# Submission
submission <- data.frame(
  index = test_df$index,
  price = pred_kaggle
)

write.csv(submission, "submission_decision_tree_final.csv", row.names = FALSE)

cat("\n==================================================\n")
cat(" FINAL DECISION TREE RESULT:\n")
cat(" Model trained on FULL dataset with maxdepth =", best_maxdepth, "\n")
cat(" Final Estimated RMSE (on price scale):", round(final_rmse_dollars, 4), "\n")
cat(" Submission saved: 'submission_decision_tree_final.csv'\n")
