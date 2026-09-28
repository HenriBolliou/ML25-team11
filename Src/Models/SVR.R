# Install and load the necessary libraries
library(e1071)      
library(tidyverse)  
library(LiblineaR)  
library(Matrix)

# Ensure reproducibility
set.seed(123)


# =========================================================
# 1. LOADING DATA
# =========================================================
cat("Loading Data...\n")

# Load feature matrices (sparse matrices)
X_tr   <- readRDS("Data/Gold/Features_TFIDF/X_tr_final.rds")
X_val  <- readRDS("Data/Gold/Features_TFIDF/X_val_final.rds")
X_test <- readRDS("Data/Gold/Features_TFIDF/X_test_final.rds")

# Load targets (ensure they are vectors)
y_tr   <- as.vector(readRDS("Data/Gold/Features_TFIDF/y_tr.rds"))
y_val  <- as.vector(readRDS("Data/Gold/Features_TFIDF/y_val.rds"))

# Combine train and validation sets for final training
X_combined <- rbind(X_tr, X_val)
y_combined <- c(y_tr, y_val)

# Load test data for submission
test_df <- read_csv("Data/Silver/test_clean_joined.csv")

cat("Training set size: ", length(y_tr), "\n")
cat("Validation set size: ", length(y_val), "\n\n")

# Helper function: RMSE (Log -> Price conversion)
rmse_price <- function(pred_log, y_log) {
  pred_price <- exp(pred_log) - 1
  y_price <- exp(y_log) - 1
  return(sqrt(mean((pred_price - y_price)^2)))
}


# =========================================================
# 2. SVR WITH LINEAR KERNEL (LiblineaR)
# =========================================================
# --- 2a. Hyperparameter Tuning ---
costs_lin <- c(0.01, 0.1, 1, 10, 100)
best_mse_lin <- Inf
best_cost_lin <- NA

for(cost in costs_lin) {
  mse_val <- LiblineaR(data = X_tr, target = y_tr, type = 13, cost = cost, cross = 5, svr_eps = 0.001)
  
  cat(sprintf("  Cost: %s - MSE: %s\n", cost, mse_val))
  
  if(mse_val < best_mse_lin) {
    best_mse_lin <- mse_val
    best_cost_lin <- cost
  }
}
cat("Best Linear Cost found:", best_cost_lin, "\n")


# --- 2b. Train on training data & validate on validation set ---
model_lin <- LiblineaR(data = X_tr, target = y_tr, type = 13, cost = best_cost_lin, svr_eps = 0.001)
pred_lin_val <- predict(model_lin, X_val)$predictions
rmse_lin <- rmse_price(pred_lin_val, y_val)

cat(sprintf(">> Linear SVR RMSE on Validation Set: %.4f\n", rmse_lin))


# --- 2c. Retrain on (train + validation) & predict test ---
final_model_lin <- LiblineaR(data = X_combined, target = y_combined, type = 13, cost = best_cost_lin, svr_eps = 0.001)
final_pred_lin <- predict(final_model_lin, X_test)$predictions

# Save linear submission
submission_lin <- tibble(
  index = test_df$index,
  price = exp(final_pred_lin) - 1
)
write_csv(submission_lin, "submission_svr_linear.csv")
cat("Saved: submission_svr_linear.csv\n\n")


# =========================================================
# 3. SVR WITH NON-LINEAR KERNELS (Radial & Polynomial)
# =========================================================
# Note: Non-linear kernels scale quadratically O(n^2). 
# We MUST use subsets for tuning and training to stay within memory/time limits.

# --- 3a. Create Subsets ---
# 1. Tuning Subset (Small, for fast grid search)
N_TUNE <- 5000 
idx_tune <- sample(1:nrow(X_tr), N_TUNE)
X_tune <- X_tr[idx_tune, ]
y_tune <- y_tr[idx_tune]

# 2. Training Subset (Medium, for feasibility)
N_TRAIN_SUB <- 50000 
idx_train_sub <- sample(1:nrow(X_tr), N_TRAIN_SUB)
X_train_sub <- X_tr[idx_train_sub, ]
y_train_sub <- y_tr[idx_train_sub]

cat("Subsets created: Tuning (n=", N_TUNE,"), Training (n=", N_TRAIN_SUB,")\n")


# ***********************
# 3.1: RADIAL KERNEL
# ***********************
# --- 3.1.a Hyperparameter Tuning (On small subset) ---
radial_grid <- list(
  cost = c(0.1, 1, 10, 50, 100),
  gamma = c(0.0001, 0.001, 0.01, 0.1, 1),
  epsilon = c(0.1) # keeping epsilon fixed to save time
)

tune_rad <- tune(svm, 
                 train.x = X_tune, 
                 train.y = y_tune, 
                 kernel = "radial",
                 ranges = radial_grid,
                 tunecontrol = tune.control(cross = 3)) # 3-fold for speed

best_params_rad <- tune_rad$best.parameters
print(best_params_rad)


# --- 3.1.b Train on training data & validate on validation set ---
model_rad <- svm(x = X_train_sub, y = y_train_sub, kernel = "radial",
                 cost = best_params_rad$cost, 
                 gamma = best_params_rad$gamma,
                 epsilon = best_params_rad$epsilon)

pred_rad_val <- predict(model_rad, X_val)
rmse_rad <- rmse_price(pred_rad_val, y_val)

cat(sprintf(">> Radial SVR RMSE on Validation Set: %.4f\n", rmse_rad))


# --- 3.1.c Retrain on (train + validation) & predict test ---
# We sample a subset from the combined data because full data is too heavy
idx_comb_sub <- sample(1:nrow(X_combined), N_TRAIN_SUB) 

final_model_rad <- svm(x = X_combined[idx_comb_sub, ], 
                       y = y_combined[idx_comb_sub], 
                       kernel = "radial",
                       cost = best_params_rad$cost, 
                       gamma = best_params_rad$gamma,
                       epsilon = best_params_rad$epsilon)

final_pred_rad <- predict(final_model_rad, X_test)

# Save radial submission
submission_rad <- tibble(
  index = test_df$index, 
  price = exp(final_pred_rad) - 1
)
write_csv(submission_rad, "submission_svr_radial.csv")
cat("Saved: submission_svr_radial.csv\n")


# ***********************
# 3.2: POLYNOMIAL KERNEL
# ***********************
# --- 3.2.a Hyperparameter Tuning (On small subset) ---
poly_grid <- list(
  degree = c(2, 3),
  cost = c(0.1, 1, 10, 50, 100)
)

tune_poly <- tune(svm, 
                  train.x = X_tune, 
                  train.y = y_tune, 
                  kernel = "polynomial",
                  ranges = poly_grid,
                  tunecontrol = tune.control(cross = 3))

best_params_poly <- tune_poly$best.parameters
print(best_params_poly)


# --- 3.2.b Train on training data & validate on validation set ---
model_poly <- svm(x = X_train_sub, y = y_train_sub, kernel = "polynomial",
                  degree = best_params_poly$degree,
                  cost = best_params_poly$cost)

pred_poly_val <- predict(model_poly, X_val)
rmse_poly <- rmse_price(pred_poly_val, y_val)
cat(sprintf(">> Polynomial SVR RMSE on Validation Set: %.4f\n", rmse_poly))


# --- 3.2.c Retrain on (train + validation) & predict test ---
# We sample a subset from the combined data because full data is too heavy
final_model_poly <- svm(x = X_combined[idx_comb_sub, ], 
                        y = y_combined[idx_comb_sub], 
                        kernel = "polynomial",
                        degree = best_params_poly$degree,
                        cost = best_params_poly$cost)

final_pred_poly <- predict(final_model_poly, X_test)

# Save polynomial submission
submission_poly <- tibble(
  index = test_df$index, 
  price = exp(final_pred_poly) - 1
)
write_csv(submission_poly, "submission_svr_poly.csv")
cat("Saved: submission_svr_poly.csv\n")


# =========================================================
# 4. FINAL SUMMARY
# =========================================================
cat("\n==============================================\n")
cat("FINAL VAL RMSE RESULTS SUMMARY:\n")
cat(sprintf("1. Linear Model (Full Data):  %.4f\n", rmse_lin))
cat(sprintf("2. Radial Model (Subset):     %.4f\n", rmse_rad))
cat(sprintf("3. Poly Model   (Subset):     %.4f\n", rmse_poly))
cat("==============================================\n")
