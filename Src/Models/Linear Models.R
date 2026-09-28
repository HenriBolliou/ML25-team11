library(glmnet)
library(dplyr)
library(tidyverse)
library(Matrix)

# Loading the data
X_tr   <- readRDS("Data/Gold/Features_TFIDF/X_tr_final.rds")
X_val  <- readRDS("Data/Gold/Features_TFIDF/X_val_final.rds")
X_test <- readRDS("Data/Gold/Features_TFIDF/X_test_final.rds")

y_tr   <- readRDS("Data/Gold/Features_TFIDF/y_tr.rds")    
y_val  <- readRDS("Data/Gold/Features_TFIDF/y_val.rds")   

test <- read_csv("Data/Silver/test_clean_joined.csv")

# RMSE function
rmse_price <- function(pred_log, true_log) {
  pred_price  <- exp(pred_log)  - 1
  true_price  <- exp(true_log)  - 1
  sqrt(mean((pred_price - true_price)^2))
}

set.seed(123)
###############################################################################

# =====================================================================
# Lambda-grids 
# =====================================================================

lambda_grid_ridge <- 10 ^ seq(-4, 1, length = 50)
lambda_grid_lasso <- 10 ^ seq(-4, 1, length = 50)

range(lambda_grid_ridge)
range(lambda_grid_lasso)

# =====================================================================
# OLS (lambda = 0)  (need enough memory to run this!)
# =====================================================================
Xtr_dense  <- as.matrix(X_tr)
Xval_dense <- as.matrix(X_val)

# OLS 
fit <- lm.fit(x = cbind(1, Xtr_dense), y = y_tr)

# Due to rank deficiency in the TF–IDF design matrix,
# non-identifiable OLS coefficients are set to zero.
# some coefficients become NA after this function.
beta <- fit$coefficients
beta[is.na(beta)] <- 0  # non-identifiable coefficients set to zero
pred_val_ols_log <- as.numeric(cbind(1, Xval_dense) %*% beta)
rmse_val_ols <- rmse_price(pred_val_ols_log, y_val)
rmse_val_ols

# =====================================================================
# Ridge (alpha = 0) with cross-validation 
# =====================================================================
ridge_cv <- cv.glmnet(
  X_tr, y_tr,
  alpha       = 0,
  lambda      = lambda_grid_ridge,  
  nfolds      = 5,
  standardize = TRUE
)

best_lambda_ridge   <- ridge_cv$lambda.min
pred_val_ridge_log  <- as.numeric(predict(ridge_cv, X_val, s = "lambda.min"))
rmse_val_ridge      <- rmse_price(pred_val_ridge_log, y_val)


# =====================================================================
# Lasso (alpha = 1) with cross-validation 
# =====================================================================
lasso_cv <- cv.glmnet(
  X_tr, y_tr,
  alpha       = 1,
  lambda      = lambda_grid_lasso,  
  nfolds      = 5,
  standardize = TRUE
)

best_lambda_lasso   <- lasso_cv$lambda.min
pred_val_lasso_log  <- as.numeric(predict(lasso_cv, X_val, s = "lambda.min"))
rmse_val_lasso      <- rmse_price(pred_val_lasso_log, y_val)

###############################################################################
# Summary on validation set
# =====================================================================
rmse_val_summary <- tibble(
  model = c("OLS", "ridge", "lasso"),
  rmse  = c(rmse_val_ols, rmse_val_ridge, rmse_val_lasso)
)

print(rmse_val_summary)

best_model <- rmse_val_summary$model[which.min(rmse_val_summary$rmse)]
cat("Best model on validation set:", best_model, "\n")

###############################################################################
# Final Fit on all data (train + val) 
# =====================================================================
X_all <- rbind(X_tr, X_val)
y_all <- c(y_tr, y_val)

if (best_model == "OLS") {
  Xall_dense  <- as.matrix(X_all)
  final_all <- lm.fit(
    x = cbind(1, Xall_dense),
    y = y_all
  )
} else if (best_model == "ridge") {
  final_all <- glmnet(
    X_all, y_all,
    alpha       = 0,
    lambda      = best_lambda_ridge,
    standardize = TRUE
  )
} else if (best_model == "lasso") {
  final_all <- glmnet(
    X_all, y_all,
    alpha       = 1,
    lambda      = best_lambda_lasso,
    standardize = TRUE
  )
}

###############################################################################

# Predict op test
# CHANGE X_TEST TO xtest_dense IF BEST MODEL IS OLS
if (best_model == "OLS") {
  Xtest_dense <- as.matrix(X_test) 
  b <- final_all$coefficients
  b[is.na(b)] <- 0
  pred_kaggle_log <- as.numeric(
    cbind(1, Xtest_dense) %*% b
  )
  
} else {
  
  pred_kaggle_log <- as.numeric(
    predict(final_all, X_test)
  )
}
pred_kaggle     <- exp(pred_kaggle_log) - 1

# Kaggle submission
test <- test %>%
  mutate(index = as.integer(as.numeric(index)))

submission <- data.frame(
  index = test$index,
  price = pred_kaggle
)

write.csv(submission, "Data/Submissions/submission_linear.csv", row.names = FALSE)
