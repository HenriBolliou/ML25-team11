library(tidyverse)
library(gbm)
library(Matrix)
library(glmnet) # Required for LASSO

rm(list = ls())
set.seed(123)


# loading the data

X_tr   <- readRDS("Data/Gold/features_TFIDF/X_tr_final.rds")
X_val  <- readRDS("Data/Gold/features_TFIDF/X_val_final.rds")
X_test <- readRDS("Data/Gold/features_TFIDF/X_test_final.rds")

y_tr   <- readRDS("Data/Gold/features_TFIDF/y_tr.rds")      
y_val  <- readRDS("Data/Gold/features_TFIDF/y_val.rds")     

test_df <- read_csv("Data/Silver/test_clean_joined.csv")

# Determine common columns first (Essential to ensure features exist in all sets)
common_cols <- Reduce(intersect, list(colnames(X_tr), colnames(X_val), colnames(X_test)))

#feature engineering with lasso due to long running time

cat("Performing LASSO feature selection (Supervised)...\n")

# 1. Subsample data for speed (25% or max 25k rows)
# LASSO on full sparse matrices can be slow; a subset is sufficient for ranking.
set.seed(123)
subset_size_lasso <- floor(0.25 * nrow(X_tr))
sample_idx  <- sample(1:nrow(X_tr), subset_size_lasso)

# Use only common_cols
X_tr_small <- X_tr[sample_idx, common_cols]
y_tr_small <- y_tr[sample_idx]

# 2. Run LASSO with Cross Validation
# alpha = 1 means LASSO (L1 regularization)
lasso_cv <- cv.glmnet(
  x = X_tr_small, 
  y = y_tr_small, 
  alpha = 1, 
  nfolds = 5,
  standardize = TRUE, 
  type.measure = "mse"
)

# 3. Extract Coefficients
# We look at the coefficients at the optimal lambda (lambda.min)
lasso_coef <- coef(lasso_cv, s = "lambda.min")

# Convert to vector, remove Intercept (first row)
# Note: lasso_coef is a sparse matrix.
feature_importance <- abs(as.vector(lasso_coef)[-1]) # -1 removes (Intercept)
feature_names      <- lasso_coef@Dimnames[[1]][-1]   # Get corresponding names

# Create a named vector for sorting
names(feature_importance) <- feature_names

# 4. Select Top N Features based on LASSO importance
n_top_features <- 300
top_features   <- names(sort(feature_importance, decreasing = TRUE))[1:n_top_features]

# Safety: Ensure these are definitely in common_cols
common_top_features <- intersect(top_features, common_cols)

cat("LASSO selection complete.\n")
cat("Selected", length(common_top_features), "features with highest predictive power.\n")



#prepare dataframes

# Create dataframes with ONLY the LASSO selected features
df_train <- as.data.frame(as.matrix(X_tr[, common_top_features]))
df_val   <- as.data.frame(as.matrix(X_val[, common_top_features]))
df_test  <- as.data.frame(as.matrix(X_test[, common_top_features]))

# Make names safe (remove spaces/special chars for GBM)
safe_names <- make.names(common_top_features, unique = TRUE)
colnames(df_train) <- safe_names
colnames(df_val)   <- safe_names
colnames(df_test)  <- safe_names

# Add target to train
df_train$target <- y_tr


###Hyperparameter tuning
# Use  30,000 rows for tuning due to long running time 

subset_size_tune <- 30000
if (subset_size_tune > nrow(df_train)) subset_size_tune <- nrow(df_train)

set.seed(123)
train_indices   <- sample(1:nrow(df_train), size = subset_size_tune)
df_train_subset <- df_train[train_indices, ]

cat(sprintf("\nStep 1: Tuning GBM on subset of train (%d rows)...\n", nrow(df_train_subset)))

#set grid search
search_grid <- expand.grid(
  interaction.depth = c(3,5,7),       
  shrinkage = c( 0.1)        
  
)


results_df <- data.frame()

for (i in 1:nrow(search_grid)) {
  
  
  curr_depth     <- search_grid$interaction.depth[i]
  curr_shrinkage <- search_grid$shrinkage[i]
  
  
  # GBM Model with CV
  gbm_model <- gbm(
    formula = target ~ .,
    data = df_train_subset,     
    distribution = "gaussian",
    n.trees = 1000,               
    interaction.depth = curr_depth,
    shrinkage = curr_shrinkage,
    n.minobsinnode = 10,          
    bag.fraction = 0.5,           
    cv.folds = 5,                 
    n.cores = parallel::detectCores(),
    verbose = 1               
    
  )
  
  
  
  # Find optimal trees using CV error
  best_iter  <- gbm.perf(gbm_model, method = "cv", plot.it = FALSE)
  min_rmse   <- sqrt(min(gbm_model$cv.error))
  
  
  results_df <- rbind(results_df, data.frame(
    
    depth = curr_depth,
    shrinkage = curr_shrinkage,
    best_trees = best_iter,       
    rmse = min_rmse
    
  ))
  
}


best_run <- results_df[order(results_df$rmse), ][1, ]
print("--- Step 1 Result: Optimal Hyperparameters ---")
print(best_run) 


###############################################################################
# testing the model trained on the full training set and tested on the real hold out
# validation set


# initialise the parameters with the parameters found in the best model
model_for_val <- gbm(
  formula = target ~ .,
  data = df_train,     
  distribution = "gaussian",
  n.trees = best_run$best_trees,  
  interaction.depth = best_run$depth, 
  shrinkage = best_run$shrinkage,  
  n.minobsinnode = 10,             
  bag.fraction = 0.5,              
  cv.folds = 0,             
  n.cores = parallel::detectCores(),
  verbose = TRUE
)

# prediction using the hold out validation set
pred_val_log <- predict(model_for_val, newdata = df_val, n.trees = best_run$best_trees)

# taking the exponential (its a logged result)
pred_val_real <- exp(pred_val_log) - 1
y_val_real    <- exp(y_val) - 1

# RMSE calculation
val_rmse_real <- sqrt(mean((y_val_real - pred_val_real)^2))
cat("Validation RMSE:", val_rmse_real, "\n")


###############################################################################
# train model on train + val set and make predictions on test for a submission in the kaggle

# Prepare combined data
df_val_with_target <- df_val
df_val_with_target$target <- y_val
df_all <- rbind(df_train, df_val_with_target)

final_rounds <- round(best_run$best_trees)


final_model <- gbm(
  formula = target ~ .,
  data = df_all,                    # <--- TRAIN + VAL used here!
  distribution = "gaussian",
  n.trees = final_rounds,
  interaction.depth = best_run$depth,
  shrinkage = best_run$shrinkage,
  n.minobsinnode = 10,
  bag.fraction = 0.8,
  cv.folds = 0,
  n.cores = parallel::detectCores(),
  verbose = TRUE
)

# Prediction on TEST set
pred_kaggle_log <- predict(final_model, newdata = df_test, n.trees = final_rounds)
pred_kaggle <- exp(pred_kaggle_log) - 1

submission <- data.frame(index = as.integer(test_df$index), price = pred_kaggle)
write.csv(submission, "Data/Submissions/submission_gbm.csv", row.names = FALSE)
