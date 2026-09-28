library(tidyverse)
library(Matrix)
library(xgboost)

rm(list = ls())
set.seed(123)
# Loading the data
X_tr   <- readRDS("Data/Gold/without_svd/X_tr.rds")
X_val  <- readRDS("Data/Gold/without_svd/X_val.rds")
X_test <- readRDS("Data/Gold/without_svd/X_test.rds")

y_tr   <- readRDS("Data/Gold/without_svd/y_tr.rds")      
y_val  <- readRDS("Data/Gold/without_svd/y_val.rds")     

test_df <- read_csv("Data/Silver/test_clean_joined.csv")

# we set some weights to the higher priced items to minimize the errors there
price_tr <- exp(y_tr) - 1                      
q80 <- quantile(price_tr, 0.8)                 
w_tr <- ifelse(price_tr > q80, 3, 1)

# transform matrices to specific xgboost matrices (memory efficient)
dtrain <- xgb.DMatrix(data = X_tr, label = y_tr, weight = w_tr)
dval   <- xgb.DMatrix(data = X_val, label = y_val)   
dtest  <- xgb.DMatrix(data = X_test)

###############################################################################

# setting up the hyperparameter grid
# this grid is quite extensive but we will mak it gradually smaller 
# based upon the results of our runs

full_grid_params <- expand.grid(
  eta = c(0.01, 0.05, 0.1, 0.3),              
  max_depth = c(5, 7, 9, 11, 13),             
  min_child_weight = c(1, 5, 10),             
  gamma = c(0, 1, 3),                         
  colsample_bytree = c(0.6, 0.8, 1),          
  subsample = c(0.8, 1)                       
)

# number of iterations of training the model with different values for out parameters
n_iter <- 20
cat("Total Grid size: ", nrow(full_grid_params))
cat("random size of our search: ", n_iter)

# make the random sample (40) for our search out of the full grid
random_indices <- sample(nrow(full_grid_params), n_iter)
search_grid <- full_grid_params[random_indices, ]

results_df <- data.frame()


###############################################################################
# Hyperparametertuning 
# we loop through the defined search grid en build a xgboost model with each combination of thes parameters
# train it only on the training set!

for (i in 1:nrow(search_grid)) {
  
  # defining the parameters list based upon the search grid
  curr_eta       <- search_grid$eta[i]
  curr_depth     <- search_grid$max_depth[i]
  curr_child     <- search_grid$min_child_weight[i]
  curr_gamma     <- search_grid$gamma[i]
  curr_colsample <- search_grid$colsample_bytree[i]
  curr_subsample <- search_grid$subsample[i]
  
  params <- list(
    booster = "gbtree",
    objective = "reg:squarederror", 
    eval_metric = "rmse",
    nthread = parallel::detectCores(),
    tree_method = "hist",
    eta = curr_eta,
    max_depth = curr_depth,
    min_child_weight = curr_child,
    gamma = curr_gamma,
    colsample_bytree = curr_colsample,
    subsample = curr_subsample,
    alpha = 0.5,  
    lambda = 0.5  
  )
  
  # run xgb with a 5 fold cross validation 
  cv_results <- xgb.cv(
    params = params,
    data = dtrain,
    nfold = 5,
    nrounds = best_run$rounds,
    early_stopping_rounds = 30,
    print_every_n = 50, 
    verbose = 1, 
    maximize = FALSE
  )
  
  # store the best results
  best_iter  <- cv_results$best_iteration
  best_score <- cv_results$evaluation_log$test_rmse_mean[best_iter]
  
  # print the results (logging while tuning)
  cat(sprintf("[%02d/%d] Log-RMSE: %.4f | Eta:%.2f D:%d Child:%d Gam:%d Col:%.1f Sub:%.1f\n", 
              i, n_iter, best_score, 
              curr_eta, curr_depth, curr_child, curr_gamma, curr_colsample, curr_subsample))
  # save results to a dataframe
  row <- data.frame(
    eta = curr_eta,
    max_depth = curr_depth,
    min_child_weight = curr_child,
    gamma = curr_gamma,
    colsample_bytree = curr_colsample,
    subsample = curr_subsample,
    score = best_score,
    rounds = best_iter
  )
  results_df <- rbind(results_df, row)
}

# at the end of the loop, save the best result founded 
results_df <- results_df[order(results_df$score), ]
best_run <- results_df[1, ]

dir.create("Src/Models/Xgboost", recursive = TRUE, showWarnings = FALSE)

saveRDS(results_df, "Src/Models/Xgboost/xgb_tested_configs.rds")
saveRDS(best_run,  "Src/Models/Xgboost/xgb_best_config.rds")

cat("Best result:")
print(best_run)

###############################################################################
# testing the model trained on the full training set and tested on the real hold out
# validation set

best_run <- read_rds("Src/Models/Xgboost/xgb_best_config.rds")
# initialise the parameters with the parameters found in the best model
val_params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eval_metric = "rmse",
  nthread = parallel::detectCores(),
  tree_method = "hist",   
  
  eta = best_run$eta,
  max_depth = best_run$max_depth,
  min_child_weight = best_run$min_child_weight,
  gamma = best_run$gamma,
  colsample_bytree = best_run$colsample_bytree,
  subsample = best_run$subsample,
  alpha = 0.5,
  lambda = 0.5
)

watchlist <- list(train = dtrain)
# xgboost model build
model_for_val <- xgb.train(
  params = val_params,
  data = dtrain,
  nrounds = best_run$rounds,
  watchlist = watchlist,   
  print_every_n = 50,      
  verbose = 1
)

# prediction using the hold out validation set
pred_val_log <- predict(model_for_val, dval)

# taking the exponential (its a logged result)
pred_val_real <- exp(pred_val_log) - 1
# target was also logged
y_val_real    <- exp(y_val) - 1

# RMSE calculation
val_rmse_real <- sqrt(mean((y_val_real - pred_val_real)^2))
cat("score on validation set:", val_rmse_real, "\n")
# val score of 27.65

###############################################################################
# train model on train + val set and make predictions on test for a submission in the kaggle

# Combine TRAIN + VAL
X_all    <- rbind(X_tr, X_val)
y_all    <- c(y_tr, y_val)

# we set some weights to the higher priced items to minimize the errors there
price_all <- exp(y_all) - 1                      
q80 <- quantile(price_all, 0.8)                 
w_all <- ifelse(price_all > q80, 3, 1)


dall <- xgb.DMatrix(data = X_all, label = y_all, weight = w_all)
dtest <- xgb.DMatrix(data = X_test)

# now we we will perform 5 runs of this final model.
# each time with a different seed. This is important and can change our result
# in a better way.

n_runs_test <- 5
pred_test_log_list <- vector("list", n_runs_test)

for (s in seq_len(n_runs_test)) {
  cat("\n===== FINAL-RUN", s, "of", n_runs_test, "=====\n")
  
  set.seed(4000 + s)  # different seed
  
  final_model_s <- xgb.train(
    params   = val_params,
    data     = dall,
    nrounds  = best_run$rounds,
    print_every_n = 50,
    verbose  = 1
  )
  
  pred_test_log_list[[s]] <- predict(final_model_s, dtest)
}

# mean prediction over the 5 runs
pred_test_log_mat  <- do.call(cbind, pred_test_log_list)
pred_test_log_mean <- rowMeans(pred_test_log_mat)

# exponential value
pred_kaggle <- exp(pred_test_log_mean) - 1

# save log predictions for stacking approach later on
saveRDS(as.numeric(pred_val_log),  "Src/Models/xgboost2/pred_val_log_xgb2.rds")
saveRDS(as.numeric(pred_test_log_mean), "Src/Models/xgboost2/pred_test_log_xgb2.rds")

submission <- data.frame(index = as.integer(test_df$index), price = pred_kaggle)
write.csv(submission, "Data/Submissions/submission_xgboost.csv", row.names = FALSE)

