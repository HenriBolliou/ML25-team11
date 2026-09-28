library(dplyr)
library(readr)
library(Matrix)
library(lightgbm)

rm(list = ls())
set.seed(1234) 

# Loading the data
X_tr   <- readRDS("Data/Gold/without_svd/X_tr.rds")
X_val  <- readRDS("Data/Gold/without_svd/X_val.rds")
X_test <- readRDS("Data/Gold/without_svd/X_test.rds")

# Targets
y_tr   <- readRDS("Data/Gold/without_svd/y_tr.rds")      
y_val  <- readRDS("Data/Gold/without_svd/y_val.rds")     

test_df <- read_csv("Data/Silver/test_clean_joined.csv")

# Cleaning feature names
colnames(X_tr)   <- make.names(colnames(X_tr), unique = TRUE)
colnames(X_val)  <- make.names(colnames(X_val), unique = TRUE)
colnames(X_test) <- make.names(colnames(X_test), unique = TRUE)

# transform matrices to specific lightboosting  (memory efficient)
cat("Preparing data...\n")
 
dtrain <- lgb.Dataset(data = X_tr, label = y_tr)
dval   <- lgb.Dataset(data = X_val, label = y_val) 

#grid set up based on literature 
#this is computationally extensive so we will do a random search on this grid search 
full_grid_params <- expand.grid(
  num_leaves = c(31, 63, 127),          
  learning_rate = c(0.01, 0.05, 0.1, 0.3),   
  max_depth = c(5, 7, 9, 11, 13),             
  min_sum_hessian_in_leaf = c(1 ,5,10),    
  min_gain_to_split = c(0, 0.1),         
  feature_fraction = c(0.6, 0.8, 1),          
  bagging_fraction = c(0.6 , 0.8, 1)
)
# Select amount of number of random search 
n_iter <- 3
cat(sprintf("Total grid : %d combinaties.\n", nrow(full_grid_params)))


# random rows from grid
random_indices <- sample(nrow(full_grid_params), n_iter)
search_grid <- full_grid_params[random_indices, ]

results_df <- data.frame()


###############################################################################
# Hyperparametertuning 
# we loop through the defined search grid en build a lightboost model with each combination of these parameters
# train it only on the training set!

# results_df has to be empty 
results_df <- data.frame()

for (i in 1:nrow(search_grid)) {
  
  # Parameters ophalen
  curr_lr     <- search_grid$learning_rate[i]
  curr_depth  <- search_grid$max_depth[i]
  curr_hess   <- search_grid$min_sum_hessian_in_leaf[i]
  curr_gain   <- search_grid$min_gain_to_split[i]
  curr_feat   <- search_grid$feature_fraction[i]
  curr_bag    <- search_grid$bagging_fraction[i]
  curr_leaves <- search_grid$num_leaves[i]
  
  params <- list(
    objective = "regression",
    metric = "rmse",
    boosting = "gbdt",
    num_threads = parallel::detectCores(),
    verbosity = -1,             
    feature_pre_filter = FALSE,
    
    learning_rate = curr_lr,
    max_depth = curr_depth,
    num_leaves = curr_leaves,
    min_sum_hessian_in_leaf = curr_hess,
    min_gain_to_split = curr_gain,
    feature_fraction = curr_feat,
    bagging_fraction = curr_bag,
    bagging_freq = 1,
    lambda_l1 = 0.5,
    lambda_l2 = 0.5
  )
  

  tryCatch({
    
    cv_results <- lgb.cv(
      params = params,
      data = dtrain,
      nfold = 5,
      nrounds = 5000,           
      early_stopping_rounds = 30,
      verbose = 1,             
      stratified = FALSE
    )
    
    # store the best results
    best_iter  <- cv_results$best_iter
    best_score <- cv_results$best_score
    
    if (is.null(best_score)) {
      best_score <- cv_results$record_evals[["test"]][["rmse"]][["eval"]][[best_iter]]
    }
    
    # Logging
    cat(sprintf("[%02d/%d] RMSE: %.4f | Lr:%.2f D:%d Hess:%d Leaves:%d\n", 
                i, nrow(search_grid), best_score, 
                curr_lr, curr_depth, curr_hess, curr_leaves))
    
    # save results to a dataframe
    row <- data.frame(
      iter = i,
      score = best_score,
      rounds = best_iter,
      lr = curr_lr,
      depth = curr_depth,
      leaves = curr_leaves,
      min_hess = curr_hess,
      min_gain = curr_gain,
      feat_frac = curr_feat,
      bag_frac = curr_bag
    )
    
    results_df <- rbind(results_df, row)
    
  }, error = function(e) {
    cat(sprintf("ERROR  iteration %d: %s\n", i, e$message))
  })
}


# at the end of the loop, save the best result founded 
results_df <- results_df[order(results_df$score), ]
best_run <- results_df[1, ]

cat("Best search result")
print(best_run)


dir.create("Src/Models/LightGBM", recursive = TRUE, showWarnings = FALSE)
saveRDS(best_run, file = "Src/Models/LightGBM/lightgbm_best_config.rds")


###############################################################################
# testing the model trained on the full training set and tested on the real hold out
# validation set

best_run <- readRDS("Src/Models/LightGBM/lightgbm_best_config.rds")

# initialise the parameters with the parameters found in the best model
val_params <- list(
  objective = "regression",
  metric = "rmse",
  boosting = "gbdt",
  num_threads = parallel::detectCores(),
  verbosity = -1,
  
  learning_rate = best_run$lr,
  max_depth = best_run$depth,
  num_leaves = best_run$leaves,
  min_sum_hessian_in_leaf = best_run$min_hess,
  min_gain_to_split = best_run$min_gain,
  feature_fraction = best_run$feat_frac,
  bagging_fraction = best_run$bag_frac,
  bagging_freq = 1,
  lambda_l1 = 0.5,
  lambda_l2 = 0.5
)

model_for_val <- lgb.train(
  params = val_params,
  data = dtrain,
  nrounds = best_run$rounds,
  verbose = 1
)

# prediction using the hold out validation set
pred_val_log <- predict(model_for_val, X_val)

# taking the exponential (its a logged result)
pred_val_real <- exp(pred_val_log) - 1
y_val_real    <- exp(y_val) - 1


# RMSE calculation
val_rmse_real <- sqrt(mean((y_val_real - pred_val_real)^2))
cat("score on validation set:", val_rmse_real, "\n")


###############################################################################
# train model on train + val set and make predictions on test for a submission in the kaggle
cat("\nRetrain op Train + Val en Submission maken...\n")

# combine x and y data
X_all <- rbind(X_tr, X_val)
y_all <- c(y_tr, y_val)
dall  <- lgb.Dataset(data = X_all, label = y_all)


final_rounds <- round(best_run$rounds)

final_model <- lgb.train(
  params = val_params,
  data = dall, 
  nrounds = final_rounds,
  verbose = 1
)

pred_kaggle_log <- predict(final_model, X_test)
pred_kaggle <- exp(pred_kaggle_log) - 1

submission <- data.frame(index = as.integer(test_df$index), price = pred_kaggle)
write.csv(submission, "Data/Submissions/submission_lightgbm.csv", row.names = FALSE)
cat("Submission saved.\n")