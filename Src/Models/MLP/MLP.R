library(reticulate)
use_condaenv("r-tensorflow", required = TRUE)
library(tensorflow)
library(keras)
library(tidyverse)

# random seed
set.seed(123)

# Data loading
X_tr   <- readRDS("Data/Gold/features_TFIDF/X_tr_final.rds")
X_val  <- readRDS("Data/Gold/features_TFIDF/X_val_final.rds")
X_test <- readRDS("Data/Gold/features_TFIDF/X_test_final.rds")

y_tr   <- readRDS("Data/Gold/features_TFIDF/y_tr.rds")
y_val  <- readRDS("Data/Gold/features_TFIDF/y_val.rds")

# NN expects dense matrices
X_tr_dense  <- as.matrix(X_tr)
X_val_dense <- as.matrix(X_val)

# this will we use as input for the NN
input_dim <- ncol(X_tr_dense)

# function to convert the log results to normal 
rmse_price <- function(pred_log, true_log) {
  pred_price <- exp(pred_log)  - 1
  true_price <- exp(true_log)  - 1
  sqrt(mean((pred_price - true_price)^2))
}

###############################################################################
# Model definition 
# =============================

# a function to build up the initial model where parameters can be overwritten later on
build_model <- function(
    input_dim,
    n_layers   = 1,
    units1     = 192,
    units2     = 64,
    units3     = 64,
    activation = "relu",
    dropout1   = 0.3,
    dropout2   = 0.3,
    l2_reg     = 1e-4,
    lr         = 3e-3
) {
  # defining the different activation functions with fallback relu(but will never happen actually)
  act_fun <- if (activation %in% c("relu", "tanh", "gelu")) activation else "relu"
  
  # defining the sequential layers of the NN (with units, regularisation and activation)
  model <- keras_model_sequential()
  
  model <- model %>%
    layer_dense(
      units = units1, activation = act_fun,
      input_shape = input_dim,
      kernel_regularizer = regularizer_l2(l2_reg)
    )
  if (dropout1 > 0) {
    model <- model %>% layer_dropout(rate = dropout1)
  }
  
  if (n_layers >= 2) {
    model <- model %>%
      layer_dense(
        units = units2, activation = act_fun,
        kernel_regularizer = regularizer_l2(l2_reg)
      )
    if (dropout2 > 0) {
      model <- model %>% layer_dropout(rate = dropout2)
    }
  }
  
  if (n_layers >= 3) {
    model <- model %>%
      layer_dense(
        units = units3, activation = act_fun,
        kernel_regularizer = regularizer_l2(l2_reg)
      )
  }
  
  model <- model %>% layer_dense(units = 1)
  
  model %>% compile(
    loss      = "mse",
    optimizer = optimizer_adam(learning_rate = lr),
    metrics   = list("mean_squared_error")
  )
  
  model
}


###############################################################################
# defining our grid
# =============================

# an extensive grid 
grid <- expand.grid(
  n_layers   = c(1, 2, 3),
  units1     = c(128, 196,256),
  units2     = c(64, 96),
  units3     = c(16, 32, 64),
  activation = c("relu", "tanh", "gelu"),
  dropout1   = c(0.1, 0.2),
  dropout2   = c(0.0, 0.05, 0.20),
  l2_reg     = c(0.00010, 0.00030, 0.00003),
  lr         = c(0.003, 0.001),
  batch      = 256,
  stringsAsFactors = FALSE
)

# we will store a table of configs that we already tested and the best result in this table
# load previous tested configs if the file exists
if (file.exists("Src/Models/MLP/mlp_tested_configs.rds")) {
  tested_old <- readRDS("Src/Models/MLP/mlp_tested_configs.rds")
  tested_old_hparams <- tested_old %>% select(-rmse_val)
  beste_rmse <- min(tested_old$rmse_val)
} else {
  tested_old <- tibble()     # initialise it
  tested_old_hparams <- grid[0, , drop = FALSE]   # empty hparam grid
  beste_rmse <- Inf                               # so every new config is better
}

# remove already tested configs (so that we know which ones to still test)
remaining_grid <- dplyr::anti_join(
  grid,
  tested_old_hparams,
  by = colnames(tested_old_hparams)
)

# this will decide how much we will test 
nrow(remaining_grid)

set.seed(123)
if (nrow(remaining_grid) > 40) {
  remaining_grid <- remaining_grid[sample(nrow(remaining_grid), 10), ]
} else {
  remaining_grid <- remaining_grid[sample(nrow(remaining_grid)), ]  # random order
}

###############################################################################
# Hyperparametertuning (using a Gridsearch)
# =============================

# defining a list to store the results
results <- list()
best_rmse <- beste_rmse

# defining the best config that we have found unitl now
if (nrow(tested_old) > 0) {
  best_cfg <- tested_old_hparams[which.min(tested_old$rmse_val), ]
} else {
  best_cfg <- NULL
}

best_model <- NULL

# K-fold CV on train set only
# So the validation set is NOT used here to avoid data leakage.
K      <- 3
n_tr   <- nrow(X_tr_dense)
set.seed(123)
fold_id <- sample(rep(1:K, length.out = n_tr))


# proces to loop through all the configs and train/evaluate models
# each config gets a full 3-fold CV on train only
for (i in seq_len(nrow(remaining_grid))) {
  cat("\n===== Config", i, "van", nrow(remaining_grid), "=====\n")
  
  cfg <- remaining_grid[i, ]
  print(cfg)
  
  fold_rmses <- numeric(K)
  
  # 3-fold cross-validation on X_tr_dense / y_tr
  for (k in 1:K) {
    cat("  -> Fold", k, "van", K, "\n")
    
    # split TRAIN into train-fold and validation-fold (so inner CV)
    train_idx <- fold_id != k
    val_idx   <- fold_id == k
    
    k_clear_session()
    gc()
    
    # overwrite the parameters with our config parameters of the testing grid
    model <- build_model(
      input_dim  = input_dim,
      n_layers   = cfg$n_layers,
      units1     = cfg$units1,
      units2     = cfg$units2,
      units3     = cfg$units3,
      activation = cfg$activation,
      dropout1   = cfg$dropout1,
      dropout2   = cfg$dropout2,
      l2_reg     = cfg$l2_reg,
      lr         = cfg$lr
    )
    
    # defining early stopping rule
    cb_es <- callback_early_stopping(
      monitor = "val_loss",
      patience = 3,
      restore_best_weights = TRUE
    )
    
    # so here training happens on TRAIN-fold only
    # and the validation happens on the inner CV fold
    history <- model %>% fit( 
      x = X_tr_dense[train_idx, , drop = FALSE], # train
      y = y_tr[train_idx],
      epochs = 15,
      batch_size = cfg$batch,
      validation_data = list(
        X_tr_dense[val_idx, , drop = FALSE], #inner cv set
        y_tr[val_idx]
      ),
      callbacks = list(cb_es),
      verbose = 0
    )
    
    # validation prediction op de fold-val set
    pred_cv_log <- model %>% predict(
      X_tr_dense[val_idx, , drop = FALSE],   # use the training val set
      batch_size = cfg$batch
    )
    fold_rmses[k] <- rmse_price(as.numeric(pred_cv_log), y_tr[val_idx])
    cat("Fold", k, "RMSE:", fold_rmses[k], "\n") # logging in the proces
  }
  
  # Mean RMSE over the folds, this gives us the tuning score
  rmse_val <- mean(fold_rmses)
  cat("Config", i, ": mean 3-fold RMSE =", rmse_val, "\n")
  
  results[[i]] <- list(
    config   = cfg,
    rmse_val = rmse_val
  )
  
  # track best tuning config (based ONLY on inner CV RMSE)
  if (rmse_val < best_rmse) {
    best_rmse <- rmse_val
    best_model <- model
    best_cfg   <- cfg
    cat("new best config with RMSE =", best_rmse, "\n")
  }
}

# print at the end the best config with corresponding RMSE score
cat("\nBest RMSE:", best_rmse, "\n")
print(best_cfg)

dir.create("Src/Models/MLP", recursive = TRUE, showWarnings = FALSE)

# save best config
saveRDS(best_cfg, file = "Src/Models/MLP/best_cfg_mlp.rds")

# save all the configs that are already tested
tested_configs <- purrr::map_dfr(
  results,
  ~ dplyr::bind_cols(.x$config, tibble::tibble(rmse_val = .x$rmse_val))
)

tested_all <- bind_rows(tested_old, tested_configs)
saveRDS(tested_all, "Src/Models/MLP/mlp_tested_configs.rds")


###############################################################################
# retrain best model on TRAIN only
# and evaluate on the true hold-out VAL set
# =============================
# So here we use our real predefined val set

best_cfg <- readRDS("Src/Models/MLP/best_cfg_mlp.rds")


final_model <- build_model(
  input_dim  = input_dim,
  n_layers   = best_cfg$n_layers[[1]],   
  units1     = best_cfg$units1[[1]],     
  units2     = best_cfg$units2[[1]],     
  units3     = best_cfg$units3[[1]],     
  activation = best_cfg$activation[[1]],
  dropout1   = best_cfg$dropout1[[1]],
  dropout2   = best_cfg$dropout2[[1]],
  l2_reg     = best_cfg$l2_reg[[1]],
  lr         = best_cfg$lr[[1]]
)

cb_es2 <- callback_early_stopping(
  monitor              = "val_loss",
  patience             = 3,
  restore_best_weights = TRUE,
  verbose              = 1
)

# deifning a lr scheduler here so that the weight will be learned in a optimised way
cb_lr2 <- callback_reduce_lr_on_plateau(
  monitor   = "val_loss",
  factor    = 0.5,
  patience  = 2,
  min_lr    = 1e-5,
  verbose   = 1
)

history_valcheck <- final_model %>% fit(
  x = X_tr_dense,
  y = y_tr,
  epochs = 40,
  batch_size = best_cfg$batch[[1]],
  validation_split = 0.05,    # internal validation set
  callbacks = list(cb_es2, cb_lr2),
  verbose = 1
)

# # plotting history of last run
# png("Doc/Plots/mlp_history.png", width = 1200, height = 800)
# plot(history_valcheck)
# dev.off()

# true hold-out RMSE (outer validation set)
pred_val_log_mlp <- final_model %>% predict(X_val_dense)
rmse_val_mlp <- rmse_price(as.numeric(pred_val_log_mlp), y_val)

cat("\nHold-out RMSE for MLP on validation set:", rmse_val_mlp, "\n")

###############################################################################
# Training on train + val
# =============================

# Combine TRAIN + VAL
X_full      <- rbind(X_tr, X_val)
y_full_log  <- c(y_tr, y_val)

X_full_dense <- as.matrix(X_full)
X_test_dense <- as.matrix(X_test)

input_dim <- ncol(X_full_dense)

print(best_cfg)

# now we we will perform 5 runs of this final model.
# each time with a different seed. This is important and can change our result
# in a better way, because initialisations of the weights are very important and influence 
# the end result in a strong way.
n_runs <- 5
pred_test_log_list <- vector("list", n_runs)

for (s in seq_len(n_runs)) {
  cat("\n===== FINAL RUN", s, "of", n_runs, "=====\n")
  
  # each run with a different seed (other initalisations)
  set.seed(1000 + s)
  tensorflow::tf$random$set_seed(1000 + s)# set another random seed
  k_clear_session()  # clearsession
  gc()
  
  # final model build
  final_model <- build_model(
    input_dim  = input_dim,
    n_layers   = best_cfg$n_layers[[1]],   
    units1     = best_cfg$units1[[1]],     
    units2     = best_cfg$units2[[1]],     
    units3     = best_cfg$units3[[1]],     
    activation = best_cfg$activation[[1]],
    dropout1   = best_cfg$dropout1[[1]],
    dropout2   = best_cfg$dropout2[[1]],
    l2_reg     = best_cfg$l2_reg[[1]],
    lr         = best_cfg$lr[[1]]
  )
  
  cb_es_final <- callback_early_stopping(
    monitor              = "loss",
    patience             = 4,
    restore_best_weights = TRUE,
    verbose              = 1
  )
  
  cb_lr_final <- callback_reduce_lr_on_plateau(
    monitor   = "loss",
    factor    = 0.5,
    patience  = 2,
    min_lr    = 1e-5,
    verbose   = 1
  )
  
  history_final <- final_model %>% fit(
    x = X_full_dense,
    y = y_full_log,
    epochs = 30,
    batch_size = best_cfg$batch[[1]],       # no validation set here (loss of data)
    callbacks = list(cb_es_final, cb_lr_final),
    verbose = 1
  )
  
  # test predictions 
  pred_test_log_list[[s]] <- as.numeric(
    final_model %>% predict(X_test_dense)
  )
}

###############################################################################
# Test prediction
# =============================

# averagen of results
pred_test_log_mat  <- do.call(cbind, pred_test_log_list)
pred_test_log_mean <- rowMeans(pred_test_log_mat)

pred_test <- exp(pred_test_log_mean) - 1

# save log predictions for stacking approach later on
saveRDS(as.numeric(pred_val_log_mlp),  "Src/Models/MLP/pred_val_log_mlp.rds")
saveRDS(as.numeric(pred_test_log_mean),"Src/Models/MLP/pred_test_log_mlp.rds")

# making submission file
test_df <- read_csv("Data/Silver/test_clean_joined.csv")

submission <- tibble(
  index = test_df$index,
  price   = pred_test
)

write_csv(submission, "Data/Submissions/submission_MLP.csv")
