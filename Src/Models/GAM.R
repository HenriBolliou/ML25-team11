library(tidyverse)
library(gam)  

# Loading the data
X_tr   <- readRDS("Data/Gold/Features_TFIDF/X_tr_final.rds")
X_val  <- readRDS("Data/Gold/Features_TFIDF/X_val_final.rds")
X_test <- readRDS("Data/Gold/Features_TFIDF/X_test_final.rds")

y_tr   <- readRDS("Data/Gold/Features_TFIDF/y_tr.rds")    
y_val  <- readRDS("Data/Gold/Features_TFIDF/y_val.rds")   

test   <- read_csv("Data/Silver/test_clean_joined.csv")

# matrices
X_tr_mat   <- as.matrix(X_tr)
X_val_mat  <- as.matrix(X_val)
X_test_mat <- as.matrix(X_test)

# RMSE on scaled prices
rmse_price <- function(pred_log, true_log) {
  pred_price  <- exp(pred_log)  - 1
  true_price  <- exp(true_log)  - 1
  sqrt(mean((pred_price - true_price)^2))
}

set.seed(123)

# we need dataframes with y included for fitting the GAM
dat_tr <- as.data.frame(X_tr_mat)
dat_tr$y_log <- y_tr

dat_val <- as.data.frame(X_val_mat)
dat_val$y_log <- y_val

###############################################################################
# lets choose our non linear predictors

# defining some candidate variables to choose from
svd_cols <- grep("^svd_text_", colnames(dat_tr), value = TRUE)

candidate_vars <- c(
  "desc_word_count",
  "name_length",
  svd_cols
)

# function to examine which predictors are the most non linear
# however they will be all highly non linear

# fitting a linear model and a non linear one
results <- lapply(candidate_vars, function(v) {
  f_lin <- as.formula(paste("y_log ~", v))  #linear formula
  f_gam <- as.formula(paste("y_log ~ s(", v, ")", sep = "")) #gam formula
  
  m_lin <- lm(f_lin, data = dat_tr)   #linear fit
  m_gam <- gam(f_gam, data = dat_tr) #gam fit
  
  # dataframe with our results
  data.frame(     
    var        = v,
    AIC_linear = AIC(m_lin),  
    AIC_gam    = AIC(m_gam),
    dAIC       = AIC(m_lin) - AIC(m_gam)  # the bigger the more its benefit
  )
})

# take a look to the results (top 10 most non linear)
results_df <- do.call(rbind, results)
results_df[order(-results_df$dAIC), ][1:10, ]


################################################################################
# GAM-model

# these predictors are chosen from the AIC evaluation above
smooth_vars <- c("svd_text_2", "svd_text_6", "svd_text_24","svd_text_5", "desc_word_count")

# because we only need to know the df of the smoothing variables, so no need to include ALL the other predictors
#this will decreas computation time and the result would not differ too much
linear_keep <- c(
  smooth_vars,          
  "name_length",        
  "item_condition_id",
  "shipping"
)
tune_vars <- c(linear_keep, "y_log")
# keep only columns that actually exist in dat_tr
tune_vars <- intersect(tune_vars, colnames(dat_tr))
# linear_keep ook alignen met de effectieve kolommen (y_log eruit)
linear_keep <- setdiff(tune_vars, "y_log")
# all other predictors will enter linearly
all_x_vars  <- setdiff(colnames(dat_tr), "y_log")


# range of df
df_range <- 2:8
# starting position where each smooth predictor is given 6 as df
best_dfs <- setNames(rep(6L, length(smooth_vars)), smooth_vars)
# list to store the results
tuning_results_list <- vector("list", length(smooth_vars))


cat("Start of hyperparametertuning\n")
# reduce only columns, keep ALL rows
dat_tr_small <- dat_tr[, tune_vars, drop = FALSE]

# K-fold CV setup 
set.seed(123)
K       <- 3
n_tr    <- nrow(dat_tr_small)
fold_id <- sample(rep(1:K, length.out = n_tr))

# for each predictor we evaluate every df in the range (Hyperparameter tuning)
for (i in seq_along(smooth_vars)) {
  pred <- smooth_vars[i]
  cat("tuning", pred, "...\n")
  
  best_rmse <- Inf
  best_df   <- NA_integer_
  
  # now loop through the range of df values for this predictor
  for (df_val in df_range) {
    
    # smoothing splines, so current predictors receives df_val and the other receive their best_df
    smooth_terms <- sprintf(
      "s(%s, %d)",
      smooth_vars,
      ifelse(smooth_vars == pred, df_val, best_dfs[smooth_vars])
    )
    smooth_part <- paste(smooth_terms, collapse = " + ")
    
    # formula to fit the GAM with only our subset of columns
    rhs <- paste(c(setdiff(linear_keep, pred), smooth_part), collapse = " + ")
    # fitting the gam
    gam_formula <- as.formula(paste("y_log ~", rhs))
    
    # K-fold CV 
    fold_rmses <- numeric(K)
    for (k in 1:K) {
      train_idx <- fold_id != k   #making the split
      val_idx   <- fold_id == k
      
      fit_k  <- gam(gam_formula, data = dat_tr_small[train_idx, ])  # fit the model of fold
      pred_k <- predict(fit_k, newdata = dat_tr_small[val_idx, ])  # prediction of fold on val set
      fold_rmses[k] <- rmse_price(pred_k, dat_tr_small$y_log[val_idx]) # rmse for particular fold
    }
    
    cv_rmse <- mean(fold_rmses) #take the mean of all the folds rmse
    
    if (cv_rmse < best_rmse) {
      best_rmse <- cv_rmse
      best_df   <- df_val
    }
  }
  
  # update for this smooth predictor
  best_dfs[pred] <- best_df
  
  # fit best model on full train to get AIC
  smooth_terms_best <- sprintf(
    "s(%s, %d)",
    smooth_vars,
    best_dfs[smooth_vars]
  )
  smooth_part_best <- paste(smooth_terms_best, collapse = " + ")
  
  # so now fit the whole model with finetuned dfs on the small training set
  rhs_best <- paste(c(setdiff(linear_keep, pred), smooth_part_best), collapse = " + ")
  gam_formula_best <- as.formula(paste("y_log ~", rhs_best))
  
  fit_best <- gam(gam_formula_best, data = dat_tr_small)
  
  # print the results table of the hyperparametertuning  (likewise to the table in our report)
  tuning_results_list[[i]] <- data.frame(
    Predictor   = pred,
    Range_df    = sprintf("[%d, %d]", min(df_range), max(df_range)),
    Selected_df = best_df,
    CV_RMSE     = best_rmse
  )
}

tuning_results <- bind_rows(tuning_results_list)
print(tuning_results)


################################################################################
# Final formula with all linear vars + tuned splines

smooth_terms_final <- paste0(
  "s(", smooth_vars, ", ", best_dfs[smooth_vars], ")",
  collapse = " + "
)
# again formula
rhs_final <- paste(c(".", smooth_terms_final), collapse = " + ")
final_gam_formula <- as.formula(paste("y_log ~", rhs_final))

# Final fit on train and evaluation on hold-out validation set
gam_fit_train <- gam(final_gam_formula, data = dat_tr)

# calculate the rmse on validation y
val_pred_log <- predict(gam_fit_train, newdata = dat_val)
val_rmse     <- rmse_price(val_pred_log, y_val)
cat("\nHold-out RMSE on validation set:", val_rmse, "\n")

# plotting the smoothing splines
par(mfrow = c(2, 3), mar = c(4, 4, 2, 1) + 0.1) 
plot(
    gam_fit_train,
    se = TRUE,
    col = "darkgreen",
    lty = 1,
    lwd = 2,
    rug = TRUE
  )
par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1)

################################################################################
# Final fit on train + validation
X_all_mat <- rbind(X_tr_mat, X_val_mat)
y_all     <- c(y_tr, y_val)
dat_all   <- as.data.frame(X_all_mat)
dat_all$y_log <- y_all

gam_fit_tuned <- gam(final_gam_formula, data = dat_all)

################################################################################
# predictions on test set
X_test_df <- as.data.frame(X_test_mat)

test_pred_log   <- predict(gam_fit_tuned, newdata = X_test_df)
test_pred_price <- exp(test_pred_log) - 1

# making submission file
test_df <- read_csv("Data/Silver/test_clean_joined.csv")

submission <- tibble(
  index = test_df$index,
  price = test_pred_price
)

write_csv(submission, "Data/Submissions/submission_GAM.csv")
