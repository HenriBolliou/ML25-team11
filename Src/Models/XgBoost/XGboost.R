library(tidyverse)
library(Matrix)
library(xgboost)
library(Ckmeans.1d.dp)
library(ggplot2)
library(SHAPforxgboost)
library(dplyr)
library(stringr)

rm(list = ls())
set.seed(123)

# Loading the data
X_tr   <- readRDS("Data/Gold/features_TFIDF/X_tr_final.rds")
X_val  <- readRDS("Data/Gold/features_TFIDF/X_val_final.rds")
X_test <- readRDS("Data/Gold/features_TFIDF/X_test_final.rds")

y_tr   <- readRDS("Data/Gold/features_TFIDF/y_tr.rds")      
y_val  <- readRDS("Data/Gold/features_TFIDF/y_val.rds")     

test_df <- read_csv("Data/Silver/test_clean_joined.csv")

# transform matrices to specific xgboost matrices (memory efficient)
dtrain <- xgb.DMatrix(data = X_tr, label = y_tr)
dval   <- xgb.DMatrix(data = X_val, label = y_val) 
dtest  <- xgb.DMatrix(data = X_test)

###############################################################################

# setting up the hyperparameter grid
# this grid is quite extensive but we will make it gradually smaller 
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
n_iter <- 1
cat("Total Grid size: ", nrow(full_grid_params))
cat("random size of our search: ", n_iter)

# make the random sample for our search out of the full grid
random_indices <- sample(nrow(full_grid_params), n_iter)
search_grid <- full_grid_params[random_indices, ]

results_df <- data.frame()

###############################################################################
# Hyperparametertuning 
# we loop through the defined search grid en build a xgboost model with each combination of these parameters
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
    nrounds = 5000,
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
pred_val_log

# taking the exponential (its a logged result)
pred_val_real <- exp(pred_val_log) - 1
# target was also logged
y_val_real    <- exp(y_val) - 1

# RMSE calculation
val_rmse_real <- sqrt(mean((y_val_real - pred_val_real)^2))
cat("score on validation set:", val_rmse_real, "\n")
# val score of 29.53

###############################################################################
# train model on train + val set and make predictions on test for a submission in the kaggle

# Combine TRAIN + VAL
X_all    <- rbind(X_tr, X_val)
y_all    <- c(y_tr, y_val)

dall     <- xgb.DMatrix(data = X_all,  label = y_all)

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
saveRDS(as.numeric(pred_val_log),  "Src/Models/Xgboost/pred_val_log_xgb.rds")
saveRDS(as.numeric(pred_test_log_mean), "Src/Models/Xgboost/pred_test_log_xgb.rds")

submission <- data.frame(index = as.integer(test_df$index), price = pred_kaggle)
write.csv(submission, "Data/Submissions/submission_xgboost.csv", row.names = FALSE)



# --- STAP 0: Bibliotheken laden ---
library(xgboost)
library(SHAPforxgboost)
library(data.table)
library(ggplot2)

# --- STAP 1: Sampling (5.000 rijen voor snelheid) ---
set.seed(42)
n_sample <- 5000

# Veiligheidscheck: als data kleiner is dan 5000, neem alles
real_n <- min(n_sample, nrow(X_tr))
sample_indices <- sample(seq_len(nrow(X_tr)), size = real_n)

# Selecteer de sample (blijft nog even sparse matrix)
X_sample <- X_tr[sample_indices, ]

# --- STAP 2: SHAP Waarden Berekenen ---
message("SHAP waarden berekenen... even geduld.")
shap_result <- shap.values(xgb_model = model_for_val, X_train = X_sample)
shap_score <- shap_result$shap_score

# Vervang eventuele NA's door 0
shap_score[is.na(shap_score)] <- 0

# --- STAP 3: Conversie (ZONDER filter) ---
message("Data voorbereiden en omzetten naar juiste formaat...")

# We filteren NIET meer, we pakken alles.
# We zetten wel om naar matrix -> dataframe omdat shap.prep niet van sparse matrices houdt.
# Dit kan even duren als je heel veel kolommen hebt.

shap_score_df <- as.data.frame(as.matrix(shap_score))
X_sample_df   <- as.data.frame(as.matrix(X_sample))

# --- STAP 4: Data Preppen voor de Plot ---
# We verplaatsen 'top_n' naar deze functie.
# Dit zorgt dat de data nu al gefilterd wordt tot de top 30.

shap_long <- shap.prep(
  shap_contrib = shap_score_df,
  X_train = X_sample_df,
  top_n = 10   # <--- HERE IS THE FIX
)

shap_long <- shap_long %>%
  mutate(variable = recode(variable,
                           "name_length"          = "Name Length",
                           "desc_word_count"      = "Description Length",
                           
                           # Shipping
                           "shipping_catseller"   = "Shipping: Paid by Seller",  # Gratis verzending voor koper
                           "shipping_catbuyer"    = "Shipping: Paid by Buyer",
                           
                           # Condition (Check even of 1 'Nieuw' is in jouw data, meestal wel)
                           "item_condition_id1"   = "Condition: New/Excellent",
                           "item_condition_id3"   = "Condition: Used",
                           
                           # Categories & Brand
                           "brand_nameunknown"    = "Brand: Unknown",
                           "categorie_2Shoes"      = "Sub-Cat: Shoes",
                           "categorie_1Women"     = "Main Cat: Women", # Let op: check of het 'categorie' of 'category' is in je data
                           
                           # Abstracte Text Features
                           "svd_text_4"           = "Text Pattern (SVD 4)"
  ))
# 1. Maak de plot (zonder top_n argument hier)
p_poster <- shap.plot.summary(shap_long) +
  
  # 1. TITELS & LABELS (Engels)
  ggtitle("Top 10 Drivers of Resale Price",
          subtitle = "How specific features impact the value") +
  labs(
    y = "Price Impact (Log Scale)",
    x = "",
    color = "Feature Value"
  ) +
  
  # 2. KLEUREN VAN DE STIPPEN
  scale_color_gradient(
    low = "#1E64C8",   # Jouw tekstkleur (zorgt voor mooie harmonie)
    high = "#FF0051",  # Fel Rood/Roze (voor hoog contrast bij 'High')
    name = "Value",
    labels = c("Low", "High"),
    breaks = c(0, 1),
    guide = guide_colorbar(
      barheight = unit(10, "lines"), 
      barwidth = unit(0.8, "lines"),
      ticks = FALSE
    )
  ) +
  
  # 3. THEMA MET JOUW KLEUREN
  theme_minimal(base_size = 18) +
  theme(
    # --- Achtergronden ---
    plot.background = element_rect(fill = "#9FC3F4", color = NA),
    panel.background = element_rect(fill = "#9FC3F4", color = NA),
    legend.background = element_rect(fill = "#9FC3F4", color = NA),
    
    # --- JOUW TEKSTKLEUR (#1E64C8) ---
    text = element_text(color = "#1E64C8"), # Basisregel voor alle tekst
    
    plot.title = element_text(face = "bold", size = 26, color = "#1E64C8"),
    plot.subtitle = element_text(size = 18, color = "#1E64C8", margin = margin(b = 15)),
    
    # Assen
    axis.text.y = element_text(face = "bold", size = 14, color = "#1E64C8"), 
    axis.text.x = element_text(size = 12, color = "#1E64C8"),
    axis.title.x = element_text(face = "bold", size = 16, color = "#1E64C8", margin = margin(t = 10)),
    
    # Legende
    legend.position = "right",
    legend.text = element_text(size = 12, color = "#1E64C8"),
    legend.title = element_text(face = "bold", size = 14, color = "#1E64C8"),
    
    # Grid (Wit blijft het mooiste en rustigste op deze achtergrond)
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_line(color = "white", size = 0.5, linetype = "dotted"),
    panel.grid.major.x = element_line(color = "white", size = 0.5)
  )

# 2. Toon de plot
print(p_poster)
