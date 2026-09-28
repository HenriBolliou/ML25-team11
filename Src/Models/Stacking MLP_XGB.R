library(tidyverse)
library(ggplot2)
library(mgcv)

# Loading the data (the log predictions on val set and test set)
y_val <- readRDS("Data/Gold/features_TFIDF/y_val.rds")

pred_val_log_mlp  <- readRDS("Src/Models/MLP2/pred_val_log_mlp.rds")
pred_val_log_xgb  <- readRDS("Src/Models/Xgboost2/pred_val_log_xgb.rds")

pred_test_log_mlp <- readRDS("Src/Models/MLP2/pred_test_log_mlp.rds")
pred_test_log_xgb <- readRDS("Src/Models/Xgboost2/pred_test_log_xgb.rds")


rmse_price <- function(pred_log, true_log) {
  pred_price <- exp(pred_log) - 1
  true_price <- exp(true_log) - 1
  sqrt(mean((pred_price - true_price)^2))
}

# a table where we add the predictors and target together (normal prices here)
stack_train <- tibble(
  mlp = exp(pred_val_log_mlp) - 1,
  xgb = exp(pred_val_log_xgb) - 1,
  y   = exp(y_val) - 1
)

# build a simple gam model (internal tuning through mgcv package)
meta_model_gam <- gam(y ~ s(mlp) + s(xgb), data = stack_train)
summary(meta_model_gam) # the 2 smoothing splines are highly significant

pred_val_stack_gam <- predict(meta_model_gam, newdata = stack_train) # predict
rmse_stack_val <- sqrt(mean((pred_val_stack_gam - (exp(y_val)-1))^2))
rmse_mlp_val   <- rmse_price(pred_val_log_mlp, y_val)
rmse_xgb_val   <- rmse_price(pred_val_log_xgb, y_val)


cat("\nRMSE VAL (MLP):      ", rmse_mlp_val)  # val rmse of mlp
cat("\nRMSE VAL (XGB):      ", rmse_xgb_val)   # val rmse of xgboost
cat("\nRMSE VAL (STACKED):  ", rmse_stack_val, "\n")  # val rmse of stack


# final predictions on test set
stack_test <- tibble(
  mlp = exp(pred_test_log_mlp) - 1,
  xgb = exp(pred_test_log_xgb) - 1, 
)

pred_test_stack <- predict(meta_model_gam, newdata = stack_test)
pred_test_stack <- as.numeric(pred_test_stack) # we have to flat the vector for our submission

# Submission 
test_df <- read_csv("Data/Silver/test_clean_joined.csv")

submission_stack <- tibble(
  index = test_df$index,
  price = pred_test_stack
)

write_csv(submission_stack, "Data/Submissions/submission_STACK_MLP_XGB.csv")



################################################################################
# EVALUATION
################################################################################

dir.create("Doc/Plots/evaluation", showWarnings = FALSE)

################################################################################
# MLP vs Target
p1 <- ggplot(stack_train, aes(x = y, y = mlp)) +
  geom_point(alpha = 0.4, color = "#4C72B0") +
  geom_smooth(method = "lm", color = "darkblue") +
  geom_abline(intercept = 0, slope = 1,
              linetype = "dashed", color = "#C44E52", linewidth = 1) +
  labs(
    title = "MLP predictions versus true prices",
    x = "True price",
    y = "MLP prediction (price)"
  ) +
  theme_minimal()

# XGB vs Target
p2 <- ggplot(stack_train, aes(x = y, y = xgb)) +
  geom_point(alpha = 0.4, color = "#55A868") +
  geom_smooth(method = "lm", color = "darkblue") +
  geom_abline(intercept = 0, slope = 1,
              linetype = "dashed", color = "#C44E52", linewidth = 1) +
  labs(
    title = "XGBoost predictions versus true prices",
    x = "True price",
    y = "XGBoost prediction (price)"
  ) +
  theme_minimal()

ggsave("Doc/Plots/evaluation/MLP_vs_target.png", p1, width = 8, height = 6, dpi = 300)
ggsave("Doc/Plots/evaluation/XGB_vs_target.png", p2, width = 8, height = 6, dpi = 300)

################################################################################
# GAM-stacked predictions vs true price

regr_results <- ggplot(stack_train, aes(x = y, y = pred_val_stack_gam)) +
  geom_point(alpha = 0.3,  color = "#8172B2" ) +
  geom_smooth(method = "lm", color = "darkblue") +
  geom_abline(intercept = 0, slope = 1,
              color = "#C44E52", linetype = "dashed", linewidth = 1) +
  labs(
    title = "Stack model predictions versus true prices",
    x = "True price",
    y = "GAM-stacked prediction (price)"
  ) +
  theme_minimal()

ggsave("Doc/Plots/evaluation/regression_results_GAM_stack.png",
       regr_results, width = 8, height = 6, dpi = 300)

################################################################################
# Residual plots 


stack_train <- stack_train %>%
  mutate(
    stack_pred_gam = as.numeric(pred_val_stack_gam),
    res_mlp   = y - mlp,
    res_xgb   = y - xgb,
    res_stack = y - stack_pred_gam
  )

# MLP: residual vs fitted
p_res_mlp <- ggplot(stack_train, aes(x = mlp, y = res_mlp)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_point(alpha = 0.3) +
  labs(
    title = "Residuals versus fitted values – MLP",
    x = "MLP prediction (price)",
    y = "Residual (true price − prediction)"
  ) +
  theme_minimal()

# XGB: residual vs fitted
p_res_xgb <- ggplot(stack_train, aes(x = xgb, y = res_xgb)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_point(alpha = 0.3) +
  labs(
    title = "Residuals versus fitted values – XGBoost",
    x = "XGBoost prediction (price)",
    y = "Residual (true price − prediction)"
  ) +
  theme_minimal()

# GAM-stack: residual vs fitted
p_res_stack <- ggplot(stack_train, aes(x = stack_pred_gam, y = res_stack)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  geom_point(alpha = 0.3) +
  labs(
    title = "Residuals versus fitted values – GAM-stacked model",
    x = "GAM-stacked prediction (price)",
    y = "Residual (true price − prediction)"
  ) +
  theme_minimal()

ggsave("Doc/Plots/evaluation/MLP_residuals_vs_fitted.png",
       p_res_mlp, width = 8, height = 6, dpi = 300)
ggsave("Doc/Plots/evaluation/XGB_residuals_vs_fitted.png",
       p_res_xgb, width = 8, height = 6, dpi = 300)
ggsave("Doc/Plots/evaluation/STACK_residuals_vs_fitted.png",
       p_res_stack, width = 8, height = 6, dpi = 300)
