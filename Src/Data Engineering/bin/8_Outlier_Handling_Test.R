library(tidyverse)

# Load test features 
test_features <- read_csv("Data/Gold/test_features_encoded.csv")

# Identify numeric columns 
numeric_cols <- test_features %>%
  select(where(is.numeric)) %>%
  colnames()

# Load the stored parameters from the training outlier analysis
params <- readRDS("Data/Gold/outlier_params.rds")

mu  <- params$mu
sdv <- params$sdv
z2_threshold <- params$z2_threshold
feat_threshold <- params$feat_threshold

# ==========================================
# Compute test Z-scores 
# ==========================================
X_test <- test_features %>%
  select(all_of(numeric_cols)) %>%
  as.matrix()

Z_test <- sweep(X_test, 2, mu, "-")
Z_test <- sweep(Z_test, 2, sdv, "/")

# Compute Z²-score
z2_score_test <- rowSums(Z_test^2)

# Compute |z| > 3 counts
z_abs_gt3_count_test <- rowSums(abs(Z_test) > 3)

# ==========================================
# Outlier flags 
# ==========================================
z2_flag_test <- z2_score_test > z2_threshold
feat_flag_test <- z_abs_gt3_count_test >= feat_threshold

# Combine 
combined_flag_test <- z2_flag_test | feat_flag_test

cat("Test outliers detected: ", sum(combined_flag_test), "\n")

# ==========================================
# Remove them
# ==========================================
test_features_outlierfree <- test_features %>%
  filter(!combined_flag_test)

cat("Remaining test rows: ", nrow(test_features_outlierfree), "\n")

# Save final clean testset
write_csv(test_features_outlierfree, "Data/Gold/test_set.csv")
