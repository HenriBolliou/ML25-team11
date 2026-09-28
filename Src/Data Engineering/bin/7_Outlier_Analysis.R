library(tidyverse)
library(pacman)
p_load(dbscan)

# Read in the data
train_features <- read_csv("Data/Gold/train_features_encoded.csv")

# Check
dim(train_features)
str(train_features)

# Only numeric features needed, no price included
numeric_cols <- train_features %>%
  select(where(is.numeric)) %>%
  colnames()

numeric_cols <- setdiff(numeric_cols, "price")  # exclude target variable

X <- train_features %>%
  select(all_of(numeric_cols)) %>%
  as.matrix()

# =======================================================
#  OUTLIERS VIA Z²-SCORE
# =======================================================

# calculate Z-scores 
mu  <- colMeans(X)
sdv <- apply(X, 2, sd)
sdv[sdv == 0] <- 1  # no division by 0 possible

Z <- sweep(X, 2, mu, "-")
Z <- sweep(Z, 2, sdv, "/")

# Z²-score per row
z2_score <- rowSums(Z^2)

# Outlier Threshold via quantile
z2_threshold <- quantile(z2_score, 0.995)  
z2_outlier_flag <- z2_score > z2_threshold

# features with |z| > 3
z_abs_gt3_count <- rowSums(abs(Z) > 3)

# define feature-outlier flag, making use of a quantile threshold
feat_threshold <- quantile(z_abs_gt3_count, 0.99)
feat_threshold 

z_feature_outlier_flag <- z_abs_gt3_count >= feat_threshold

# =======================================================
# OUTLIERS VIA DBSCAN
# =======================================================

set.seed(123)

# scale
X_scaled <- scale(X)

# Subsample (performance), from +/- 100.000 rows is DBSCAN very slow
n_total   <- nrow(X_scaled)
sample_n  <- min(50000, n_total) # good enough to be representative and good for calculations  
sample_ix <- sample(seq_len(n_total), size = sample_n)

X_sample <- X_scaled[sample_ix, , drop = FALSE]

# 2 parameters in DBSCAN (epsilon and MinPoints)
# Kies minPts (rule of thumb: ~ dimension * 2 to 4)
minPts <- 4 * ncol(X_sample)  
# kNN-distances to estimate eps 
knn_dist <- kNNdist(X_sample, k = minPts)
# knn_dist is a vector of distances to the k^the neighbour
# take 95e percentiel of kNN-distance
eps <- quantile(knn_dist, 0.95)    

cat("Chosen eps:", eps, "\n")
cat("MinPts:", minPts, "\n")

# DBSCAN 
db <- dbscan(X_sample, eps = eps, minPts = minPts)

# cluster == 0 is 'noise' -> outlier
dbscan_outlier_sample <- ifelse(db$cluster == 0, 1L, 0L)

# Projecteer terug naar volledige dataset (sample heeft label, rest NA)
dbscan_outlier_flag <- rep(NA_integer_, n_total)
dbscan_outlier_flag[sample_ix] <- dbscan_outlier_sample


# =======================================================
# MERGE RESULTS
# =======================================================

train_outliers <- train_features %>%
  mutate(
    z2_score                = z2_score,
    z2_outlier_flag         = z2_outlier_flag,
    z_abs_gt3_count         = z_abs_gt3_count,
    z_feature_outlier_flag  = z_feature_outlier_flag,
    dbscan_outlier_flag     = dbscan_outlier_flag,
    combined_outlier        = z2_outlier_flag |
      z_feature_outlier_flag |
      coalesce(dbscan_outlier_flag == 1L, FALSE)
  )


# Check conclusions of different methods
cat("Z² outliers: ", sum(train_outliers$z2_outlier_flag), "\n")
cat("Feature-level z outliers: ", sum(train_outliers$z_feature_outlier_flag), "\n")
cat("DBSCAN outliers (in sample): ", sum(train_outliers$dbscan_outlier_flag == 1, na.rm = TRUE), "\n")
cat("Combined outliers: ", sum(train_outliers$combined_outlier), "\n\n")


# Histogram Z²-score
p <- ggplot(train_outliers, aes(z2_score)) +
  geom_histogram(bins = 200, fill = "steelblue", alpha = 0.7) +
  geom_vline(xintercept = z2_threshold, color = "red", linewidth = 1.2) +
  scale_x_continuous(limits = c(0, quantile(train_outliers$z2_score, 0.999))) +
  theme_minimal() +
  ggtitle("Z²-score distribution with threshold")

ggsave(
  filename = "Z²-score.png",
  plot = p,
  path = "Doc/Plots",
  width = 8,
  height = 5,
  dpi = 300
)

# Percentage outliers
pct <- mean(train_outliers$combined_outlier) * 100
cat("\nPercentage combined outliers: ", round(pct, 2), "%\n\n")

# Impact on dataset
remaining <- sum(!train_outliers$combined_outlier)
cat("Remaining after removing combined outliers: ", remaining, "\n")
cat("Original rows: ", nrow(train_outliers), "\n")

# =======================================================
#  Remove Outliers from trainingset
# =======================================================

# drop outliers 
train_features_outlierfree <- train_outliers %>%
  filter(!combined_outlier) %>%              # keep only non-outliers
  select(                                    # drop de outlier flag columns
    -z2_score,
    -z2_outlier_flag,
    -z_abs_gt3_count,
    -z_feature_outlier_flag,
    -dbscan_outlier_flag,
    -combined_outlier
  )

# Safety check
dim(train_features_outlierfree)
colnames(train_features_outlierfree)[1:10]


if (!dir.exists("Data/Gold")) {
  dir.create("Data/Gold", recursive = TRUE, showWarnings = FALSE)
} else {
  cat(paste("Gold", "Data/Gold", "does already exists.\n"))
}
write_csv(train_features_outlierfree, "Data/Gold/train.csv")

# save parameters for testset
saveRDS(list(
  mu = mu,
  sdv = sdv,
  z2_threshold = z2_threshold,
  feat_threshold = feat_threshold
), file = "Data/Gold/outlier_params.rds")








