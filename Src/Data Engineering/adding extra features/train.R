library(tidyverse)
library(text2vec)
library(Matrix)
library(stringi)
library(ggplot2)
library(irlba)
library(viridisLite)
library(SnowballC)
library(sentimentr)


set.seed(123)

train <- read_csv("Data/Silver/train_clean_joined.csv")

colSums(is.na(train))

###############################################################################

# First of all we take a look to the distribution of the description lengths
desc_len <- train %>%
  mutate(
    desc_len = if_else(
      is.na(item_description) | item_description == "",
      0L,
      str_count(item_description, "\\S+")
    )
  ) %>%
  pull(desc_len)
summary(desc_len)
quantile(desc_len, probs = c(0.8, 0.9, 0.95, 0.99))
# So we can truncate our descriptions to max 90 words (95th percentile)
max_words <- 90
# function to truncate text to max_words
truncate_text <- function(txt, max_words = 90) {
  if (is.na(txt) || txt == "") return("")
  words <- stringr::str_split(txt, "\\s+", simplify = FALSE)[[1]]
  paste(head(words, max_words), collapse = " ")
}

# boxplot of description lengths
desc_df <- tibble(desc_len = desc_len) %>%
  filter(desc_len > 0)

p <- ggplot(desc_df, aes(y = desc_len)) +
  geom_boxplot(
    fill= 'steelblue',
    color = "black",
    linewidth = 0.5,
    width = 0.3
  ) +
  labs(
    title = "Boxplot of Description Lengths",
    y = "Number of Words"
  ) +
  theme_minimal(base_size = 12)
# ggsave(
#   filename = "Boxplot of Description Lengths.png",
#   plot = p,
#   path = "Doc/Plots",
#   width = 8,
#   height = 5,
#   dpi = 300
# )


# there are a lot of brandnames so, filtering is needed.
# keep frequent brandnames en store the less frequent ones in a category 'other'
brand_counts <- train %>%
  count(brand_name, sort = TRUE)
summary(brand_counts$n)
quantile(brand_counts$n, probs = c(0.5, 0.75, 0.90, 0.95, 0.99))
# how many brands do we keep at different threholds
thresholds <- c(5, 10, 20, 50,75, 100, 200, 500)

remaining <- tibble(
  threshold = thresholds,
  brands_left = map_int(thresholds, ~ sum(brand_counts$n >= .x))
)

remaining

p2 <- ggplot(brand_counts, aes(x = n)) +
  geom_histogram(bins = 100, fill = 'steelblue', color = "black") +
  scale_x_log10() +
  labs(
    title = "Distribution of brand frequencies",
    x = "Brand frequency (log-scale)",
    y = "Count of brands"
  ) +
  theme_minimal()
# ggsave(
#   filename = "Distribution of brand frequencies.png",
#   plot = p2,
#   path = "Doc/Plots",
#   width = 8,
#   height = 5,
#   dpi = 300
# )

# decided to set the threshold on 50 (+/- 3000 -> 721)
brand_freq <- table(train$brand_name)
popular_brands <- names(brand_freq[brand_freq >= 50])
train$brand_name <- ifelse(train$brand_name %in% popular_brands,
                           train$brand_name,
                           "other_brand")


# there are also a lot of categories so, filtering is again needed.
length(unique(train$categorie_1))
length(unique(train$categorie_2))
length(unique(train$categorie_3))
# how is the distribution in these 3 categories
cat1_counts <- train %>% count(categorie_1, sort = TRUE)
cat2_counts <- train %>% count(categorie_2, sort = TRUE)
cat3_counts <- train %>% count(categorie_3, sort = TRUE)
summary(cat1_counts$n)
quantile(cat1_counts$n, probs = c(0.5, 0.75, 0.90, 0.95, 0.99))

summary(cat2_counts$n)
quantile(cat2_counts$n, probs = c(0.5, 0.75, 0.90, 0.95, 0.99))

summary(cat3_counts$n)
quantile(cat3_counts$n, probs = c(0.5, 0.75, 0.90, 0.95, 0.99))

# how many categories do we keep at different thresholds
thresholds <- c(2,5, 10, 20, 50, 100, 200)

remaining_cat1 <- tibble(
  threshold = thresholds,
  categories_left = map_int(thresholds, ~ sum(cat1_counts$n >= .x))
)
remaining_cat1

remaining_cat2 <- tibble(
  threshold = thresholds,
  categories_left = map_int(thresholds, ~ sum(cat2_counts$n >= .x))
)
remaining_cat2

remaining_cat3 <- tibble(
  threshold = thresholds,
  categories_left = map_int(thresholds, ~ sum(cat3_counts$n >= .x))
)
remaining_cat3


# category 1 (no filtering) (keep 51 categories)
# category 2 threshold = 4 (small filtering) (keep +/- 200)
cat2_freq <- table(train$categorie_2, useNA = "no")
popular_cat2 <- names(cat2_freq[cat2_freq >= 4])

train$categorie_2 <- ifelse(
  !is.na(train$categorie_2) & train$categorie_2 %in% popular_cat2,
  train$categorie_2,
  ifelse(is.na(train$categorie_2), NA, "other_cat2")
)


# category 3 threshold = 50 (stronger filtering because 865 is much -> keep 430)
cat3_freq <- table(train$categorie_3, useNA = "no")
popular_cat3 <- names(cat3_freq[cat3_freq >= 50])

train$categorie_3 <- ifelse(
  !is.na(train$categorie_3) & train$categorie_3 %in% popular_cat3,
  train$categorie_3,
  ifelse(is.na(train$categorie_3), NA, "other_cat3")
)


###############################################################################


# Cleaning our table
train <- train %>%
  filter(!is.na(price)) %>%
  mutate(
    shipping_cat = case_when(
      str_detect(shipping, regex("buyer",  ignore_case = TRUE)) ~ "buyer",
      str_detect(shipping, regex("seller", ignore_case = TRUE)) ~ "seller",
      TRUE ~ "unknown"  
    ),
    shipping_cat = factor(shipping_cat),
    
    # Dealing with NA's
    item_condition_id = factor(replace_na(item_condition_id, 0)),
    categorie_1 = factor(replace_na(categorie_1, "Unknown_cat")),
    categorie_2 = factor(replace_na(categorie_2, "Unknown_cat")),
    categorie_3 = factor(replace_na(categorie_3, "Unknown_cat")),
    brand_name  = factor(replace_na(brand_name, "unknown")),
    
    # truncate descriptions to 90 words
    item_description_trunc = vapply(
      item_description,
      truncate_text,
      FUN.VALUE = character(1),
      max_words = max_words
    ),
    # Concatenate ALL texts (to reduce the dimensionality)
    text_all = str_squish(paste(
      coalesce(name, ""),
      item_description_trunc,
      coalesce(category_name, ""),
      sep = " "
    )),
    
    # We will make 3- grams for our name column in this feature
    name_clean = str_squish(coalesce(name, "")),
    
    # numeric features
    desc_word_count = if_else(
      is.na(item_description) | item_description == "",
      0L,
      str_count(item_description, "\\S+")
    ),
    desc_missing = as.integer(desc_word_count == 0),
    name_length  = nchar(name_clean),
    
    # target (skewed price distribution, make it more symmetrical)
    price_log = log(price + 1)
  )


# histograms of price


p1 <- ggplot(train, aes(x = price)) +
  geom_histogram(
    aes(fill = ..count..),   
    bins = 1000,
    color = "black",
    linewidth = 0.2
  ) +
  scale_fill_viridis_c(option = "viridis") +   
  coord_cartesian(xlim = c(0, 250)) +
  labs(
    title = "Histogram of price",
    x = "Price",
    y = "Frequency",
    fill = "Count"
  ) +
  theme_minimal(base_size = 12)

# ggsave(
#   filename = "Price_Distribution.png",
#   plot = p1,
#   path = "Doc/Plots",
#   width = 8,
#   height = 5,
#   dpi = 300
# )
## histogram of price_log
p2 <- ggplot(train, aes(x = price_log)) +
  geom_histogram(
    aes(fill = ..count..),   
    bins = 100,
    color = "black",
    linewidth = 0.2
  ) +
  scale_fill_viridis_c(option = "viridis") +   
  coord_cartesian(xlim = c(0, 7)) +
  labs(
    title = "Histogram of log(price + 1)",
    x = "log(price + 1)",
    y = "Frequency",
    fill = "Count"
  ) +
  theme_minimal(base_size = 12)

# ggsave(
#   filename = "Price_log_Distribution.png",
#   plot = p2,
#   path = "Doc/Plots",
#   width = 8,
#   height = 5,
#   dpi = 300
# )

###############################################################################

# making a validation set out of the training set
idx_train <- sample.int(nrow(train), size = floor(0.8 * nrow(train)))
train_tr  <- train[idx_train, ]
train_val <- train[-idx_train, ]

###############################################################################
# Brand-level mean log-price (target encoding on TRAIN only, to avoid leakage)

# mean log(price + 1) per brand, enkel op training subset
brand_stats <- train_tr %>%
  group_by(brand_name) %>%
  summarise(
    brand_mean_price_log = mean(price_log, na.rm = TRUE),
    .groups = "drop"
  )

# global mean als fallback voor onbekende brands
global_mean_price_log <- mean(train_tr$price_log, na.rm = TRUE)

# voeg feature toe aan TRAIN
train_tr <- train_tr %>%
  left_join(brand_stats, by = "brand_name")

# voeg feature toe aan VALIDATION
train_val <- train_val %>%
  left_join(brand_stats, by = "brand_name") %>%
  mutate(
    brand_mean_price_log = if_else(
      is.na(brand_mean_price_log),
      global_mean_price_log,    
      brand_mean_price_log
    )
  )

###############################################################################
# Category-level price statistics (log-scale)
# (computed on train_tr only to avoid leakage)

# cat1
cat1_price_stats <- train_tr %>%
  group_by(categorie_1) %>%
  summarise(
    cat1_mean_price_log = mean(price_log, na.rm = TRUE),
    .groups = "drop"
  )

# cat2
cat2_price_stats <- train_tr %>%
  group_by(categorie_2) %>%
  summarise(
    cat2_mean_price_log = mean(price_log, na.rm = TRUE),
    .groups = "drop"
  )

# cat3
cat3_price_stats <- train_tr %>%
  group_by(categorie_3) %>%
  summarise(
    cat3_mean_price_log = mean(price_log, na.rm = TRUE),
    .groups = "drop"
  )

# join to TRAIN
train_tr <- train_tr %>%
  left_join(cat1_price_stats, by = "categorie_1") %>%
  left_join(cat2_price_stats, by = "categorie_2") %>%
  left_join(cat3_price_stats, by = "categorie_3")

# join to VAL (uses stats from train_tr)
train_val <- train_val %>%
  left_join(cat1_price_stats, by = "categorie_1") %>%
  left_join(cat2_price_stats, by = "categorie_2") %>%
  left_join(cat3_price_stats, by = "categorie_3")

# fallback for categories in 1 that occur in val set but not in train.
global_cat1_mean <- mean(train_tr$cat1_mean_price_log, na.rm = TRUE)

train_val$cat1_mean_price_log <- if_else(
  is.na(train_val$cat1_mean_price_log),
  global_cat1_mean,
  train_val$cat1_mean_price_log
)


###############################################################################
# Sentiment score of item_description

clean_text_for_sentiment <- function(x) {
  x %>%
    tolower() %>%                                 # consistentie
    str_replace_all("<.*?>", " ") %>%             # HTML tags
    str_replace_all("http\\S+|www\\S+", " ") %>%  # URLs
    str_replace_all("[^[:alnum:][:space:][:punct:]]", " ") %>%  # rare unicode
    str_replace_all("[[:cntrl:]]", " ") %>%       # control chars
    str_replace_all("\\s+", " ") %>%              # spaar whitespace
    str_trim()
}

# helper: leeg als NA (gebruik truncated descriptions!)
train_tr_desc  <- if_else(is.na(train_tr$item_description_trunc),  "", train_tr$item_description_trunc)
train_val_desc <- if_else(is.na(train_val$item_description_trunc), "", train_val$item_description_trunc)

train_tr_desc_clean  <- clean_text_for_sentiment(train_tr_desc)
train_val_desc_clean <- clean_text_for_sentiment(train_val_desc)

# sentimentr in chunks
compute_sentiment_chunks <- function(texts, chunk_size = 20000) {
  n   <- length(texts)
  idx <- split(seq_len(n), ceiling(seq_len(n) / chunk_size))
  
  res_list <- vector("list", length(idx))
  pb <- txtProgressBar(min = 0, max = length(idx), style = 3)
  
  for (i in seq_along(idx)) {
    # sentiment_by geeft o.a. kolom 'ave_sentiment'
    res_list[[i]] <- sentiment_by(texts[idx[[i]]], by = NULL)
    setTxtProgressBar(pb, i)
  }
  close(pb)
  
  dplyr::bind_rows(res_list)$ave_sentiment
}

# effectief berekenen
train_tr$sentiment_desc  <- compute_sentiment_chunks(train_tr_desc_clean)
train_val$sentiment_desc <- compute_sentiment_chunks(train_val_desc_clean)


###############################################################################

# scaling the 2 numeric columns that we have te go beyond the boundary of [0,1]
num_cols <- c("desc_word_count", "desc_missing", "name_length","brand_mean_price_log","cat1_mean_price_log",
              "cat2_mean_price_log",
              "cat3_mean_price_log",
              "sentiment_desc")

# Compute mean and std on training set
means <- sapply(train_tr[, num_cols], mean)
sds   <- sapply(train_tr[, num_cols], sd)

z_scale <- function(x, mean_val, sd_val) {
  (x - mean_val) / sd_val
}

# Apply on train and val
for (col in num_cols) {
  train_tr[[col]]  <- z_scale(train_tr[[col]],  means[col], sds[col])
  train_val[[col]] <- z_scale(train_val[[col]], means[col], sds[col])
}

###############################################################################


####### 'All' the text as feature ##########
# tokenizer with stemming
word_tok <- function(x) {
  tokens <- word_tokenizer(x)      
  lapply(tokens, SnowballC::wordStem, language = "en")
}
# apply our tokenization function on the created column 'all_text'
it_tr_word <- itoken(train_tr$text_all,
                     tokenizer = word_tok,
                     progressbar = FALSE)

# vocab 
vocab_word <- create_vocabulary(
  it_tr_word,
  ngram = c(1L, 2L)        # adding 2-grams to deal with grammatical errors
) %>%
  prune_vocabulary(
    term_count_min      = 20,   # minimum occurance of 20 
    doc_proportion_max  = 0.5,  # max. in 50% of docs 
    vocab_term_max      = 8000  # top 8000 n-grams 
  )

vectorizer_word <- vocab_vectorizer(vocab_word) # making vectorizer for dtm

dtm_tr_word <- create_dtm(it_tr_word, vectorizer_word)  # sparse DTM - matrix

# Creating TF-IDF
tfidf_word <- TfIdf$new()
X_tr_word <- tfidf_word$fit_transform(dtm_tr_word)


####### adding the char 3-grams from 'name' column #######
# custom 3-gram tokenizer function
char_ngram_tokenizer <- function(text, n = 3) {
  # lowercasen + cleanup (no stemming because we work with characters)
  text <- tolower(text)
  text <- gsub("[^a-z0-9]", "_", text)  # non alphanumerics 
  
  tokens_list <- lapply(text, function(t) {
    L <- nchar(t)
    if (L < n) return(character(0))
    # sliding 3-char window
    starts <- 1:(L - n + 1)
    sapply(starts, function(i) substr(t, i, i + n - 1))
  })
  
  return(tokens_list)
}

# tokenizer
it_tr_char <- itoken(
  train_tr$name_clean,
  tokenizer = char_ngram_tokenizer,
  progressbar = FALSE
)

# vocab
vocab_char <- create_vocabulary(it_tr_char) %>%
  prune_vocabulary(
    term_count_min = 30,    # 3-gram needs to be in >= 30 names
    vocab_term_max = 2000   # max 2000 3-grams
  )
# vectorizer
vectorizer_char <- vocab_vectorizer(vocab_char)
#dtm
dtm_tr_char <- create_dtm(it_tr_char, vectorizer_char)
# binary presence
X_tr_char <- sign(dtm_tr_char)

####### categorical features in a sparse OHE matrix #######
X_tr_cat <- sparse.model.matrix(
  ~ shipping_cat + item_condition_id +
    categorie_1 + categorie_2 + categorie_3 +
    brand_name - 1,
  data = train_tr
)

####### numerical features in sparse matrix #######
X_tr_num <- Matrix(
  as.matrix(train_tr %>%
              select(all_of(num_cols))),
  sparse = TRUE
)
colnames(X_tr_num) <- c("desc_word_count", "desc_missing", "name_length", "brand_mean_price_log","cat1_mean_price_log",
                        "cat2_mean_price_log",
                        "cat3_mean_price_log",
                        "sentiment_desc")


# add together all the sparse matrices to one big sparse matrix
X_tr <- cbind(
  X_tr_word,   # word 1-2-gram TF-IDF
  X_tr_char,   # name char-3-grams
  X_tr_cat,    # categoricals
  X_tr_num     # numeric features
)

y_tr <- train_tr$price_log  # log(price + 1) target

dim(X_tr)
X_tr[1:10, 1:10]

## X_tr is the final sparse feature matrix used to train the model.
## It combines four types of features into one high-dimensional Matrix:
##   1) TF-IDF from the product texts
##   2) character 3-grams from the item name
##   3) one-hot encoded categorical variables
##   4) numeric engineered features
##
## Each row represents one item, and each column represents a feature.
## Most cells are zero (sparse), which makes the matrix memory-efficient

## Validation set

# Word features 
it_val_word <- itoken(train_val$text_all,
                      tokenizer = word_tok,
                      progressbar = FALSE)
dtm_val_word <- create_dtm(it_val_word, vectorizer_word)
X_val_word   <- tfidf_word$transform(dtm_val_word)

# Char features 
it_val_char <- itoken(
  train_val$name_clean,
  tokenizer = char_ngram_tokenizer,   
  progressbar = FALSE
)

dtm_val_char <- create_dtm(it_val_char, vectorizer_char)
X_val_char   <- sign(dtm_val_char)

# categorical
X_val_cat <- sparse.model.matrix(
  ~ shipping_cat + item_condition_id +
    categorie_1 + categorie_2 + categorie_3 +
    brand_name - 1,
  data = train_val
)
# numerical features
X_val_num <- Matrix(
  as.matrix(train_val %>%
              select(all_of(num_cols))),
  sparse = TRUE
)
colnames(X_val_num) <- c("desc_word_count", "desc_missing", "name_length", "brand_mean_price_log","cat1_mean_price_log",
                         "cat2_mean_price_log",
                         "cat3_mean_price_log",
                         "sentiment_desc")


# Full Validation sparse matrix
X_val <- cbind(
  X_val_word,
  X_val_char,
  X_val_cat,
  X_val_num
)

y_val <- train_val$price_log

# a big sparse matrix for both train and validation set
dim(X_tr)
dim(X_val)
cat("Word TF-IDF: ", ncol(X_tr_word), "\n")
cat("Char n-grams:", ncol(X_tr_char), "\n")
cat("Categoricals:", ncol(X_tr_cat), "\n")
cat("Numeric:     ", ncol(X_tr_num), "\n")
cat("TOTAL:        ", ncol(X_tr), "\n")

nnz <- X_tr@x %>% length()      
total <- prod(dim(X_tr))        
sparsity <- 1 - (nnz / total)

cat("Non-zero entries: ", nnz, "\n")
cat("Total entries:    ", total, "\n")
cat("Sparsity:         ", round(sparsity * 100, 4), "%\n")

# The sparse feature matrix is 99.55% zeros.
# This is expected for TF-IDF + n-grams + categorical dummies.
# Only ~0.45% of all entries contain information.
# The matrix is large but extremely memory-efficient,

# however for the modelling phase it will be beneficial to reduce the dimensionality
# we apply SVD for this only on the text features (word TF-IDF + char n-grams)

set.seed(123)

text_tr  <- cbind(X_tr_word, X_tr_char)
text_val <- cbind(X_val_word, X_val_char)
k_svd <- 100

# svd for X_tr
svd_fit_text <- irlba(text_tr, nv = k_svd)
str(svd_fit_text) # u,d and v

X_tr_text_svd  <- text_tr  %*% svd_fit_text$v
X_val_text_svd <- text_val %*% svd_fit_text$v

dim(X_tr_text_svd)
dim(X_val_text_svd)

# how many of the variance is explained by these 100 components?
d <- svd_fit_text$d
explained <- d^2 / sum(d^2)
cum_explained <- cumsum(explained)

# Plot cumulative variance
# png(
#   filename = "Doc/Plots/svd_cumulative_variance.png",
#   width = 1200, height = 600, res = 150
# )
plot(
  cum_explained,
  type = "l",
  xlab = "Number of components (k)",
  ylab = "Cumulative explained variance",
  main = "Cumulative variance explained by SVD components"
)
abline(h = 0.9, col = "red", lty = 2)
abline(h = 0.95, col = "blue", lty = 2)

dev.off()


colnames(X_tr_text_svd)  <- paste0("svd_text_", 1:k_svd)
colnames(X_val_text_svd) <- paste0("svd_text_", 1:k_svd)

X_tr_text_svd  <- Matrix(X_tr_text_svd,  sparse = TRUE)
X_val_text_svd <- Matrix(X_val_text_svd, sparse = TRUE)

X_tr_final <- cbind(
  X_tr_text_svd,  # 100 SVD-text features
  X_tr_cat,       # categoricals
  X_tr_num        # numerics
)

X_val_final <- cbind(
  X_val_text_svd,
  X_val_cat,
  X_val_num
)

dim(X_tr_final)
dim(X_val_final)

dir.create("Data/Gold/new_features", recursive = TRUE, showWarnings = FALSE)

# 1. Final model matrices
saveRDS(X_tr_final, "Data/Gold/new_features/X_tr_final.rds")
saveRDS(X_val_final, "Data/Gold/new_features/X_val_final.rds")

# 2. Targets
saveRDS(y_tr,  "Data/Gold/new_features/y_tr.rds")
saveRDS(y_val, "Data/Gold/new_features/y_val.rds")


colSums(is.na(train_val[, num_cols]))





