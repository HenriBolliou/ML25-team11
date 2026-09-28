library(tidyverse)
library(text2vec)
library(Matrix)

test <- read_csv("Data/Silver/test_clean_joined.csv")

###############################################################################

# identical filtering as in trainingset
test$brand_name <- ifelse(
  test$brand_name %in% popular_brands,
  test$brand_name,
  "other_brand"
)

# identical filtering as in trainingset

# category 2 threshold = 4 -> other_cat2
test$categorie_2 <- ifelse(
  !is.na(test$categorie_2) & test$categorie_2 %in% popular_cat2,
  test$categorie_2,
  ifelse(is.na(test$categorie_2), NA, "other_cat2")
)

# category 3 threshold = 50 -> other_cat3
test$categorie_3 <- ifelse(
  !is.na(test$categorie_3) & test$categorie_3 %in% popular_cat3,
  test$categorie_3,
  ifelse(is.na(test$categorie_3), NA, "other_cat3")
)

###############################################################################


# Identical cleaning as the trainingset
test <- test %>%
  mutate(
    shipping_cat = case_when(
      str_detect(shipping, regex("buyer",  ignore_case = TRUE)) ~ "buyer",
      str_detect(shipping, regex("seller", ignore_case = TRUE)) ~ "seller",
      TRUE ~ "unknown"
    ),
    shipping_cat = factor(shipping_cat),
    
    item_condition_id = factor(replace_na(item_condition_id, 0)),
    categorie_1 = factor(replace_na(categorie_1, "Unknown_cat")),
    categorie_2 = factor(replace_na(categorie_2, "Unknown_cat")),
    categorie_3 = factor(replace_na(categorie_3, "Unknown_cat")),
    brand_name  = factor(replace_na(brand_name, "unknown")),
    
    item_description_trunc = vapply(
      item_description,
      truncate_text,
      FUN.VALUE = character(1),
      max_words = max_words
    ),
    
    text_all = str_squish(paste(
      coalesce(name, ""),
      item_description_trunc,
      coalesce(category_name, ""),
      sep = " "
    )),
    
    name_clean = str_squish(coalesce(name, "")),
    
    desc_word_count = if_else(
      is.na(item_description) | item_description == "",
      0L,
      str_count(item_description, "\\S+")
    ),
    desc_missing = as.integer(desc_word_count == 0),
    name_length = nchar(name_clean)
  )

colSums(is.na(test))

## safe factor so that unseen levels in test are mapped to "Unknown_cat"
safe_factor <- function(x, train_levels, unknown) {
  x_chr <- as.character(x)
  x_chr[is.na(x_chr)] <- unknown
  x_chr[!(x_chr %in% train_levels)] <- unknown
  factor(x_chr, levels = train_levels)
}

test$shipping_cat    <- safe_factor(test$shipping_cat,    levels(train_tr$shipping_cat),    unknown = "unknown")
test$item_condition_id <- safe_factor(test$item_condition_id, levels(train_tr$item_condition_id), unknown = "0")
test$categorie_1     <- safe_factor(test$categorie_1,     levels(train_tr$categorie_1),     unknown = "Unknown_cat")
test$categorie_2     <- safe_factor(test$categorie_2,     levels(train_tr$categorie_2),     unknown = "Unknown_cat")
test$categorie_3     <- safe_factor(test$categorie_3,     levels(train_tr$categorie_3),     unknown = "Unknown_cat")
test$brand_name      <- safe_factor(test$brand_name,      levels(train_tr$brand_name),      unknown = "unknown")

# check if there are any NA's (important to have the same dimensions)
table(is.na(test$categorie_1))  
table(is.na(test$categorie_2))  
table(is.na(test$categorie_3))  
table(is.na(test$brand_name))   
table(is.na(test$shipping_cat)) 

table(is.na(test$brand_name)) 

# standardized scaling on test set (with use scaler from train)
for (col in num_cols) {
  test[[col]] <- z_scale(test[[col]], means[col], sds[col])
}



# tokenizing using the same tokenizers as in training set
it_test_word <- itoken(
  test$text_all,
  tokenizer = word_tok,
  progressbar = TRUE
)

dtm_test_word <- create_dtm(it_test_word, vectorizer_word)
X_test_word   <- tfidf_word$transform(dtm_test_word)


it_test_char <- itoken(
  test$name_clean,
  tokenizer = char_ngram_tokenizer,
  progressbar = TRUE
)

dtm_test_char <- create_dtm(it_test_char, vectorizer_char)
X_test_char   <- sign(dtm_test_char)

# SVD-projectie met bestaande svd_fit_text
text_test <- cbind(X_test_word, X_test_char)
X_test_text_svd <- text_test %*% svd_fit_text$v
colnames(X_test_text_svd) <- paste0("svd_text_", 1:ncol(X_test_text_svd))
X_test_text_svd <- Matrix(X_test_text_svd, sparse = TRUE)

X_test_cat <- sparse.model.matrix(
  ~ shipping_cat + item_condition_id +
    categorie_1 + categorie_2 + categorie_3 +
    brand_name - 1,
  data = test
)

X_test_num <- Matrix(
  as.matrix(test %>% select(desc_word_count, desc_missing, name_length)),
  sparse = TRUE
)


# check dimensions before concatenation of matrices
cat("nrow(test):       ", nrow(test),        "\n")
cat("nrow(X_test_text_svd):", nrow(X_test_text_svd), "\n")
cat("nrow(X_test_cat): ", nrow(X_test_cat),  "\n")
cat("nrow(X_test_num): ", nrow(X_test_num),  "\n")

stopifnot(
  nrow(X_test_word) == nrow(test),
  nrow(X_test_char) == nrow(test),
  nrow(X_test_cat)  == nrow(test),
  nrow(X_test_num)  == nrow(test)
)

# concatenation of matrices
X_test_final <- cbind(
  X_test_text_svd,
  X_test_cat,
  X_test_num
)
dim(X_tr_final)
dim(X_test_final)
stopifnot(ncol(X_tr_final) == ncol(X_test_final))

saveRDS(X_test_final,"Data/Gold/Features_TFIDF/X_test_final.rds")


# Characteristics of saveRDS() in R
#   
# Single Object Storage: saveRDS() saves only one R object at a time, unlike save() which can store multiple.
# Custom File Naming: we can specify any name for the file, though .rds is commonly used.
# Object Reusability: readRDS() returns the object, allowing us to assign it to any variable.
# No Environment Impact: readRDS() does not directly load into the global environment, offering more control.
# Compact Format: Saves the object in a small-sized binary format, suitable for sharing or storage.

