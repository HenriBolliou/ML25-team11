library(tidyverse)
library(word2vec)
library(stringr)
library(tm)
library(pacman)
p_load(text2vec,Rtsne,scales,ggrepel)

# Read the testing data
test <- read_csv("Data/Silver/test_clean_joined.csv")

sapply(test, class)
colSums(is.na(test))
# ==============================================================================
# FEATURE ENGINEERING FOR 'shipping' (Label Encoding from Text)
# ==============================================================================

# Create the new shipping_label column
shipping_encoded_test <- test %>%
  mutate(
    shipping_label = case_when(
      str_detect(shipping, regex("buyer", ignore_case = TRUE))  ~ 1,
      str_detect(shipping, regex("seller", ignore_case = TRUE)) ~ 2,
      TRUE                                                      ~ 0
    )
  ) %>%
  select(shipping_label)


# ==============================================================================
# FEATURE ENGINEERING FOR 'item_description'
# ==============================================================================
# Word Count for Description
description_word_count_test <- test$item_description %>% 
  stringr::str_count(pattern = "\\S+")
description_word_count_test[is.na(description_word_count_test)] <- 0

description_word_count_test_df <- data.frame(description_word_count = description_word_count_test)

# Word2Vec Embeddings for Description
description_text_vector_test <- test$item_description %>% tolower()

# Embeddings via w2v_description_model of the trainset
# Aggregate word vectors to document vectors (mean method)
item_description_vectors_test <- doc2vec(w2v_description_model, 
                                    newdata = description_text_vector_test,
                                    method = "mean") 

# Convert matrix to DataFrame and rename columns for clarity (desc_V1, desc_V2, ...)
description_vectors_test_df <- as.data.frame(item_description_vectors_test) %>%
  # Rename columns V1, V2, ... to desc_V1, desc_V2, ...
  rename_with(~ paste0("desc_", .x), starts_with("V"))

# NA → 0
description_vectors_test_df[is.na(description_vectors_test_df)] <- 0

# ==============================================================================
# FEATURE ENGINEERING FOR 'name'
# ==============================================================================
# Word Count for Name
name_word_count_test <- test$name %>%
  stringr::str_count(pattern = "\\S+")

name_word_count_test_df <- data.frame(name_word_count = name_word_count_test)

# Word2Vec Embeddings for Name
name_text_vector_test <- test$name %>% tolower()

# Embeddings via w2v_name_model of the trainset
# Aggregate word vectors to document vectors (mean method)
name_vectors_test <- doc2vec(w2v_name_model, 
                        newdata = name_text_vector_test,
                        method = "mean") 

# Convert matrix to DataFrame and rename columns for clarity (name_V1, name_V2, ...)
name_vectors_test_df <- as.data.frame(name_vectors_test) %>%
  # Rename columns V1, V2, ... to name_V1, name_V2, ...
  rename_with(~ paste0("name_", .x), starts_with("V"))

name_vectors_test_df[is.na(name_vectors_test_df)] <- 0

# ==============================================================================
# FEATURE ENGINEERING FOR 'item_condition' and 'brand_name'
# ==============================================================================

test_prepped <- test %>%
  mutate(
    item_condition_id = replace_na(item_condition_id, 0),
    item_condition_id = as.factor(item_condition_id),
    brand_name        = replace_na(brand_name, "Unknown"),
    brand_name        = as.factor(brand_name)
  )

# use trained recipe of trainset
test_processed <- bake(ohe_recipe_trained, new_data = test_prepped)

brand_itemc_test <- test_processed %>%
  select(-index, -name, -category_name, -shipping,
         -item_description, -categorie_1, -categorie_2, -categorie_3,
         any_of("price"))

# ==============================================================================
# FEATURE ENGINEERING FOR 'category column'
# ==============================================================================

# use trained recipe of trainset
cat_processed_test <- bake(cat_recipe_trained, new_data = test)

cat_processed_test <- cat_processed_test %>%
  select( -name, -category_name,
         -shipping, -brand_name, -item_condition_id,
         -item_description, any_of("price"))

# if NA's, set them to 0
cat_processed_test[is.na(cat_processed_test)] <- 0

# ==============================================================================
# FINAL MERGE (COMBINE ALL FEATURES)
# ==============================================================================

test_features <- bind_cols(
  test,                        # original test-features
  shipping_encoded_test,       # shipping label
  brand_itemc_test,            # OHE brand + item_condition
  cat_processed_test,          # category dummies
  description_word_count_test_df,
  description_vectors_test_df,
  name_word_count_test_df,
  name_vectors_test_df
)

test_features <- test_features %>%
  select( -name, -category_name, -item_condition_id,
         -shipping, -item_description, -brand_name,
         -categorie_1, -categorie_2, -categorie_3)

# for any remaining NA's
test_features[is.na(test_features)] <- 0

# Check
dim(test_features)
colSums(is.na(test_features))
sapply(test_features, class)
# Do we have the exact same features in test and train?
setdiff(names(train_features), c("price", names(test_features)))
setdiff(names(test_features), names(train_features))


if (!dir.exists("Data/Gold")) {
  dir.create("Data/Gold", recursive = TRUE, showWarnings = FALSE)
} else {
  cat(paste("Gold", "Data/Gold", "does already exists.\n"))
}
write_csv(test_features, "Data/Gold/test_features_encoded.csv")