library(tidyverse)
library(word2vec)
library(stringr)
library(tm)
library(pacman)
p_load(Rtsne,scales,ggrepel, recipes, ggplot2)

# Read the training data
train <- read_csv("Data/Silver/train_clean_joined.csv")

sapply(train, class)
colSums(is.na(train))


# ==============================================================================
# FEATURE ENGINEERING FOR 'shipping' (Label Encoding from Text)
# ==============================================================================

# Create the new shipping_label column
shipping_encoded <- train %>%
  mutate(
    shipping_label = case_when(
      # Label 1: Shipping paid by Buyer (using case-insensitive detection)
      str_detect(shipping, regex("buyer", ignore_case = TRUE)) ~ 1,     
      
      # Label 0: Shipping paid by Seller (using case-insensitive detection)
      str_detect(shipping, regex("seller", ignore_case = TRUE)) ~ 2,     
      
      # Label 2: handle the NA values
      TRUE ~ 0               
    )
  ) %>%
  # Select only the new, encoded column
  select(shipping_label)


# ==============================================================================
# FEATURE ENGINEERING FOR 'item_description'
# ==============================================================================
# Word Count for Description 
description_word_count <- train$item_description %>% 
  stringr::str_count(pattern = "\\S+") 

description_word_count_df <- data.frame(description_word_count = description_word_count) 
# Word2Vec Embeddings for Description 
description_text_vector <- train$item_description %>% tolower() 
# Train W2V model (100 dimensions) 
w2v_description_model <- word2vec(x = description_text_vector, 
                                  type = "cbow", dim = 100, 
                                  window = 3, iter = 10, 
                                  stopwords = stopwords()) 
# Aggregate word vectors to document vectors (mean method) 
item_description_vectors <- doc2vec(w2v_description_model, 
                                    newdata = description_text_vector, 
                                    method = "mean") 
# Convert matrix to DataFrame and rename columns for clarity (desc_V1, desc_V2, ...) 
description_vectors_df <- as.data.frame(item_description_vectors) %>% 
  #Rename columns V1, V2, ... to desc_V1, desc_V2, ... 
  rename_with(~ paste0("desc_", .x), starts_with("V"))

# documents without text will be given NA, for our ML model we want a numeric value
description_vectors_df[is.na(description_vectors_df)] <- 0 



# ==============================================================================
# FEATURE ENGINEERING FOR 'name'
# ==============================================================================
# Word Count for Name
name_word_count <- train$name %>% 
  stringr::str_count(pattern = "\\S+") 

name_word_count_df <- data.frame(name_word_count = name_word_count) 

# Word2Vec Embeddings for Name 
name_text_vector <- train$name %>% tolower() 
# Train a separate W2V model for 'name' (100 dimensions) 
w2v_name_model <- word2vec(x = name_text_vector,
                           type = "cbow", 
                           dim = 100, window = 3,
                           iter = 10,
                           stopwords = stopwords()) 
# Aggregate word vectors to document vectors (mean method)
name_vectors <- doc2vec(w2v_name_model, newdata = name_text_vector,
                        method = "mean") 
# Convert matrix to DataFrame and rename columns for clarity (name_V1, name_V2, ...) 
name_vectors_df <- as.data.frame(name_vectors) %>% 
  # Rename columns V1, V2, ... to name_V1, name_V2, ... 
  rename_with(~ paste0("name_", .x), starts_with("V"))

name_vectors_df[is.na(name_vectors_df)] <- 0


# ==============================================================================
# FEATURE ENGINEERING FOR 'item_condition' and 'brand_name'
# ==============================================================================


# Prepping (NA's)
train_prepped <- train %>%
  mutate(
    # item_condition_id 
    item_condition_id = replace_na(item_condition_id, 0),
    item_condition_id = as.factor(item_condition_id),
    
    # brand_name 
    brand_name = replace_na(brand_name, "Unknown"),
    brand_name = as.factor(brand_name)
  )

# Threshold because brands taht aren't much chosen is just noise for the model
min_freq_top_25 <- train_prepped %>%
  count(brand_name, sort = TRUE) %>%
  slice(25) %>%
  pull(n)

print(paste("Minimale frequentie voor top 25 is:", min_freq_top_25))

# One recipe
ohe_recipe <- recipe(price ~ ., data = train_prepped) %>%
  # Threshold for rare brand names
  step_other(
    brand_name,
    threshold = min_freq_top_25,
    other = "other_brand"
  ) %>%
  # making dummies
  step_dummy(item_condition_id) %>%
  step_dummy(brand_name)

# Prep en bake
ohe_recipe_trained <- prep(ohe_recipe, training = train_prepped)
train_processed <- bake(ohe_recipe_trained, new_data = train_prepped)
brand_itemc_processed <- train_processed %>%
  select( -name, -category_name, -shipping, -item_description, -categorie_1, 
         -categorie_2, -categorie_3, -price)

# ==============================================================================
# FEATURE ENGINEERING FOR 'category column'
# ==============================================================================

cat_recipe <- recipe(price ~ ., data = train) %>%
  step_unknown(categorie_1, categorie_2, categorie_3,
               new_level = "Unknown_cat") %>%
  step_other(categorie_1, threshold = 0.01) %>%  # setting thresholds for categories that aren't much chosen (=noise)
  step_other(categorie_2, threshold = 0.01) %>%
  step_other(categorie_3, threshold = 0.01) %>%
  step_dummy(categorie_1, categorie_2, categorie_3)

cat_recipe_trained <- prep(cat_recipe, training = train)
cat_processed <- bake(cat_recipe_trained, new_data = train)
cat_processed <- cat_processed %>% select(-index, -name, -category_name, 
                                          -shipping,- brand_name, - item_condition_id, 
                                          -item_description, -price)


# ==============================================================================
# FINAL MERGE (COMBINE ALL FEATURES)
# ==============================================================================

# Bind all feature dataframes to the original 'train' data
train_features <- bind_cols(
  train,                             # Original data
  shipping_encoded,                  # Shipping Label (0, 1, 2)
  brand_itemc_processed,             # brand names and item_condition (0,1)
  cat_processed,                     # categories (0,1)
  description_word_count_df,         # Description Word Count
  description_vectors_df,            # 100 Description W2V features 
  name_word_count_df,                # Name Word Count
  name_vectors_df                    # 100 Name W2V features 
)


train_features <- train_features %>%
  select( -name, -category_name,- item_condition_id, -shipping, -item_description, - brand_name,
         -categorie_1,-categorie_2,-categorie_3 )

# Final check of the new dimensions
print(dim(train_features))

# Final check if every column is processed properly
sapply(train_features, class)
colSums(is.na(train_features))

# handle the NA values of the description word count feature
train_features$description_word_count[is.na(train_features$description_word_count)] <- 0
# delete the 17 rows where the target column (price) is NA
train_features <- train_features %>% filter(!is.na(price))
summary(train_features$price)
p <- ggplot(train_features, aes(price)) +
      geom_histogram(bins = 500, fill = "steelblue", color = "white") +
      coord_cartesian(xlim = c(0, 150)) +
      theme_minimal()

if (!dir.exists("Doc/Plots")) {
dir.create("Doc/Plots", showWarnings = FALSE)
} else {
  cat(paste("Doc/Plots", "does already exists.\n"))
}
ggsave(
  filename = "Price_Distribution.png",
  plot = p,
  path = "Doc/Plots",
  width = 8,
  height = 5,
  dpi = 300
)


if (!dir.exists("Data/Gold")) {
  dir.create("Data/Gold", recursive = TRUE, showWarnings = FALSE)
} else {
  cat(paste("Gold", "Data/Gold", "does already exists.\n"))
}
write_csv(train_features, "Data/Gold/train_features_encoded.csv")
