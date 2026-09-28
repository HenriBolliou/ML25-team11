######## ONE HOT ENCODE BRAND NAME + categorie LEVELS ########

train_clean_joined_brandname <- read.csv("Data/Silver/train_clean_joined.csv", header = TRUE)

# 1. DATA VOORBEREIDEN
train_clean_joined_brandname <- train_clean_joined_brandname %>%
  mutate(
    # item_condition_id
    item_condition_id = replace_na(item_condition_id, 0),
    item_condition_id = as.factor(item_condition_id),
    
    # brand_name
    brand_name = replace_na(brand_name, "Unknown"),
    brand_name = as.factor(brand_name),
    
    # categorie-niveaus
    categorie_1 = replace_na(categorie_1, "Unknown"),
    categorie_2 = replace_na(categorie_2, "Unknown"),
    categorie_3 = replace_na(categorie_3, "Unknown"),
    
    categorie_1 = as.factor(categorie_1),
    categorie_2 = as.factor(categorie_2),
    categorie_3 = as.factor(categorie_3)
  )

# 2. DREMPELDREMPEL (top 25) VOOR BRAND_NAME
min_freq_top_25 <- train_clean_joined_brandname %>%
  count(brand_name, sort = TRUE) %>%
  slice(25) %>%
  pull(n)

print(paste("Minimale frequentie voor top 25:", min_freq_top_25))


# 3. RECEPT MET OHE VOOR ALLES (GEEN PRICE!)
ohe_recipe <- recipe(~ ., data = train_clean_joined_brandname) %>%
  
  # ITEM CONDITION
  step_dummy(item_condition_id) %>%
  
  # BRAND  
  step_other(
    brand_name,
    threshold = min_freq_top_25,
    other = "other_brand"
  ) %>%
  step_dummy(brand_name) %>%
  
  # categorie LEVEL 1
  step_other(
    categorie_1,
    threshold = 0.01,       # 1% van data → pas aan indien gewenst
    other = "other_cat1"
  ) %>%
  step_dummy(categorie_1) %>%
  
  # categorie LEVEL 2
  step_other(
    categorie_2,
    threshold = 0.01,
    other = "other_cat2"
  ) %>%
  step_dummy(categorie_2) %>%
  
  # categorie LEVEL 3
  step_other(
    categorie_3,
    threshold = 0.01,
    other = "other_cat3"
  ) %>%
  step_dummy(categorie_3)


# 4. RECEPT TRAINEN EN TOE PASSEN
ohe_recipe_trained <- prep(ohe_recipe, training = train_clean_joined_brandname)
train_processed <- bake(ohe_recipe_trained, new_data = train_clean_joined_brandname)

# 5. RESULTAAT BEKIJKEN
glimpse(train_processed)

write.csv(train_processed, "Data/Silver/train_ohe.csv", row.names = FALSE)

