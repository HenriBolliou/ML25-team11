################################################################################
# Script overview
#
# This script cleans and prepares the raw training data set for the column 'category_name'.
# It splits the hierarchical category_name into to three structured category levels.
#
# The cleaned and enriched dataset is saved for use in later feature engineering
# and modeling steps.
################################################################################

library(tidyverse)

train <- read.csv('Data/Bronze/train.csv')
train_cleaned <- read.csv('Data/Bronze/train_clean.csv')

# small normalization to avoid unwanted splits on tshirt
train <- train %>%
  mutate(
    category_name = str_replace_all(
      category_name, 
      regex("t-shirt", ignore_case = TRUE), 
      "tshirt"
    )
  )


# extract known category tokens to build a reference dictionary
p1_data <- train %>%
  filter(str_detect(category_name, "\\/|\\>|_|\\||\\-")) 

p1_splits <- p1_data %>%
  separate_rows(category_name, sep = "\\s*(\\/+|\\>|_|\\||\\-)\\s*") %>%
  mutate(category_name = str_trim(category_name))

# dictionary of known categories (lowercase)
bekende_categorien_lower <- p1_splits %>%
  pull(category_name) %>%
  tolower() %>%
  unique()

print(bekende_categorien_lower)


# Build a split function 
split_logisch <- function(string, dictionary_lower) {
  
  string <- str_trim(string)
  
  # case 1: explicit separators present
  if (str_detect(string, "\\/|\\>|_|\\||\\-")) { 
    
    splits <- str_split(string, "\\s*(\\/+|\\>|_|\\||\\-)\\s*")[[1]]
    result <- str_trim(splits)
    
    # case 2: no explicit separator, rely on dictionary-based splitting
  } else {
    temp_string <- string %>%
      str_replace_all("([a-z0-9])([A-Z])", "\\1|\\2") %>% 
      str_replace_all("\\s+", "|")                       
    
    kandidaten <- str_split(temp_string, "\\|")[[1]] %>% str_trim()
    
    # keep only valid category candidates
    goede_kandidaten <- kandidaten[tolower(kandidaten) %in% dictionary_lower]
    
    result <- goede_kandidaten
  }
  
  
  result_padded <- c(result, rep(NA, 3))[1:3]
  

  return(list(
    categorie_1 = result_padded[1],
    categorie_2 = result_padded[2],
    categorie_3 = result_padded[3]
  ))
}

# apply the split function to all category entries
gesplitste_data <- map_dfr(
  train$category_name, 
  ~ split_logisch(.x, bekende_categorien_lower)
)

# combine original data with extracted category levels
resultaat <- bind_cols(train, gesplitste_data)

print(resultaat)


# joining of cleaned train table of the 'Cleaning_Train' script and the table above
# which handled the category column

colnames(resultaat)
colnames(train_cleaned)

train_joined <- train_cleaned %>%
  left_join(
    resultaat %>% select(index, categorie_1, categorie_2, categorie_3),
    by = "index"
  )

# write to silver map for further feature engineering
if (!dir.exists("Data/Silver")) {
  dir.create("Data/Silver", recursive = TRUE)
}

write.csv(train_joined, "Data/Silver/train_clean_joined.csv", row.names = FALSE)