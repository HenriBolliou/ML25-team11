library(tidyverse)

test <- read.csv('Data/Bronze/test_X.csv')
test_cleaned <- read.csv('Data/Bronze/test_clean.csv')

# small normalization to avoid artificial splits
# Delete - from t-shirt

test <- test %>%
  mutate(
    category_name = str_replace_all(
      category_name, 
      regex("t-shirt", ignore_case = TRUE), 
      "tshirt"
    )
  )


# extract category tokens using explicit separators
p1_data <- test %>%
  filter(str_detect(category_name, "\\/|\\>|_|\\||\\-")) 

p1_splits <- p1_data %>%
  separate_rows(category_name, sep = "\\s*(\\/+|\\>|_|\\||\\-)\\s*") %>%
  mutate(category_name = str_trim(category_name))

# dictionary of known categories (lowercase)
bekende_categorien_lower <- p1_splits %>%
  pull(category_name) %>%
  tolower() %>%
  unique()

print("Woordenboek van bekende categorieën (lowercase):")
print(bekende_categorien_lower)


# split function
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

# apply the split logic to all category entries
gesplitste_data <- map_dfr(
  test$category_name, 
  ~ split_logisch(.x, bekende_categorien_lower)
)

# combine original test data with extracted category levels
resultaat <- bind_cols(test, gesplitste_data)


print("Eindresultaat:")
print(resultaat)

# merge extracted category levels into the cleaned test dataset
colnames(resultaat)
colnames(test_cleaned)

test_joined <- test_cleaned %>%
  left_join(
    resultaat %>% select(index, categorie_1, categorie_2, categorie_3),
    by = "index"
  )

# write to silver layer for further feature engineering
if (!dir.exists("Data/Silver")) {
  dir.create("Data/Silver", recursive = TRUE)
}

write.csv(test_joined, "Data/Silver/test_clean_joined.csv", row.names = FALSE)
