library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(tidyverse)
library(stringi)

# read data
test <- read.csv('Data/Bronze/test_X.csv', header = TRUE)

## Initial check of testing data


# Check the overall structure of the data
print(str(test))
# Identify the types of all columns (useful for distinguishing features)

print("Data Types of Columns:")
print(sapply(test, class))

# A more robust check for missing values across all columns
# This gives the count of NA's per column
missing_values_count <- colSums(is.na(test))
print("Missing Value Counts per Column:")
print(missing_values_count)

# So a test table with dim 489985 x 8. 
# We received features as name, item condition id, category, brand name, shipping type, item description and price. 
# the data contains mainly character types, this implies that cleaning the the data will be crucial. The text is dirty and a lot of cells are NA.


################################################################################

## clean the brand_name column


## Define common non-standard missing indicators
missing_indicators_to_clean <- c("", "-", "?", ".", " ", "#N/A", "n/a", "NA_", "N/A","NA")
## Convert the defined non-standard missing indicators to explicit NA in 'brand_name'
test$brand_name[test$brand_name %in% missing_indicators_to_clean] <- NA
## Now, recalculate the total missing values for 'brand_name' to see the new count
new_missing_brand_count <- sum(is.na(test$brand_name))
new_missing_brand_percentage <- new_missing_brand_count / nrow(test) * 100

print(paste("New total NA count for 'brand_name':", new_missing_brand_count))
print(paste("New total NA percentage for 'brand_name':", round(new_missing_brand_percentage, 2), "%"))

## exploring the different brandnames to see if we have to delete extra 
## strange tokens
unique_brands <- unique(test$brand_name)
num_unique_brands <- length(unique_brands)

# print("Eerste 20 unieke merknamen (inclusief NA):")
# print(head(unique_brands, 500))
print(paste("Total distinct brand names (inclusive NA):", num_unique_brands))
print(paste("Total distinct brand names (exclusive NA):", sum(!is.na(unique_brands))))

known_brands <- test$brand_name %>% na.omit() %>% unique()
brands_with_specials <- known_brands[str_detect(known_brands, "[^A-Za-z0-9 ]")]
# a lot of brand names contain special tokens
# we make 2 functions to clean the text 

# Get rid of accents: “L'Oréal” -> “L'Oreal”
rm_accents <- function(x) {
  # we can do this by converting the text to ASCII, because ASCII doesn't contain accents
  out <- iconv(x, from = "", to = "ASCII//TRANSLIT")
  out[is.na(out)] <- x[is.na(out)]
  out
}

# normalizing the text for matching: lowercase, trim,...
normalize <- function(x) {
  x <- as.character(x)
  x <- str_squish(tolower(x))
  x <- rm_accents(x)
  
  # by using regex, symbols become consistent
  x <- str_replace_all(x, "&", " and ")
  x <- str_replace_all(x, "\\+", " and ")
  x <- str_replace_all(x, "\\|", " ")     
  x <- str_replace_all(x, "[/\\\\]", " ") 
  x <- str_replace_all(x, "’", "'")       
  x <- str_replace_all(x, "®|™|©", " ")   
  x <- str_replace_all(x, "[\\.-]", " ")  
  x <- str_replace_all(x, "[^a-z0-9' ]", " ")
  # delete excessive white spaces
  x <- str_squish(x)
  x
}


# selecting the most used not normalized brand names
brand_map <- test %>%
  filter(!is.na(brand_name)) %>%
  mutate(brand_norm = normalize(brand_name)) %>%
  count(brand_norm, brand_name, sort = TRUE) %>%
  group_by(brand_norm)%>% slice_max(n, with_ties = FALSE) %>%
  ungroup() %>%
  select(brand_norm, brand_canonical = brand_name)

short_brands <- known_brands[str_length(known_brands) <= 3] %>% sort()
short_brands


# define a blacklist and whitelist
blacklist <- c(
  "and","all","the","for","by","with","from","of","on","in",
  "com","inc","co","ltd","usa","pro","raw","so","op","art","act","api","apt","ask", "pink","silver"
)

whitelist_short <- c(
  "ugg","huf","htc","ibm","hp","3m","nyx","nba","nfl","nhl","rei","rbc","bmx","bmw","fox","gap","jbl","jvc","mac","mcm","rca","rei","tdk","tyr","ugg","wwe","xxi"
)

brand_terms_raw <- brand_map$brand_norm %>%
  unique() %>%
  .[order(nchar(.), decreasing = TRUE)]

brand_terms_raw <- gsub("\\bo b\\b", "ob", brand_terms_raw)
brand_terms_filtered <- tibble(term = brand_terms_raw) %>%
  filter(
    !(term %in% blacklist),
    nchar(term) > 3 | term %in% whitelist_short
  ) %>%
  pull(term)
brand_terms <- brand_terms_filtered %>%
  .[order(nchar(.), decreasing = TRUE)] %>%
  (\(x) gsub("([\\.\\^\\$\\*\\+\\?\\(\\)\\[\\{\\|\\\\])", "\\\\\\1", x, perl = TRUE))()

# we make some chunks to search the name in the description and name column to make the search faster
chunkify <- function(v, size = 800L) split(v, ceiling(seq_along(v)/size))
make_boundary_pattern <- function(terms) {
  paste0("(^|[^[:alnum:]])(", paste(terms, collapse = "|"), ")(?=$|[^[:alnum:]])")
}

# function to take the first match over the regex chunks
first_match_vec <- function(x, patterns) {
  res <- rep(NA_character_, length(x))
  for (pat in patterns) {
    m <- str_match(x, pat)
    hit <- m[,3]
    fill_idx <- which(is.na(res) & !is.na(hit))
    if (length(fill_idx)) res[fill_idx] <- hit[fill_idx]
    if (!any(is.na(res))) break
  }
  res
}

# Building the chunks of brand terms
brand_patterns <- lapply(chunkify(brand_terms, size = 800L), make_boundary_pattern)

# normalise the names and descriptions and seek for the first match
test <- test %>%
  mutate(
    brand_norm_existing = if_else(is.na(brand_name), NA_character_, normalize(brand_name)),
    name_norm = normalize(name),
    desc_norm = normalize(item_description),
    
    brand_hit_name = first_match_vec(name_norm, brand_patterns),
    brand_hit_desc = first_match_vec(desc_norm, brand_patterns),
    
    # prioriteit: bestaande merknaam > match in name > match in description
    brand_imputed_norm = coalesce(brand_norm_existing, brand_hit_name, brand_hit_desc)
  )

# Restore the original names
test <- test %>%
  left_join(brand_map, by = c("brand_imputed_norm" = "brand_norm")) %>%
  mutate(
    brand_name = coalesce(brand_name, brand_canonical, brand_imputed_norm),
    brand_name = na_if(brand_name, "")
  ) %>%
  select(-name_norm, -desc_norm, -brand_hit_name, -brand_hit_desc,
         -brand_imputed_norm, -brand_norm_existing, -brand_canonical)

# Take a look to remaining amount of NA values
cat("Remaining NA in brand_name:", sum(is.na(test$brand_name)), "\n")

test$brand_name[is.na(test$brand_name)] <- "unknown"

################################################################################

## Clean the Shipping column

unique(test$shipping)


test <- test %>%
  mutate(
    shipping = if_else(
      shipping %in% c("shipping by buyer", "shipping by seller"),
      shipping,
      NA_character_
    )
  )
print(unique(test$shipping))


################################################################################


## clean the item_condition_id column

print(typeof(test$item_condition_id))
print(unique(test$item_condition_id))
missing_indicators_to_clean <- c("", "-", "?"," ", "N/A", "unknown", "NA")
test$item_condition_id[test$item_condition_id %in% missing_indicators_to_clean] <- NA


print(paste("Unique values after cleaning:",
            paste(unique(test$item_condition_id), collapse = ", ")))
print(paste("Number of NA values after cleaning: ", sum(is.na(test$item_condition_id))))

################################################################################

# writing our cleaned test table to a csv file

write.csv(test, "Data/Bronze/test_clean.csv", row.names = FALSE)