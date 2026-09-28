library(pacman)
p_load(caret)


set.seed(123)
# Read the training data
train <- read_csv("Data/Gold/train.csv")

# creating a validation set of 20% of the training set
train_index <- createDataPartition(train$price, p = 0.8, list = FALSE)

train_set <- train[train_index, ]
valid_set <- train[-train_index, ]

dim(train_set)
dim(valid_set)

if (!dir.exists("Data/Gold")) {
  dir.create("Data/Gold", recursive = TRUE, showWarnings = FALSE)
} else {
  cat(paste("Gold", "Data/Gold", "does already exists.\n"))
}
write_csv(train_set,  "Data/Gold/train_set.csv") # 80%
write_csv(valid_set, "Data/Gold/valid_set.csv")  # 20%
