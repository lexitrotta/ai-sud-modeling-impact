# r/eda_modeling.R

library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)

data_path <- "data/processed/ai_sud_clean.csv"
df <- read_csv(data_path)

# Basic EDA
df %>%
  select(citation_count, impact_factor, year) %>%
  summary()

# Train/test split
set.seed(42)
train_index <- createDataPartition(df$citation_count, p = 0.8, list = FALSE)
train <- df[train_index, ]
test  <- df[-train_index, ]

features <- c("impact_factor", "year") # add more once encoded
X_train <- as.matrix(train[, features])
y_train <- train$citation_count
X_test  <- as.matrix(test[, features])
y_test  <- test$citation_count

# Random Forest
rf_model <- randomForest(x = X_train, y = y_train)
rf_pred  <- predict(rf_model, X_test)

rf_r2   <- cor(rf_pred, y_test)^2
rf_mae  <- mean(abs(rf_pred - y_test))
rf_rmse <- sqrt(mean((rf_pred - y_test)^2))

print(glue::glue("RF -> R2={round(rf_r2, 2)}, MAE={round(rf_mae,2)}, RMSE={round(rf_rmse,2)}"))
