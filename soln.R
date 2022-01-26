# Begin Provided Code
##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                         genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# Remove unnecessary variables from memory
rm(dl, ratings, movies, test_index, temp, movielens, removed)

# End Provided Code

# Finding review year and extracting from the timestamp for validation set
dates_edx <- as.Date(as.POSIXct(edx$timestamp, origin="1970-01-01"))
years_edx <- format(dates_edx, format="%Y")

# Producing a new dataframe with years and renaming column for consistency
# for training set
edx_combined <- cbind(edx,years_edx) 
edx_combined <- rename(edx_combined, years = years_edx)

# Finding review year and extracting from the timestamp for validation set
dates_validation <- as.Date(as.POSIXct(validation$timestamp, origin="1970-01-01"))
years_validation <- format(dates_validation, format="%Y")

# Producing a new dataframe with years and renaming column for consistency
# for validation set
validation_combined <- cbind(validation,years_validation)
validation_combined <- rename(validation_combined, years = years_validation)

# Loss Function as RMSE
# 
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


# Training set mean rating
mu <- mean(edx_combined$rating)

# Final Model accepting lambda as variable
final_model <- function(l) {
  
  # calculate bias term for movie
  b_i <- edx_combined%>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu_hat)/(n()+l))
  
  # calculate bias term for user
  b_u <- edx_combined %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu_hat)/(n()+l))
  
  # calculate bias term for genre
  b_g <- edx_combined %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - b_u - b_i - mu_hat)/(n()+l))
  
  # calculate bias term for time of review
  b_t <- edx_combined %>% 
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_g, by="genres") %>%
    group_by(years) %>%
    summarize(b_t = sum(rating - b_g - b_u - b_i - mu_hat)/(n()+l))
  
  # create predictions for validation set
  predicted_ratings <- 
    validation_combined %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_t, by = "years") %>%
    mutate(pred = mu_hat + b_t + b_g + b_i + b_u) %>%
    pull(pred) 
  
  # Calculate RMSE between predictions and validation set ratings
  return(RMSE(predicted_ratings, validation_combined$rating))
}

# RMSE for the final model is 0.8644092
final_rmse <- final_model(5.25)
final_rmse