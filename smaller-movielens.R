library(tidyverse)
library(dslabs)
data("movielens")
movielens %>% as_tibble()

movielens %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))

# multiplication leads to more than 5M rows
# our data is 100K, indicating not all users rated all movies

library(caret)
set.seed(755, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.2, 
                                  list = FALSE)
train_set <- movielens[-test_index,]
test_set <- movielens[test_index,]

# Remove users and movies from test that do not appear in training

test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# First model as a mean value
mu_hat <- mean(train_set$rating)
mu_hat

# Naive RMSE by comparing to basic mean
# our absolute base case

naive_rmse <- RMSE(test_set$rating, mu_hat)
naive_rmse

# Any other value will produce a worse RMSE
# 1.187517
predictions <- rep(3, nrow(test_set))
RMSE(test_set$rating, predictions)

rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)


# Initial Model
# Primary predictor is the users themselves
# Therefore using them we should do better than the Naive approach
fit <- lm(rating ~ as.factor(movieId), data = movielens)

fit_lm <- speedglm::speedlm(rating ~ as.factor(movieId) + as.factor(userId), data = train_set, model=FALSE, sparse=TRUE)
fit_glm <- speedglm::speedglm(rating ~ as.factor(movieId) + as.factor(userId), data = train_set, model=FALSE, sparse=TRUE)

pr_lm <- predict(fit_lm, test_set)
pr_glm <- predict(fit_glm, test_set)

# RMSE: 0.8942266
lm_rmse <- RMSE(pr_lm, test_set$rating)

# RMSE: 0.894246
glm_rmse <- RMSE(pr_glm, test_set$rating)

mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
qplot(b_i, data = movie_avgs, bins = 10, color = I("black"))


predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
RMSE(predicted_ratings, test_set$rating)


# User effects
train_set %>% 
  group_by(userId) %>% 
  filter(n()>=100) %>%
  summarize(b_u = mean(rating)) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
RMSE(predicted_ratings, test_set$rating)



# Final Forms

lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda

full_apply <- function(l) {
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
    predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
}

final_rmse <- full_apply(3.75)
