### Load Libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

## MovieLens 10M dataset:
## https://grouplens.org/datasets/movielens/10m/
## http://files.grouplens.org/datasets/movielens/ml-10m.zip

### Download data
dl <- tempfile() #set up temporary file
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl) #download zip file from url and save to temp

### Read the data
#read in the first file with the ratings
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))), 
                 col.names = c("userId", "movieId", "rating", "timestamp"))

# read in the second file with the movie names and genres
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3) 
colnames(movies) <- c("movieId", "title", "genres") 
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

# join the two tables into a single dataframe
movielens <- left_join(ratings, movies, by = "movieId") 

### Format the Data
movielens$genres <- factor(movielens$genres) #convert the genre column to factor
movielens$year <- as.numeric(str_sub(movielens$title, start = -5, end = -2)) #extract the release year from the movie title
movielens$date <- as_datetime(movielens$timestamp) #convert the timestamp to a date

### Partition Validation Set (10% of MovieLens data)
set.seed(1, sample.kind="Rounding") #set seed for reproducibility
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE) #create index
edx <- movielens[-test_index,] #subset train set (edx)
temp <- movielens[test_index,] #create temporary subset for validation

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
    semi_join(edx, by = "movieId") %>%  #remove rows with movieIDs in validation set that do not appear in train set (edx)
    semi_join(edx, by = "userId") #remove rows with userIDs in validation set that do not appear in train set (edx)

# Add the rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# remove the intermediate data objects from environment
rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Set up the Training (90%) and Testing Set (10%)
set.seed(2, sample.kind = "Rounding") #set seed for reproducibility
inTest <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE) #create index
training <- edx[-inTest,] #subset training set
temp_test <- edx[inTest,] #create temporary subset for testing set

# Make sure userId and movieId in testing set are also in training set
testing <- temp_test %>% 
    semi_join(training, by = "movieId") %>% #remove rows with movieIDs in testing set that do not appear in training set
    semi_join(training, by = "userId") #remove rows with userIDs in testing set that do not appear in training set

# Add the rows removed from testing set back into training set
removed_test <- anti_join(temp_test, testing) 
training <- rbind(training, removed_test)

# remove the intermediate data objects from environment
rm(temp_test, removed_test, inTest)

### Write RMSE function
RMSE <- function(true_ratings, predicted_ratings){
    sqrt(mean((true_ratings - predicted_ratings)^2))
}

## Use training set to train model, testing set for intermediate evaluation of the RMSE ##
## Use validation set only once on final model ##

### Model 1 - Model the Average
mu_hat <- mean(training$rating) #compute average rating across all movies on the training set
model_1_rmse <- RMSE(testing$rating, mu_hat) #compute the RMSE using defined function on the testing set, using training mu_hat as the predicted rating
rmse_results <- tibble(method = "Average Rating (Naive)", RMSE = model_1_rmse) #add the test RMSE from model 1 to the running kable

rmse_results %>% knitr::kable()

### Model 2 - Model Movie Effects
# adjust the general ratings (naive) average by the average rating of each movie
movie_avg <- training %>% group_by(movieId) %>% summarise(b_i = mean(rating - mu_hat)) #calculate b_i, the adjustement value for each movie

# prediction: add/subtract b_i for each movie from the average for all movies (mu_hat) to adjust mu_hat for the movie rating prediction
pred_bi <- testing %>% 
    left_join(movie_avg, by = "movieId") %>% #add column of b_i for available rows 
    mutate(pred = mu_hat + b_i) %>%.$pred  #calculate prediction

model_2_rmse <- RMSE(pred_bi, testing$rating) #compute the RMSE on the testing set
rmse_results <- bind_rows(rmse_results, tibble(method = "Movie Effect Model", RMSE = model_2_rmse)) #add the test RMSE from model 2 to the running kable

### Model 3 - Add User Effect
# further adjust the ratings by the average rating of each user
user_avg <- training %>% 
    left_join(movie_avg, by = "movieId") %>% #add column of b_i for available rows
    group_by(userId) %>% summarise(b_u = mean(rating - mu_hat - b_i)) # calculate b_u, the adjustement value for each user

# prediction: add/subtract b_i for each movie AND b_u for each user from the average for all movies
pred_bi_bu <- testing %>% 
    left_join(movie_avg, by = "movieId") %>%  #add column of b_i for available the rows
    left_join(user_avg, by = "userId") %>% #add column of b_u for available  rows
    mutate(pred = mu_hat + b_i + b_u) %>% .$pred #calculate prediction

model_3_rmse <- RMSE(pred_bi_bu, testing$rating) #compute the RMSE on the testing set
rmse_results <- bind_rows(rmse_results, tibble(method = "Movie + User Effects Model", RMSE = model_3_rmse)) #add RMSE to the running kable

rmse_results %>% knitr::kable()

### Model 4 - Add Genre Effect
# further adjust the ratings by the average rating for each genre combination
genre_avg <- training %>% 
    left_join(movie_avg, by = "movieId") %>% #add column of b_i for available rows
    left_join(user_avg, by = "userId") %>% #add column of b_u for available rows
    group_by(genres) %>% summarise(b_g = mean(rating - mu_hat - b_i - b_u)) #calculate b_g, the adjustement value for each genre combo

# prediction: add/subtract b_i for each movie, b_u for each user, and b_g for each genre from the average for all movies
pred_bi_bu_bg <- testing %>% 
    left_join(movie_avg, by = "movieId") %>%  #add column of b_i for available rows
    left_join(user_avg, by = "userId") %>% #add column of b_ui for available rows
    left_join(genre_avg, by = "genres") %>% #add column of b_g for available rows
    mutate(pred = mu_hat + b_i + b_u + b_g) %>% .$pred #calculate prediction

model_4_rmse <- RMSE(pred_bi_bu_bg, testing$rating) #compute the RMSE
rmse_results <- bind_rows(rmse_results, tibble(method = "Movie + User + Genres Effects Model", RMSE = model_4_rmse)) #add RMSE to kable

rmse_results %>% knitr::kable()

### Model 5 - Add Release Year Effect
# further adjust the ratings by the average rating for each release (premier) year of the movie 
year_avg <- training %>% 
    left_join(movie_avg, by = "movieId") %>% ##add column of b_i for available rows
    left_join(user_avg, by = "userId") %>% #add column of b_u for available rows
    left_join(genre_avg, by = "genres") %>% #add column of b_g for available rows
    group_by(year) %>% summarise(b_r = mean(rating - mu_hat - b_i - b_u - b_g)) #calculate b_r, the adjustement value for each release year

# prediction: add/subtract b_i for each movie, b_u for each user, b_g for each genre, and b_r for each release year from the average for all movies
pred_bi_bu_bg_br <- testing %>% 
    left_join(movie_avg, by = "movieId") %>% #add column of b_i for available rows
    left_join(user_avg, by = "userId") %>% #add column of b_u for available rows
    left_join(genre_avg, by = "genres") %>% #add column of b_g for available rows
    left_join(year_avg, by = "year") %>% #add column of b_r for available rows
    mutate(pred = mu_hat + b_i + b_u + b_g + b_r) %>% .$pred #calculate prediction

model_5_rmse <- RMSE(pred_bi_bu_bg_br, testing$rating) #compute the RMSE
rmse_results <- bind_rows(rmse_results, tibble(method = "Movie + User + Genres + Release Year Effects Model", RMSE = model_5_rmse)) #add RMSE to kable

rmse_results %>% knitr::kable()

### Model 6 - Regularisaion
# write a function that takes a training and testing dataset as well as a tuning parameter lambda.
# first, the function calculates the terms of the model as above for b_i (movie effect), b_u (user effect), 
# b_g (genre effect), and b_r (release year effect) but additionally penalises the terms based on the 
# sample size n and thus also uses sum() instead of mean() since we are dividing by (n + l) explictly.
# second, the function computes the prediction as in Model 5 by adding or subtracting each calculated and shunken term from mu.
# finally, the function  evaluates the RMSE internally by the usual formula and returns the RMSE as the output.
rmse_reg_fun <- function(training, testing, l){
    mu <- mean(training$rating)
    b_i <- training %>% group_by(movieId) %>%
        summarise(b_i = sum(rating - mu)/(n()+l))
    b_u <- training %>% 
        left_join(b_i, by = "movieId") %>%
        group_by(userId) %>% summarise(b_u = sum(rating - b_i - mu)/(n()+l))
    b_g <- training %>% 
        left_join(b_i, by = "movieId") %>% #add column of b_i for available rows
        left_join(b_u, by = "userId") %>% #add column of b_u for available rows
        group_by(genres) %>% summarise(b_g = sum(rating - b_i - b_u - mu)/(n()+l)) #calculate b_g, the penalised adjustement value for each genre
    b_r <- training %>% 
        left_join(b_i, by = "movieId") %>% 
        left_join(b_u, by = "userId") %>% 
        left_join(b_g, by ="genres") %>%
        group_by(year) %>% summarise(b_r = sum(rating - b_i - b_u - b_g - mu)/(n()+l))
    
    predicted_ratings <- testing %>% 
        left_join(b_i, by = "movieId") %>%
        left_join(b_u, by = "userId") %>%
        left_join(b_g, by ="genres") %>%
        left_join(b_r, by = "year") %>%
        mutate(pred = mu + b_i + b_u + b_g + b_r) %>%
        .$pred
    
    return(RMSE(predicted_ratings, testing$rating))
}

# run the regularisation function over a vector of lambdas using the training and testing sets
lambdas <- seq(0, 10, 0.25)
rmses_reg <- sapply(lambdas, rmse_reg_fun, training = training, testing = testing) #note: this takes a while to run!

model_6_rmse <- min(rmses_reg) #select the mininised RMSE 
rmse_results <- bind_rows(rmse_results, tibble(method = "Regularised Model (Final Model)", RMSE = model_6_rmse)) #add RMSE to kable

rmse_results %>% knitr::kable()

### Evaluate RMSE on Validation Set
lambda <- lambdas[which.min(rmses_reg)] #choose lambda that minimised the RMSE on the testing set 
validation_rmse <- rmse_reg_fun(training, validation, lambda) #use the defined function to calculate the RMSE with regularisation at the selected lambda
rmse_results <- bind_rows(rmse_results, tibble(method = "Final Validation", RMSE = validation_rmse)) #add RMSE from validation to the running kable

rmse_results %>% knitr::kable()
