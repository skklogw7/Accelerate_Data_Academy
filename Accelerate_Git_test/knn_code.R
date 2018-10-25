## Comparing linear regression to KNN ##
library(MASS)
library(tidyverse)
library(car)
library(FNN)

packages.required <- c("ggplot2", "dplyr", "MASS", "FNN")
packages.we.need.to.install <- packages.required[!(packages.required %in% installed.packages()[,"Package"])]
if(length(packages.we.need.to.install) > 0) install.packages(packages.required)


## Read real estate data in from the directory where you downloaded
setwd("~/Google Drive/Coursework/Graduate/Year 5/STA567 Statistical Learning/lecture-notes/Day7-CV")
realestate <- read.csv("realestate.csv")
head(realestate)

# Standardize input variables for more balanced neighborhood distance calculations
realestate$std_sqft <- scale(realestate$sqft)
realestate$std_lot <- scale(realestate$lot)
head(realestate)

## split data into training/test sets
set.seed(42)
split_level <- .80
train_index <- sort(sample(1:nrow(realestate),
                           round(split_level*nrow(realestate))))
train_index

train_data <- realestate[train_index, ]
test_data <- realestate[-train_index, ]

## -------------------------------------------------------------------------------------
# KNN
## -------------------------------------------------------------------------------------

## Try to find neighbors based on lot size and sqft of house
ggplot() + 
        geom_point(aes(x=sqft, y=lot), data=train_data) +
        labs(x="House Sqft", y="Lot Size") + theme_bw()


## Note the standardized versions maintain same relationship for inputs
ggplot() + 
        geom_point(aes(x=std_sqft, y=std_lot), data=train_data) +
        labs(x="House Sqft", y="Lot Size") + theme_bw()

###------------------------------------------
# Using prebuilt functions in FNN package

# Fitting KNN then checking test performance using validation set
knnTest <- knn.reg(train = train_data[,c("std_sqft","std_lot")],
                   test = test_data[,c("std_sqft","std_lot")],
                   y = train_data$price, k = 3, algorithm = "brute")
str(knnTest)
knnTest$pred
testMSE <- sum((test_data$price-knnTest$pred)^2)/nrow(test_data)
testMSE
sqrt(testMSE)

# Note, let's play around with K and find the best

# Set up bin to store results
k_tuning_frame <- data.frame(K=1:50,
                             TestMSE=numeric(10))

k_tuning_frame

# Iterate over the tuning frame and cross validate
for (i in 1:nrow(k_tuning_frame)){
        knnTest <- knn.reg(train = train_data[,c("std_sqft","std_lot")],
                           test = test_data[,c("std_sqft","std_lot")],
                           y = train_data$price, k = k_tuning_frame$K[i], algorithm = "brute")
        testMSE <- sqrt(sum((test_data$price-knnTest$pred)^2)/nrow(test_data))
        k_tuning_frame$TestMSE[i] <- testMSE
}

# Plot MSE by K
ggplot(k_tuning_frame, aes(x=K, y=TestMSE)) + geom_point() + 
        labs(y='Mean Squared Prediction Error', title='Prediction Error by K') + 
        theme(plot.title = element_text(hjust = 0.5)) + 
        geom_point(data = k_tuning_frame[which.min(k_tuning_frame$TestMSE), ], 
                   aes(K, TestMSE), colour = "red", size = 2)


# fitted vs residuals
ggplot() + geom_point(aes(x=knnTest$pred,y=price-knnTest$pred), data=test_data) + 
        labs(x="fitted",y="residual",title="Residual Plot for KNN Regression (K=5)") + theme_bw()


###-------------------------------------------------------------
# Leave One Out Cross-Validation for KNN regression
###-------------------------------------------------------------

# run for first observation in real estate
realestate[1,]
# Fitting KNN then checking test performance
knnTest <- knn.reg(train = realestate[-1,c("std_sqft","std_lot")],
                   test = realestate[1,c("std_sqft","std_lot")],
                   y = realestate[-1,"price"], k = 5, algorithm = "brute")
MSE1 <- (realestate[1,"price"]-knnTest$pred)^2
MSE1

### Repeat the LOOCV for all 522 houses
# Initialize a storage space to keep all the (Yi-Yhati)^2 values
sqdev <- rep(NA, nrow(realestate))
for(i in 1:nrow(realestate)){
        knnTest <- knn.reg(train = realestate[-i,c("std_sqft","std_lot")],
                           test = realestate[i,c("std_sqft","std_lot")],
                           y = realestate[-i,"price"], k = 5, algorithm = "brute")
        sqdev[i] <- (realestate[i,"price"]-knnTest$pred)^2
}
# MSE is simply the average of stored squared deviations
(MSE <- mean(sqdev))
sqrt(MSE)


### Tuning the k parameter
# Now what if we play around with different neighborhood sizes?
K=50
tuning <- data.frame(sqdev = rep(NA,K), K=1:K)
for(k in 1:K){
        print(k)
        mses <- rep(NA, nrow(realestate))
        for(i in 1:nrow(realestate)){
                knnTest <- knn.reg(train = realestate[-i,c("std_sqft","std_lot")],
                                   test = realestate[i,c("std_sqft","std_lot")],
                                   y = realestate[-i,"price"], k = k, algorithm = "brute")
                mses[i] <- (realestate[i,"price"]-knnTest$pred)^2
        }
        tuning$sqdev[k] <- mean(mses)
}

### plot them to see what neighborhood size works best for MSE
ggplot() +
        geom_point(aes(x=K, y=sqdev), data=tuning)
# It looks like the k=9 neighborhood is best (could probably stop at K=5 and be just about as good)
sqrt(tuning[9,"sqdev"])


### ----------------------------------------------------
# Cross validation for classification
### ----------------------------------------------------

# knn function from FNN package will help to fit the classifier
?knn

### lets try to predict the home quality 
# Start with first observation in real estate
realestate[1,]
# Fitting KNN then checking test performance
knnClass <- knn(train = realestate[-1,c("std_sqft","std_lot")],
                test = realestate[1,c("std_sqft","std_lot")],
                cl = realestate[-1,"quality"], k = 5, algorithm = "brute")
str(knnClass)
as.character(knnClass)
as.character(knnClass)==realestate[1,"quality"]
# We got it right!

### Check for each house individually using LOOCV
preds <- rep(NA,nrow(realestate))
for(i in 1:nrow(realestate)){
        knnClass <- knn(train = realestate[-i,c("std_sqft","std_lot")],
                        test = realestate[i,c("std_sqft","std_lot")],
                        cl = realestate[-i,"quality"], k = 5, algorithm = "brute")
        preds[i] <- as.character(knnClass)
}
head(preds)
realestate$quality == preds
sum(1:5 > 2)
# Calculate the number of correct predictions out of number total
error_rate <- sum(realestate$quality != preds)/nrow(realestate) 


# Compare this to the training error rate from fiting and testing to ALL the data
train_preds <- knn(train = realestate[,c("std_sqft","std_lot")],
                   test = realestate[,c("std_sqft","std_lot")],
                   cl = realestate[,"quality"], k = 5, algorithm = "brute")
train_error_rate <- sum(realestate$quality != train_preds)/nrow(realestate) 


### Tuning K for classification
K=30
tuning <- data.frame(K=1:K, error_rate=rep(NA,K))
for(k in 1:K){
        print(k)
        preds <- rep(NA,nrow(realestate))
        for(i in 1:nrow(realestate)){
                knnClass <- knn(train = realestate[-i,c("std_sqft","std_lot")],
                                test = realestate[i,c("std_sqft","std_lot")],
                                cl = realestate[-i,"quality"], k = k, algorithm = "brute")
                preds[i] <- as.character(knnClass)
        }
        # Calculate the error rate
        tuning$error_rate[k] <- sum(realestate$quality != preds)/nrow(realestate) 
}
head(tuning)

ggplot()+
        geom_point(aes(x=K,y=error_rate), data=tuning)
tuning[10,]


###--------------------------------------------------------
# K-fold Cross validation

# use this function to add K grouping indeces
add_cv_cohorts <- function(dat,cv_K){
        if(nrow(dat) %% cv_K == 0){ # if perfectly divisible
                dat$cv_cohort <- sample(rep(1:cv_K, each=(nrow(dat)%/%cv_K)))
        } else { # if not perfectly divisible
                dat$cv_cohort <- sample(c(rep(1:(nrow(dat) %% cv_K), each=(nrow(dat)%/%cv_K + 1)),
                                          rep((nrow(dat) %% cv_K + 1):cv_K,each=(nrow(dat)%/%cv_K)) ) )
        }
        return(dat)
}

# add 10-fold CV labels to real estate data
realestate_cv <- add_cv_cohorts(realestate,10)
head(realestate_cv)

## Use the 10 groups to iteratively fit and check the 5-nearest neighbors model
# initialize for 10 cohorts, errors and counts
cohorts <- data.frame(cohort=1:10,
                      errors = rep(NA,10), 
                      n=rep(NA,10))
# loop over each validation cohort
for(cv_k in 1:10){
        cohort_rows <- which(realestate_cv$cv_cohort == cv_k)
        knnClass <- knn(train = realestate[-cohort_rows , c("std_sqft","std_lot")],
                        test = realestate[cohort_rows , c("std_sqft","std_lot")],
                        cl = realestate[-cohort_rows , "quality"], k = 5, algorithm = "brute")
        preds <- as.character(knnClass)
        cohorts$errors[cv_k] <- sum(realestate$quality[cohort_rows] != preds)
        cohorts$n[cv_k] <- length(cohort_rows)
}
cohorts
# 10-fold CV error rate as total error count
with(cohorts, sum(errors)/sum(n))












