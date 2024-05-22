##  ---
##  Title: Data Analysis and Visualisation - MAT6206
##         Machine Learning Modelling - Assignment 2
##  Name:  Leon Wu(10582390)
##  All codes were referenced from the MAT6206.2 lecture material.
##  ---
  
# Install necessary packages for data analysis and visualization
install.packages(c("tidyverse","caret","ranger","ggpubr"))

# Load required libraries
library(tidyverse)
library(ggpubr)
library(ranger)  #For random forest
library(caret)  #Classification and Regression Training package

# Set the working directory
# Note that you may need to change the path to your work directory
setwd("/Users/leonfuns/Projects/ECU/data-analysis/a2-data/a2")
options(scipen = 999) # show all numbers

## ------------------------------------------------------------------
#        Part 1 – General data preparation and cleaning
## ------------------------------------------------------------------
# You may need to change/include the path of your working directory
mydata = read.csv("HealthCareData_2024.csv", stringsAsFactors = TRUE)
dim(mydata)

## i. Clean the dataset based on the feedback received from Assignment 1.
summary(mydata)

# Merge categories
mydata$AlertCategory = fct_collapse(mydata$AlertCategory, 
                        Informational = c("Informational", "Info"))
mydata$NetworkEventType = fct_collapse(mydata$NetworkEventType, 
                          PolicyViolation = c("Policy_Violation", "PolicyViolation"))
# Remove outliers
mydata$NetworkAccessFrequency[mydata$NetworkAccessFrequency == -1] = NA
mydata$ResponseTime[mydata$ResponseTime > 150 ] = NA
summary(mydata)

## ii. merge the ‘Regular’ and ‘Unknown’ categories together.
mydata$NetworkInteractionType = fct_collapse(mydata$NetworkInteractionType, 
                                Others = c("Regular", "Unknown"))
summary(mydata)

## iii. Remove NA values.
# Delete the System Access Rate, as this column has too many NA data, 
# and it is a weak discriminator
dat.cleaned = na.omit(mydata[,-9]);dat.cleaned
summary(dat.cleaned)

## ------------------------------------------------------------------
#        Part 1 – Generated taning datasets and test datasets
## ------------------------------------------------------------------
# Separate samples of normal and malicious events
dat.class0 = dat.cleaned %>% filter(Classification == "Normal") # normal
dat.class1 = dat.cleaned %>% filter(Classification == "Malicious") # malicious
# Randomly select 9600 non-malicious and 400 malicious samples using your student
# ID, then combine them to form a working data set
set.seed(10582390)
rows.train0 = sample(1:nrow(dat.class0), size = 9600, replace = FALSE)
rows.train1 = sample(1:nrow(dat.class1), size = 400, replace = FALSE)
# Your 10000 ‘unbalanced’ training samples
train.class0 = dat.class0[rows.train0,] # Non-malicious samples
train.class1 = dat.class1[rows.train1,] # Malicious samples
mydata.ub.train = rbind(train.class0, train.class1)
# Your 19200 ‘balanced’ training samples, i.e. 9600 normal and malicious samples each.
set.seed(10582390)
train.class1_2 = train.class1[sample(1:nrow(train.class1), size = 9600,
                                     replace = TRUE),]
mydata.b.train = rbind(train.class0, train.class1_2)
# Your testing samples
test.class0 = dat.class0[-rows.train0,]
test.class1 = dat.class1[-rows.train1,]
mydata.test = rbind(test.class0, test.class1)

## ------------------------------------------------------------------
#       Part 2 – Two supervised learning modelling algorithms (a)
## ------------------------------------------------------------------
set.seed(10582390)
models.list1 = c("Logistic Ridge Regression",
                 "Logistic LASSO Regression",
                 "Logistic Elastic-Net Regression")
models.list2 = c("Classification Tree",
                 "Bagging Tree",
                 "Random Forest")
myModels = c(sample(models.list1, size = 1),
             sample(models.list2, size = 1))
myModels %>% data.frame

## ------------------------------------------------------------------
#              Part 2 – Logistic Elastic-Net Regression
## ------------------------------------------------------------------
#A sequence of lambdas and alphas
lambdas = 10^seq(-3, 3, length = 100)
alphas = seq(0.1,0.9,by=0.1)
# Modeling Logistic Elastic-Net Regression
set.seed(10582390)
mod.ridge.b = train(
            Classification~.,
            data = mydata.b.train,
            method = "glmnet",
            preProcess = NULL,
            trControl = trainControl("repeatedcv", number = 10, repeats = 2),
            tuneGrid = expand.grid(alpha = alphas, lambda = lambdas))
set.seed(10582390)
mod.ridge.ub = train(
            Classification~.,
            data = mydata.ub.train,
            method = "glmnet",
            preProcess = NULL,
            trControl = trainControl("repeatedcv", number = 10, repeats = 2),
            tuneGrid = expand.grid(alpha = alphas, lambda = lambdas))
# Show the best tune results
mod.ridge.b$bestTune;mod.ridge.ub$bestTune

# Make Tuning and search range graphic
gg.en.b = ggplot(mod.ridge.b$results, aes(x = lambda, y = Accuracy, 
              color = factor(alpha))) + geom_point() + geom_line() + 
              scale_x_log10() + 
              labs(title = "Elastic-Net Tuning: Balanced Data Set",
              x = "Lambda (log scale)", y = "Accuracy", color = "Alpha") +
              theme_minimal()

gg.en.ub = ggplot(mod.ridge.ub$results, aes(x = lambda, y = Accuracy, 
              color = factor(alpha))) + geom_point() + geom_line() + 
              scale_x_log10() +
              labs(title = "Elastic-Net Tuning: Unbalanced Data Set",
              x = "Lambda (log scale)", y = "Accuracy", color = "Alpha") +
              theme_minimal()
# Show Tuning and search range graphic
gg.en.b
gg.en.ub

# Model coefficients
coef(mod.ridge.b$finalModel, mod.ridge.b$bestTune$lambda)
coef(mod.ridge.ub$finalModel, mod.ridge.ub$bestTune$lambda)

# Predict with test dataset and show confusion matrix
pred.class.b = predict(mod.ridge.b,new=mydata.test)
pred.class.ub = predict(mod.ridge.ub,new=mydata.test)

cf.b = table(relevel(pred.class.b, ref="Malicious"), 
             relevel(mydata.test$Classification, ref="Malicious"))
cf.ub = table(relevel(pred.class.ub, ref="Malicious"),
              relevel(mydata.test$Classification, ref="Malicious"))

prop.b = round(prop.table(cf.b, 2), digits = 3);prop.b
prop.ub = round(prop.table(cf.ub, 2), digits = 3);prop.ub

cm.cf.b = confusionMatrix(cf.b,mode="everything");cm.cf.b
cm.cf.ub = confusionMatrix(cf.ub,mode="everything");cm.cf.ub

## ------------------------------------------------------------------
#                   Part 2 – RANDOM FOREST
## ------------------------------------------------------------------
# Create a search grid for the tuning parameters
grid.rf = expand.grid(num.trees = c(200, 300, 400, 500), mtry = c(3:11),
                      min.node.size = seq(2, 10, 2), replace = c(TRUE, FALSE),
                      sample.fraction = c(0.5, 0.6, 0.7, 0.8, 1),
                      OOB.misclass = NA, accuracy = NA, b_accuracy = NA,
                      specificity = NA, precision = NA, reccall = NA,
                      f1_score = NA, tp = NA, fp = NA, fn = NA, tn = NA)
# Show the search grid
grid.rf

# A function for random forest modeling
rf.train = function(data.train) {
  for (I in 1:nrow(grid.rf)) {
    rf = ranger(Classification ~ .,
                data = data.train,
                num.trees = grid.rf$num.trees[I],
                mtry = grid.rf$mtry[I],
                min.node.size = grid.rf$min.node.size[I],
                replace = grid.rf$replace[I],
                sample.fraction = grid.rf$sample.fraction[I],
                seed = 10582390,
                respect.unordered.factors = "order")
    
    grid.rf$OOB.misclass[I] = rf$prediction.error %>% round(5) * 100
    # Predict classification
    pred.test = predict(rf, data = mydata.test)$predictions
    # Summary of confusion matrix
    test.cf = confusionMatrix(relevel(pred.test, ref="Malicious"),
                              relevel(mydata.test$Classification, ref="Malicious"))
    
    grid.rf$accuracy[I] = test.cf$overall["Accuracy"]
    grid.rf$b_accuracy[I] = test.cf$byClass["Balanced Accuracy"]
    grid.rf$specificity[I] = test.cf$byClass["Specificity"]
    grid.rf$precision[I] = test.cf$byClass["Precision"]
    grid.rf$reccall[I] = test.cf$byClass["Recall"]
    grid.rf$f1_score[I] = test.cf$byClass["F1"]
    grid.rf$tp[I] = test.cf$table[1,1]
    grid.rf$fn[I] = test.cf$table[2,1]
    grid.rf$fp[I] = test.cf$table[1,2]
    grid.rf$tn[I] = test.cf$table[2,2]
  }
  # Return top 10 sorted results by the OOB misclassification error
  return(grid.rf[order(grid.rf$OOB.misclass, decreasing = FALSE)[1:10],])
}
# Random forest training
train.rf.b = rf.train(mydata.b.train);train.rf.b
train.rf.ub.t = rf.train(mydata.ub.train);train.rf.ub.t

# Function for top 10 grid's confusion matrix
confusion.matrix = function(train.rf){
  cm.vec = c(train.rf[1,13],train.rf[1,15],train.rf[1,14],train.rf[1,16])
  cm.rf = matrix(cm.vec, nrow = 2, byrow = TRUE)
  rownames(cm.rf) = c("Malicious","Normal")
  colnames(cm.rf) = c("Malicious","Normal")
  cm.rf
}

# Show confusion matrix
cm.rf.b = confusion.matrix(train.rf.b)
cm.rf.ub = confusion.matrix(train.rf.ub.t)
cm.rf.b.r = round(prop.table(cm.rf.b, 2), digits = 3)
cm.rf.ub.r = round(prop.table(cm.rf.ub, 2), digits = 3)
cm.rf.b;cm.rf.ub;cm.rf.b.r;cm.rf.ub.r

# Make Tuning and search range graphic
train.rf.b$rownum = rownames(train.rf.b)
gg.rf.b = ggplot(train.rf.b, aes(x = rownum, y = OOB.misclass, 
            color = as.factor(num.trees), shape = as.factor(min.node.size), 
            group = 1)) + geom_point(size = 3) + geom_line() +
            geom_text(aes(label = paste("mtry:", as.character(mtry))), 
              vjust = 1.5, size = 3) +
            labs(title = 
    "Random Forest: Hyperparameter Tuning/Search Strategy for balance dataset",
            x = "Top 10 Hyperparameter Tuning/Search Grid", 
            y = "OOB misclassification error") + theme_bw()
gg.rf.b
train.rf.ub.t$rownum = rownames(train.rf.ub.t)
gg.rf.ub = ggplot(train.rf.ub.t, aes(x = rownum, y = OOB.misclass, 
              color = as.factor(num.trees), shape = as.factor(min.node.size), 
              group = 1)) + geom_point(size = 3) + geom_line() +
              geom_text(aes(label = paste("mtry:", as.character(mtry))), 
                vjust = 1.5, size = 3) +
              labs(title = 
  "Random Forest: Hyperparameter Tuning/Search Strategy for unbalance dataset",
              x = "Top 10 Hyperparameter Tuning/Search Grid", 
              y = "OOB misclassification error") + theme_bw()
gg.rf.ub

# Output the png images
output.img = function(x,y) {
  ggexport(x, filename=y ,width = 512 ,height = 384)
}
output.img(gg.en.b,"Elastic-Net-Tuning-balance.png")
output.img(gg.en.ub,"Elastic-Net-Tuning-unbalance.png")
output.img(gg.rf.b,"Random-Forest-Tuning-balance.png")
output.img(gg.rf.ub,"Random-Forest-Tuning-unbalance.png")






