
#Load packages
library(sqldf)
library(ggplot2)
library(plotly)
library(data.table)
library(h2o)
library(dplyr)


#Set working directory
setwd("F:/kaggle/Mercedes-Benz")


#Load data files
sample_submission <- fread("data/sample_submission.csv")
train <- fread("data/train.csv")
test <- fread("data/test.csv")


#Add Train 0/1 Column to both train ans test datasets before merge
train$Train <- 1
test$Train <- 0


#Combine Train and Test datasets
train_test <- train
train_test$y <- NULL #remove response variable before rbind of two datasets
train_test <- rbind(train_test,test)


#### Feature Engineering Section
# #Identify duplicate columns in Training dataset
# NonDuplicateFields <- colnames(train[, !duplicated(t(train)), with=FALSE])
# NonDuplicateFields <- NonDuplicateFields[-2] #remove response variable
# 
# 
# #Remove duplicate columns from both
# train_test <- train_test[, NonDuplicateFields, with=FALSE]


#Split train_test dataset into train_model and valid_model for model creation, also add back response variable
train_valid_w_features <- train_test[Train==1,]
train_valid_w_features <- cbind(y=train$y,train_valid_w_features)
set.seed(12345)
sample_size <- round(0.8*dim(train_valid_w_features)[1],0)
train_ind <- sample(train_valid_w_features$ID, sample_size)
train_model <- train_valid_w_features[(ID %in% train_ind),]
valid_model <- train_valid_w_features[!(ID %in% train_ind),]




####Modeling Section
# h2o.init(nthreads = 6, max_mem_size = '10G') # uses 6 machine cores, 10G of RAM 
# h2o.shutdown(prompt = FALSE)
# Monitor at http://localhost:54321/flow/index.html

train_model_h2o <- as.h2o(train_model, destination_frame = "train_model.hex" )
valid_model_h2o <- as.h2o(valid_model, destination_frame = "valid_model.hex" )

# y <- "y"
# x <- colnames(train_model[,2:379, with=FALSE])
# gbm_model_1 <- h2o.gbm(training_frame = train_model_h2o,      ## H2O frame holding the training data
#                        validation_frame = valid_model_h2o,  ## extra holdout piece for three layer modeling
#                        x=x,                 ## this can be names or column numbers
#                        y=y,                   ## target: using the logged variable created earlier
#                        model_id="gbm_model_1",              ## internal H2O name for model
#                        ntrees = 500,                  ## use fewer trees than default (50) to speed up training
#                        learn_rate = 0.005,             ## lower learn_rate is better, but use high rate to offset few trees
#                        max_depth = 5,
#                        score_tree_interval = 15,      ## score every 3 trees
#                        sample_rate = 0.5,            ## use half the rows each scoring round
#                        col_sample_rate = 0.8        ## use 4/5 the columns to decide each split decision
# )
# #summary(gbm_model_1)
# (h2o.r2(gbm_model_1,train=T)) # 0.6065437
# (h2o.r2(gbm_model_1,valid=T)) # 0.5493548
# #h2o.saveModel(gbm_model_1,"gbm_model_1")
# h2o.performance(gbm_model_1, train_model_h2o)
# h2o.performance(gbm_model_1, valid_model_h2o)





y <- "y"
x <- colnames(train_model[,2:379, with=FALSE])
gbm_model_1 <- h2o.gbm(training_frame = train_model_h2o,      ## H2O frame holding the training data
                       validation_frame = valid_model_h2o,  ## extra holdout piece for three layer modeling
                       x=x,                 ## this can be names or column numbers
                       y=y,                   ## target: using the logged variable created earlier
                       model_id="gbm_model_1",              ## internal H2O name for model
                       ntrees = 900,                  ## use fewer trees than default (50) to speed up training
                       learn_rate = 0.004,             ## lower learn_rate is better, but use high rate to offset few trees
                       max_depth = 3,
                       score_tree_interval = 15,      ## score every 3 trees
                       sample_rate = 0.5,            ## use half the rows each scoring round
                       col_sample_rate = 0.8        ## use 4/5 the columns to decide each split decision
)
summary(gbm_model_1)
(h2o.r2(gbm_model_1,train=T)) # 0.5886679
(h2o.r2(gbm_model_1,valid=T)) # 0.553584
#h2o.saveModel(gbm_model_1,"gbm_model_1")
h2o.performance(gbm_model_1, train_model_h2o)
h2o.performance(gbm_model_1, valid_model_h2o)







# y <- "y"
# x <- colnames(train_model[,2:379, with=FALSE])
# gbm_model_1 <- h2o.gbm(training_frame = train_model_h2o,      ## H2O frame holding the training data
#                        validation_frame = valid_model_h2o,  ## extra holdout piece for three layer modeling
#                        x=x,                 ## this can be names or column numbers
#                        y=y,                   ## target: using the logged variable created earlier
#                        model_id="gbm_model_1",              ## internal H2O name for model
#                        ntrees = 1350,                  ## use fewer trees than default (50) to speed up training
#                        learn_rate = 0.002,             ## lower learn_rate is better, but use high rate to offset few trees
#                        max_depth = 4,
#                        score_tree_interval = 30,      ## score every 3 trees
#                        sample_rate = 0.5,            ## use half the rows each scoring round
#                        col_sample_rate = 0.8        ## use 4/5 the columns to decide each split decision
# )
# summary(gbm_model_1)
# (h2o.r2(gbm_model_1,train=T)) # 0.5948274
# (h2o.r2(gbm_model_1,valid=T)) # 0.5506783
# #h2o.saveModel(gbm_model_1,"gbm_model_1")
# h2o.performance(gbm_model_1, train_model_h2o)
# h2o.performance(gbm_model_1, valid_model_h2o)
# 
# 
# 
# y <- "y"
# x <- colnames(train_model[,2:379, with=FALSE])
# gbm_model_1 <- h2o.gbm(training_frame = train_model_h2o,      ## H2O frame holding the training data
#                        validation_frame = valid_model_h2o,  ## extra holdout piece for three layer modeling
#                        x=x,                 ## this can be names or column numbers
#                        y=y,                   ## target: using the logged variable created earlier
#                        model_id="gbm_model_1",              ## internal H2O name for model
#                        ntrees = 3100,                  ## use fewer trees than default (50) to speed up training
#                        learn_rate = 0.002,             ## lower learn_rate is better, but use high rate to offset few trees
#                        max_depth = 2,
#                        score_tree_interval = 50,      ## score every 3 trees
#                        sample_rate = 0.5,            ## use half the rows each scoring round
#                        col_sample_rate = 0.8        ## use 4/5 the columns to decide each split decision
# )
# summary(gbm_model_1)
# (h2o.r2(gbm_model_1,train=T)) #  0.5838713
# (h2o.r2(gbm_model_1,valid=T)) # 0.5589884
# #h2o.saveModel(gbm_model_1,"gbm_model_1")
# h2o.performance(gbm_model_1, train_model_h2o)
# h2o.performance(gbm_model_1, valid_model_h2o)
# 
# 
# 
# y <- "y"
# x <- colnames(train_model[,2:379, with=FALSE])
# gbm_model_1 <- h2o.gbm(training_frame = train_model_h2o,      ## H2O frame holding the training data
#                        validation_frame = valid_model_h2o,  ## extra holdout piece for three layer modeling
#                        x=x,                 ## this can be names or column numbers
#                        y=y,                   ## target: using the logged variable created earlier
#                        model_id="gbm_model_1",              ## internal H2O name for model
#                        ntrees = 3000,                  ## use fewer trees than default (50) to speed up training
#                        learn_rate = 0.01,             ## lower learn_rate is better, but use high rate to offset few trees
#                        max_depth = 1,
#                        score_tree_interval = 50,      ## score every 3 trees
#                        sample_rate = 0.5,            ## use half the rows each scoring round
#                        col_sample_rate = 0.8        ## use 4/5 the columns to decide each split decision
# )
# summary(gbm_model_1)
# (h2o.r2(gbm_model_1,train=T)) # 0.5707609
# (h2o.r2(gbm_model_1,valid=T)) # 0.539055
# #h2o.saveModel(gbm_model_1,"gbm_model_1")
# h2o.performance(gbm_model_1, train_model_h2o)
# h2o.performance(gbm_model_1, valid_model_h2o)
# 
# 
# 
# 
# #change to factors in h20
# y <- "y"
# x <- colnames(train_model[,2:379, with=FALSE])
# gbm_model_1 <- h2o.gbm(training_frame = train_model_h2o,      ## H2O frame holding the training data
#                        validation_frame = valid_model_h2o,  ## extra holdout piece for three layer modeling
#                        x=x,                 ## this can be names or column numbers
#                        y=y,                   ## target: using the logged variable created earlier
#                        model_id="gbm_model_1",              ## internal H2O name for model
#                        ntrees = 50,                  ## use fewer trees than default (50) to speed up training
#                        learn_rate = 0.05,             ## lower learn_rate is better, but use high rate to offset few trees
#                        max_depth = 3,
#                        score_tree_interval = 2,      ## score every 2 trees
#                        sample_rate = 0.5,            ## use half the rows each scoring round
#                        col_sample_rate = 0.8        ## use 4/5 the columns to decide each split decision
# )
# summary(gbm_model_1)
# (h2o.r2(gbm_model_1,train=T)) # 0.5995478
# (h2o.r2(gbm_model_1,valid=T)) # 0.5464069
# #h2o.saveModel(gbm_model_1,"gbm_model_1")
# h2o.performance(gbm_model_1, train_model_h2o)
# h2o.performance(gbm_model_1, valid_model_h2o)






#Score Test Data
test_h2o <- as.h2o(test, destination_frame = "test.hex")
h2o.loadModel("F:\\kaggle\\Mercedes-Benz\\gbm_model_1\\gbm_model_1") #gbm_model_1
gbm_model_1 <- h2o.getModel("gbm_model_1")
test_pred_y <- as.data.frame(h2o.predict(gbm_model_1,test_h2o))[,1]

#Create Submission File
submission_name <- "MB_JMH_Submission_1"  #Only need to change in this one place in future code versions
assign(paste(submission_name,sep=''), data.frame(ID=test$ID,y=test_pred_y))

#Check Submission File for NULLs and generate summary stats
length(get(submission_name)$y[is.na(get(submission_name)$y)]) #0
min(get(submission_name)$y) # 78.39783
max(get(submission_name)$y) # 119.852
mean(get(submission_name)$y) # 101.0064

#Output Submission File
write.csv(get(submission_name),paste(submission_name,'.csv',sep=''), row.names = FALSE)

#Leaderboard score = 0.55861
