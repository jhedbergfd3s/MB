
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


#Define R2 Function
R2 <- function(predicted, actual) {
  #sample_data_for_r2 <- data.frame(predicted=c(2,6.7143,11.4286,16.1429,20.8572,25.5715,30.2858),actual=c(8,8,8,9,15,30,35))
  y_bar <- mean(actual)
  yi_minus_ybar_squared <- c()
  yi_minus_yhat_squared <- c()
  for(i in seq.int(1,length(predicted),1)){
    yi_minus_ybar_squared[i] <- (actual[i]-y_bar)^2 
    yi_minus_yhat_squared[i] <- (actual[i]-predicted[i])^2
  }
  SStot <- sum(yi_minus_ybar_squared)
  SSres <- sum(yi_minus_yhat_squared)
  R2 <- 1-(SSres/SStot)
  rm(y_bar,yi_minus_ybar_squared,yi_minus_yhat_squared,i,SStot,SSres)
  return(R2)
  #R2(sample_data_for_r2$predicted,sample_data_for_r2$actual) #answer should be 0.77897
}




####Modeling Section
# h2o.init(nthreads = 6, max_mem_size = '10G') # uses 6 machine cores, 10G of RAM 
# h2o.shutdown(prompt = FALSE)
# Monitor at http://localhost:54321/flow/index.html

train_model_h2o <- as.h2o(train_model, destination_frame = "train_model.hex" )
valid_model_h2o <- as.h2o(valid_model, destination_frame = "valid_model.hex" )



x <- colnames(train_model[,2:379, with=FALSE])
pca_model_5 <- h2o.prcomp(
                  train_model_h2o,
                  x,
                  model_id = "pca_model_5",
                  validation_frame = valid_model_h2o,
                  ignore_const_cols = TRUE,
                  score_each_iteration = FALSE,
                  transform = "NONE", #c("NONE", "STANDARDIZE", "NORMALIZE", "DEMEAN", "DESCALE"),
                  pca_method = "GramSVD", #c("GramSVD", "Power", "Randomized", "GLRM")
                  k = 200,
                  max_iterations = 1000,
                  use_all_factor_levels = TRUE,
                  compute_metrics = TRUE,
                  impute_missing = FALSE,
                  seed = -1,
                  max_runtime_secs = 0)
#h2o.saveModel(pca_model_5,"pca_model_5")


temp <- as.data.frame(h2o.predict(pca_model_5,train_model_h2o))
temp2 <- as.data.frame(h2o.predict(pca_model_5,valid_model_h2o))

train_model <- cbind(train_model,temp)
valid_model <- cbind(valid_model,temp2)
rm(temp,temp2)


train_model_h2o <- as.h2o(train_model, destination_frame = "train_model.hex" )
valid_model_h2o <- as.h2o(valid_model, destination_frame = "valid_model.hex" )



y <- "y"
x <- colnames(train_model[,2:579, with=FALSE])
gbm_model_5 <- h2o.gbm(training_frame = train_model_h2o,      ## H2O frame holding the training data
                       validation_frame = valid_model_h2o,  ## extra holdout piece for three layer modeling
                       x=x,                 ## this can be names or column numbers
                       y=y,                   ## target: using the logged variable created earlier
                       model_id="gbm_model_5",              ## internal H2O name for model
                       ntrees = 2000,                  ## use fewer trees than default (50) to speed up training
                       learn_rate = 0.005,             ## lower learn_rate is better, but use high rate to offset few trees
                       max_depth = 3,
                       seed = 12345,
                       score_tree_interval = 10,      ## score every 3 trees
                       sample_rate = 0.9,            ## use half the rows each scoring round
                       col_sample_rate = 0.9,        ## use 4/5 the columns to decide each split decision
                       stopping_rounds = 2,
                       stopping_metric = "deviance", # "AUTO", "deviance", "logloss", "MSE", "RMSE", "MAE", "RMSLE", "AUC", "lift_top_group", "misclassification", "mean_per_class_error"
                       distribution = "gaussian", # "AUTO","bernoulli", "multinomial", "gaussian", "poisson", "gamma", "tweedie","laplace", "quantile", "huber")
                       categorical_encoding = "Enum" #"AUTO" "Enum" "OneHotInternal" "OneHotExplicit" "Binary" "Eigen"
                       
)
summary(gbm_model_5)
(h2o.r2(gbm_model_5,train=T)) # 0.5940161
(h2o.r2(gbm_model_5,valid=T)) # 0.5562581
#h2o.saveModel(gbm_model_5,"gbm_model_5")
#h2o.performance(gbm_model_5, train_model_h2o)
#h2o.performance(gbm_model_5, valid_model_h2o)
#path <- getwd()
#h2o.download_mojo(gbm_model_5, path=path)
# java -cp h2o.jar hex.genmodel.tools.PrintMojo --tree 0 --detail --levels 25 -i gbm_model_5.zip -o gbm_model_5.gv
# dot -Tpng gbm_model_5.gv -o gbm_model_5.png


valid_model$gbm_model_5 <- as.data.frame(h2o.predict(gbm_model_5,valid_model_h2o))[,1]
plot_ly(x=valid_model$y,y=valid_model$gbm_model_5,type="scatter", mode="markers") %>% add_trace(x=seq.int(80,120,1),y=seq.int(80,120,1), mode='lines') %>% layout(xaxis=list(title="Actual"),yaxis=list(title="Predicted-gbm_model_5"))
R2(valid_model$gbm_model_5,valid_model$y) # 0.5562581










#Check for scaling improvements
scale_multiplier_i <- c()
scale_results_i <- c()
for(i in seq.int(.99,1.01,0.0005)){
  print(paste('i=',i,'    ','multplier=',i,'    ','R2=',R2(i*valid_model$gbm_model_5,valid_model$y),sep=''))
  scale_multiplier_i <- c(scale_multiplier_i,i)
  scale_results_i <- c(scale_results_i,R2(i*valid_model$gbm_model_5,valid_model$y))
}
scale_results <- data.frame(scale_multiplier_i=scale_multiplier_i,scale_results_i=scale_results_i)
rm(scale_results_i,scale_multiplier_i,i)


valid_model$gbm_model_5 <- 0.9995*valid_model$gbm_model_5
R2(valid_model$gbm_model_5,valid_model$y) # 0.5562843



      #REFERENCE - gbm_model__1
      # y <- "y"
      # x <- colnames(train_model[,2:379, with=FALSE])
      # gbm_model_1 <- h2o.gbm(training_frame = train_model_h2o,      ## H2O frame holding the training data
      #                        validation_frame = valid_model_h2o,  ## extra holdout piece for three layer modeling
      #                        x=x,                 ## this can be names or column numbers
      #                        y=y,                   ## target: using the logged variable created earlier
      #                        model_id="gbm_model_1",              ## internal H2O name for model
      #                        ntrees = 900,                  ## use fewer trees than default (50) to speed up training
      #                        learn_rate = 0.004,             ## lower learn_rate is better, but use high rate to offset few trees
      #                        max_depth = 3,
      #                        score_tree_interval = 15,      ## score every 3 trees
      #                        sample_rate = 0.5,            ## use half the rows each scoring round
      #                        col_sample_rate = 0.8        ## use 4/5 the columns to decide each split decision
      # )
      # summary(gbm_model_1)
      # (h2o.r2(gbm_model_1,train=T)) # 0.5886679
      # (h2o.r2(gbm_model_1,valid=T)) # 0.553584
      # #h2o.saveModel(gbm_model_1,"gbm_model_1")
      # h2o.performance(gbm_model_1, train_model_h2o)
      # h2o.performance(gbm_model_1, valid_model_h2o)



#Score Test Data
test_h2o <- as.h2o(test, destination_frame = "test.hex")
h2o.loadModel("F:\\kaggle\\Mercedes-Benz\\pca_model_5\\pca_model_5") #pca_model_5
temp3 <- as.data.frame(h2o.predict(pca_model_5,test_h2o))
test <- cbind(test,temp3)
test_h2o <- as.h2o(test, destination_frame = "test.hex") #re-import after adding PCA features

h2o.loadModel("F:\\kaggle\\Mercedes-Benz\\gbm_model_5\\gbm_model_5") #gbm_model_5
gbm_model_5 <- h2o.getModel("gbm_model_5")
test_pred_y_gbm_model_5 <- as.data.frame(h2o.predict(gbm_model_5,test_h2o))[,1]


test_pred_y <- 0.9995*test_pred_y_gbm_model_5



#Create Submission File
submission_name <- "MB_JMH_Submission_5"  #Only need to change in this one place in future code versions
assign(paste(submission_name,sep=''), data.frame(ID=test$ID,y=test_pred_y))

#Check Submission File for NULLs and generate summary stats
length(get(submission_name)$y[is.na(get(submission_name)$y)]) #0
min(get(submission_name)$y) # 79.58483
max(get(submission_name)$y) # 118.3851
mean(get(submission_name)$y) # 100.9703

#Output Submission File
write.csv(get(submission_name),paste(submission_name,'.csv',sep=''), row.names = FALSE)

#Leaderboard score = 0.55299
