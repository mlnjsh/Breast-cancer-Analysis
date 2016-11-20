# Read the data

bcdt<-read.csv("BCA.csv")

#Load dplyr package and shift Diagnosis to last position

library(dplyr)

bcdt<-bcdt %>% select(-Diagnosis, everything())

#Remove patiantID

bcdt<-bcdt[,-1]

# code Malignent to be 1 and Benign to be 0

bcdt$Diagnosis<-ifelse(bcdt$Diagnosis=="Malignant",1,0)

# Convert Diagnosis to factor variable 

bcdt$Diagnosis<-as.factor(bcdt$Diagnosis)

# Understand the structure of dataset

str(bcdt)

# Proportion of Malignent and Benign with package ggplot2 and scales

library(ggplot2)

library(scales)

ggplot(bcdt, aes(x = as.factor(Diagnosis))) +
  
  geom_bar(aes(y = (..count..)/sum(..count..))) +
  
  geom_text(aes(y = ((..count..)/sum(..count..)), 
                
                label = scales::percent((..count..)/sum(..count..))), 
            
            stat = "count", vjust = -0.25) +   scale_y_continuous(labels = percent) +
  
  labs(title = "Benign vs Malignent", y = "Percent", x = "Diagnosis")

# Missing value with VIM Package

library(VIM)

aggr_plot <- aggr(bcdt, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE,
                  
              labels=names(bcdt), cex.axis=.7, gap=3,
              
                    ylab=c("Histogram of missing data","Pattern"))

# Pre-Processing of data using caret

library(caret)

summary(bcdt)

prpc<-preProcess(bcdt[,c("Radius","Texture","Perimeter","Area","Area.SE",
                         
                         "Worst.Radius","Worst.Texture","Worst.Perimeter","Worst.Area")], 
                 
                 method = c("center", "scale", "YeoJohnson"))

bcdt<-predict(prpc, newdata = bcdt)

str(bcdt)

# Density Plots with caret with AppliedPredictiveModeling package

library(AppliedPredictiveModeling)

transparentTheme(trans = .9)

featurePlot(x = bcdt[, 1:10], 
            y = bcdt$Diagnosis,
            plot = "density", 
            ## Pass in options to xyplot() to 
            ## make it prettier
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5,
            pch = "|", 
            layout = c(5, 2), 
            auto.key = list(columns = 2))

featurePlot(x = bcdt[, 11:20], 
            y = bcdt$Diagnosis,
            plot = "density", 
            ## Pass in options to xyplot() to 
            ## make it prettier
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5,
            pch = "|", 
            layout = c(5, 2), 
            auto.key = list(columns = 2))

featurePlot(x = bcdt[, 21:30], 
            y = bcdt$Diagnosis,
            plot = "density", 
            ## Pass in options to xyplot() to 
            ## make it prettier
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")), 
            adjust = 1.5,
            pch = "|", 
            layout = c(5, 2), 
            auto.key = list(columns = 2))

# calculate correlation matrix

correlationMatrix <- cor(bcdt[,1:30])

# summarize the correlation matrix

print(correlationMatrix)

# find attributes that are highly corrected (ideally >0.75)

highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)

# print indexes of highly correlated attributes

print(highlyCorrelated)

# Divide datset into train and test set
library(caret)

set.seed(1)

inTraining <- createDataPartition(bcdt$Diagnosis, p = .75, list = FALSE)

train<-bcdt[ inTraining,]

test<-bcdt[-inTraining,]

# Feature Selection 

# ensure the results are repeatable

set.seed(7)

control <- rfeControl(functions=rfFuncs, method="cv", repeats = 10, number=10)

# run the RFE algorithm

results <- rfe(train[,1:30], train[,31], sizes=c(1:30), rfeControl=control)

# summarize the results

print(results)

# list the chosen features

predictors(results)

# plot the results

plot(results, type=c("g", "o"))

set.seed(8)

control <- trainControl(method="repeatedcv", number=10, repeats=3)

# train the model

model <- train(Diagnosis~., data=bcdt[,-1], method="lvq", preProcess="scale", trControl=control)

# estimate variable importance

importance <- varImp(model, scale=FALSE)

# summarize importance

print(importance)

# plot importance

plot(importance)

# ensure the results are repeatable

set.seed(9)

# define the control using a random forest selection function

control <- rfeControl(functions=rfFuncs, method="cv", number=10,repeats = 10)

# run the RFE algorithm

results <- rfe(train[,1:30], train[,31], sizes=c(1:30), rfeControl=control)

# summarize the results

print(results)

# list the chosen features

predictors(results)

# plot the results

plot(results, type=c("g", "o"))


train<-train[,c("Worst.Area","Worst.Concave.Points","Worst.Radius","Worst.Perimeter",     
                       "Area.SE","Worst.Texture","Concave.Points","Texture",         
                       "Worst.Concavity","Area","Worst.Smoothness","Concavity",           
                       "Worst.Symmetry", "Radius","Radius.SE","Perimeter",         
                       "Perimeter.SE","Worst.Compactness","Diagnosis")] 

# Training and Tunning RF RRF and RRE

# Traing RF 

library(caret)

control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")

set.seed(123)

tunegrid <- expand.grid(.mtry=c(5:15))

rf_gridsearch <- train(Diagnosis~., data=train, method="rf",  
                       
                       tuneGrid=tunegrid, trControl=control)

print(rf_gridsearch)

plot(rf_gridsearch)

preds <- predict(rf_gridsearch, test)
test$Diagnosis<-as.factor(test$Diagnosis)
confusionMatrix(preds,test$Diagnosis)

Xtrain<-train[,-31]
ytrain<-train[,31]
Xtest<-test[,-31]
ytest<-test[,31]
parameterGrid <- expand.grid(K=round(ncol(Xtrain)/c(2,3), 0), L=c(10,20), 
                             
                             mtry=c(1, 3))

optTune <- findOptimalTuning(Xtrain, ytrain, k=5, paraGrid = parameterGrid)


fitcontrol <- trainControl(method="repeatedcv", number=10, repeats=3)

RFgrid <- expand.grid(.mtry=c(1:15), .ntree=c(1000, 1500, 2000, 2500))

CVerror <- kFoldRun(Xtrain, ytrain, k=10)

mod <- RRotF(Xtrain, ytrain, model="rnd",control=fitcontrol,tunegrid=RFgrid)

preds <- predict(mod, Xtest)
head(preds)
ytest$Diagnosis<-as.factor(ytest$Diagnosis)
head(ytest)
confusionMatrix(preds,ytest)

control <- trainControl(method="repeatedcv", number=10, repeats=3)
tunegrid <- expand.grid(.mtry=c(1:15))
rf_gridsearch <- train(Diagnosis~., data=train, method="rf",tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)
rfFit

Grid <-expand.grid(k=4,L=3)

fitControl<-trainControl(method ="repeatedcv",
                           number =10,
                           repeats =10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)
                          



test$Diagnosis<-ifelse(test$Diagnosis=="Malignent", 1,0)

test$Diagnosis<-as.factor(test$Diagnosis)

head(train$Diagnosis)
library(pROC)
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)

rtfFit <- train(Diagnosis~.,data=train,
                 method = "rotationForest", 
                 trControl = fitControl,
                 verbose = FALSE,
                 metric = "ROC")
rtfFit


preds <- predict(rtfFit, test)
test$Diagnosis<-as.factor(test$Diagnosis)
confusionMatrix(preds,test$Diagnosis)
plot(rtfFit)
modelLookup("rotationForest")
x<-train[,-31]
y<-train[,31]
library(rotationForest)

rotationForest(x, y, K = round(ncol(x)/3, 0), L = 10)

