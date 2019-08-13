######################################################################################
# Sentiment Analysis towards Smartphone                                              #
#                                                                                    #
# This project is about investigating the sentiment towards smartphones              #
# for the use of medical app for aid workers in developing country.                  #           
# The data was collected using Amazon Web Services.
#                                                                                    #
# Version 1.0                                                                        #
#                                                                                    #            
# Date 13.08.2019                                                                    #
#                                                                                    #
# Mizuki Kakei                                                                       #
#                                                                                    #
#                                                                                    #     
######################################################################################

# Calling Packages ####
library(readr)                                                                       # Reading data set
library(caret)                                                                       # Prediction
library(doParallel)                                                                  # Providing a parallel backend
library(corrplot)                                                                    # Correlation matrix format
library(C50)                                                                         # C5.0 algorythm
library(inum)                                                                        # representation of vectors and intervals
library(plyr)                                                                        # Breaking big problems down into managable pieces and bringing back together
library(ggplot2)                                                                     # Visualization
library(plotly)                                                                      # Visualization
library(cowplot)                                                                     # Visualization


# Sentiment Analysis towards Iphone ####
# Reading Small Matrix (Train data) for iphone ####
RawData_Iphone <- read.csv("iphone_smallmatrix_labeled_8d.csv", 
                           TRUE, 
                           sep =","
                          )
Iphone_Train <- RawData_Iphone


# Reading Large Matrix (Validation data) ####
RawData_LargeMatrix <- read.csv("LargeMatrix.csv", 
                                TRUE, 
                                sep =","
                               )
LargeMatrix <- RawData_LargeMatrix


## Feature Selection ####
# Setting Iphone_Train data as data frame
Iphone_Train.df <- as.data.frame(Iphone_Train)

# Finding Near Zero Variances
nzv <- nearZeroVar(Iphone_Train.df , 
                   saveMetrics = FALSE
                  ) 

# Creating a new data set without near zero variance
Iphone_No_NZV <- Iphone_Train.df[ , -nzv ]


# Checking the correlation coefficient of Data set ####
Iphone_No_NZV.correlation <- cor(Iphone_No_NZV)
Iphone_No_NZV.correlation.plot <- corrplot(Iphone_No_NZV.correlation, 
                                           method = "number", 
                                           type = "upper",
                                           tl.cex = 0.6,   
                                           number.cex = 0.6
                                          )


# Matching the columns of LargeMatrix with SmallMatrix ####
# Setting Large Matrix as data frame
LargeMatrix <- as.data.frame(LargeMatrix)

# Preprocessing data for matching
Iphone_No_NZV_ForMatch  <- Iphone_No_NZV
Iphone_No_NZV_ForMatch$iphonesentiment <- NULL 

# Matching the columns of LargeMatrix with Small Matrix
LargeMatrix_Iphone_Match <- LargeMatrix[ , match(colnames(Iphone_No_NZV_ForMatch), 
                                                 colnames(LargeMatrix)
                                                )
                                       ]


## Modeling ####
# Changing Data type
Iphone_No_NZV$iphonesentiment <- as.factor(Iphone_No_NZV$iphonesentiment) 

# Sampling the training data set
set.seed(123)
Iphone_No_NZV_Sample <- Iphone_No_NZV[sample(1:nrow(Iphone_No_NZV), 
                                             nrow(Iphone_No_NZV), 
                                             replace = FALSE
                                            ),
                                     ]


# Splitting Data into Training and Testing Data set
inTraining <- createDataPartition(Iphone_No_NZV_Sample$iphonesentiment, 
                                  p = .7, 
                                  list = FALSE 
                                 )
training <- Iphone_No_NZV_Sample[inTraining, ]
testing <- Iphone_No_NZV_Sample[-inTraining, ]

# fixing the connetion
registerDoSEQ()

# Setting cross validation within training set
fitControl <- trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 1
                          )

# Modeling with RandomForest ####
rfGrid <- expand.grid(mtry=c(2))
rfFitModel_Iphone <- train(iphonesentiment~.,
                           data = training,
                           method = "rf",
                           trControl = fitControl,
                           tuneGrid = rfGrid
                          )

# Prediction of Iphone sentiment ####
# Prediction within SmallMatrix
predictionIphone_SmallMatrix <- predict(rfFitModel_Iphone, testing)

# Changing data types
predictionIphone_SmallMatrix <- as.factor(predictionIphone_SmallMatrix)
testing$iphonesentiment <- as.factor(testing$iphonesentiment)

# Checking the accuracy
postResample(predictionIphone_SmallMatrix, testing$iphonesentiment)

# Prediction with LargeMatrix
PredictionIphone_LargeMatrix <- predict(rfFitModel_Iphone, LargeMatrix_Iphone_Match)


# Visualization ####
# Creating Matrix to conbined prediction based on LargeMatrix and SmallMatrix
PredictionIphone_LargeMatrix <- as.matrix(PredictionIphone_LargeMatrix)
IphonesentimentSmall <- as.matrix(Iphone_No_NZV$iphonesentiment)

# Combining Matrix 
CombinedIphoneSent <- rbind(PredictionIphone_LargeMatrix, IphonesentimentSmall)

# Changing the data frame for visualization
CombinedIphoneSent <- as.data.frame(CombinedIphoneSent)

# Changing the name of the columns
colnames(CombinedIphoneSent) <- c("Iphonesentiment")

# Drawing a bargraph
BargraphIphoneSent <- ggplot(CombinedIphoneSent, aes(x = Iphonesentiment, binwidth = 5,  
                                                main = "Histogram for Age", 
                                                xlab = "Age",  
                                                fill=I("#00AFBB"), 
                                                col=I("#0072B2")
                                               )
                        ) + geom_bar()



## Sentiment Analysis towards Galaxy ####
# Reading Small Matrix (Train data) for Galaxy ####
RawData_Galaxy <- read.csv("galaxy_smallmatrix_labeled_8d.csv",
                          TRUE,
                          sep =","
                         )
Galaxy_Train <- RawData_Galaxy

# Reading Large Matrix (Validation data) for Galaxy ####
RawData_LargeMatrix <- read.csv("LargeMatrix.csv", TRUE, sep =",")
LargeMatrix <- RawData_LargeMatrix


## Feature Selection ####
# Setting Galaxy_Train data as data frame
Galaxy_Train.df <- as.data.frame(Galaxy_Train)

# Finding Near Zero Variances
nzv <- nearZeroVar(Galaxy_Train.df, saveMetrics = FALSE) 
nzv

# Creating a new data set without near zero variance
Galaxy_No_NZV <- Galaxy_Train.df[,-nzv]

# Checking the correlation coefficient of Data set ####
Galaxy_No_NZV.correlation <- cor(Galaxy_No_NZV)
Galaxy_No_NZV.correlation.plot <- corrplot(Galaxy_No_NZV.correlation, 
                                           method = "number", 
                                           type = "upper",
                                           tl.cex = 0.6,   
                                           number.cex = 0.6
)


# Matching the columns of LargeMatrix with SmallMatrix ####
# Setting Large Matrix as data frame
LargeMatrix <- as.data.frame(LargeMatrix)

# Preprocessing data for matching

Galaxy_No_NZV_ForMatch <- Galaxy_No_NZV
Galaxy_No_NZV_ForMatch$galaxysentiment <- NULL

LargeMatrix_Galaxy_Match <- LargeMatrix[ , match(colnames(Galaxy_No_NZV_ForMatch), 
                                                 colnames(LargeMatrix)
                                                 )
                                       ]


## Modeling ####
# Changing Data type
Galaxy_No_NZV$galaxysentiment <- as.factor(Galaxy_No_NZV$galaxysentiment)

# Sampling training data set
set.seed(123)
Galaxy_No_NZV_Sample <- Galaxy_No_NZV[sample(1:nrow(Galaxy_No_NZV), 
                                    nrow(Galaxy_No_NZV),
                                    replace=FALSE
                                   ),
                            ]

# Splitting Data into Training and Testing Data set
inTraining <- createDataPartition(Galaxy_No_NZV_Sample$galaxysentiment, 
                                  p = .7, 
                                  list = FALSE 
                                 )
training <- Galaxy_No_NZV_Sample[inTraining, ]
testing <- Galaxy_No_NZV_Sample[-inTraining, ]

# fix the connetion
registerDoSEQ()

# Setting cross validation within training set
fitControl <- trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 1
                           )

# Modeling with RandomForest ####
rfGrid <- expand.grid(mtry = c(2))
rfFitModel_Galaxy <- train(galaxysentiment~ .,
                           data = training,
                           method = "rf",
                           trControl = fitControl,
                           tuneGrid = rfGrid
                          )

# Prediction of Galaxy sentiment ####
# Prediction within SmallMatrix
PredictionGalaxy_SmallMatrix <- predict(rfFitModel_Galaxy, testing)

# Changing data types
PredictionGalaxy_SmallMatrix <- as.factor(PredictionGalaxy_SmallMatrix)
testing$galaxysentiment <- as.factor(testing$galaxysentiment)

# Checking the accuracy
postResample(PredictionGalaxy_SmallMatrix, testing$galaxysentiment)

# Predition with LargeMatrix
PredictionGalaxy_LargeMatrix <- predict(rfFitModel_Galaxy, LargeMatrix_Galaxy_Match)


# Visualization ####
# Creating Matrix to combine prediction based on LargeMatrix and SmallMatrix
PredictionGalaxy_LargeMatrix <- as.matrix(PredictionGalaxy_LargeMatrix)
GalaxysentimentSmall <- as.matrix(Galaxy_No_NZV$galaxysentiment)

# Combining Matrix
CombinedGalaxySent <- rbind(PredictionGalaxy_LargeMatrix, GalaxysentimentSmall)

# Changing the data frame for Visualization
CombinedGalaxySent <- as.data.frame(CombinedGalaxySent)

# Changing the name of the columns
colnames(CombinedGalaxySent) <- c("Galaxysentiment")

# Drawing a bargraph
bargraphGalaxy <- ggplot(CombinedGalaxySent, aes(x = Galaxysentiment, binwidth = 5,  
                                                      main = "Histogram for Age", 
                                                      xlab = "Age",  
                                                      fill=I("#FFDB6D"), 
                                                      col=I("#C4961A")
                                                 )
                         ) + geom_bar()

