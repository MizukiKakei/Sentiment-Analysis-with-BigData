# Sentiment Analysis towards smartphone using Amazon Web Services

In this project, we analyse the sentiment towards smartphone for the use of medical apps for aid workers in developing countries. 
In order to do it in large scale, we used Amazon Web services which is one of the most widely used cloud services.

### Data set
In this analysis, Two different data sets were used. 
#### Small Data set
Given data set which includes 12000 instances. This data set was used for the modeling. 
The sentiments toward smartphones were filled manually. 
#### Large Data set
Large data set with 20000 instances were based on the collected data through AWS.
The sentiments toward smartphones were missing

For both data sets, the attributes includes : 
-the relevancy of the webpage toward each device, 
-the sentiment toward the operating system used on the phone, 
-the sentiment toward a phone’s camera, display, and performance. 


### Methods of analysis
Since the data set was quite big, we reduced the data near to zero variances. 
Then, the reduced data set was used for the modeling within small data set, and the model was applied to large data set to predict the sentiment. 
Randomforest was used for the modeling.
