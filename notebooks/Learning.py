# Databricks notebook source
# MAGIC %md
# MAGIC #Learning Model

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Imports

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
from elasticsearch import Elasticsearch, helpers
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reading the Data

# COMMAND ----------

# MAGIC %md
# MAGIC Reading from the index pattern we have created using the kibana interface: "trump__covid-19_tweets_*". 
# MAGIC <br>
# MAGIC We manually made sure that the hashtags are loaded as array of strings as we saw at the workshow.

# COMMAND ----------

ES_HOST = 'dds2019s-1002.eastus.cloudapp.azure.com'
index='trump_covid-19_tweets_*'
es = Elasticsearch([{'host': ES_HOST}], timeout=60000)

if not es.indices.exists(index):
    raise Exception("Index doesn't exist!")

data =  spark.read\
            .format("org.elasticsearch.spark.sql")\
            .option("es.nodes.wan.only","true")\
            .option("es.port","9200")\
            .option("es.nodes",ES_HOST)\
            .option("pushdown", "true")\
            .option("es.read.field.as.array.include",  "hashtags")\
            .load(index)

# COMMAND ----------

data.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC In order to use the logistic regression model we need to transform the data so it will contain a 'label' column and a 'features' column.
# MAGIC <br>
# MAGIC To do so we have used a udf to label each tweet if it contatins any 'trump' related hashtags or not.
# MAGIC <br>
# MAGIC After labeling the tweets we've used vector assembler to organize the data in a label column and a feature column so that the logistic regression model could use it.

# COMMAND ----------

def hashtag_to_label(hashtags):
  for word in hashtags:
    if 'trump' in word.lower():
      return 1
  return 0

hashtag_to_label_udf = F.udf(hashtag_to_label, IntegerType())

model_df = data.select(F.col('favorite_count').alias('likes'), F.col('hashtags'), F.col('retweet_count').alias('retweets'))
model_df = model_df.withColumn('label', hashtag_to_label_udf(F.col('hashtags')))
assembler = VectorAssembler(
    inputCols=['likes', 'retweets'],
    outputCol='features')

final_df = assembler.transform(model_df).select('label', 'features')
display(final_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Undersampling

# COMMAND ----------

# MAGIC %md
# MAGIC Since the data is very biased towards the class of 'No trump hashtag' (class 0), we have used a technique called 'undersampling', meaning we took every tweet which contained 'trump' (class 1), counted how many of those we have and then taking an equal amount of random tweets from the second class.

# COMMAND ----------

major_df = final_df.filter(F.col("label") == 0)
minor_df = final_df.filter(F.col("label") == 1)
ratio = float(major_df.count())/float(minor_df.count())
sampled_majority_df = major_df.sample(False, 1/ratio, seed=1)
undersampled_df = sampled_majority_df.unionAll(minor_df)
undersampled_df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train model 

# COMMAND ----------

# MAGIC %md
# MAGIC Splitting the data to train and test sets, both normal dataset(no changes made) and the undersampled dataset.

# COMMAND ----------

trainNormal, testNormal = final_df.randomSplit([0.7, 0.3])
trainUnder, testUnder = undersampled_df.randomSplit([0.7, 0.3])

# COMMAND ----------

# MAGIC %md
# MAGIC Training both models: Normal model and undersampled model.

# COMMAND ----------

lr = LogisticRegression()
fittedLRNormal = lr.fit(trainNormal)
fittedLRUnder = lr.fit(trainUnder)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training Summary

# COMMAND ----------

# Print the coefficients and intercept for multinomial logistic regression
print("Coefficients with no transformation: \n" + str(fittedLRNormal.coefficientMatrix))
print("Intercept with no transformation: " + str(fittedLRNormal.interceptVector))
print("-"*6)
trainingSummaryNormal = fittedLRNormal.summary

# Print the coefficients and intercept for multinomial logistic regression
print("Coefficients with undersampling: \n" + str(fittedLRUnder.coefficientMatrix))
print("Intercept with undersampling: " + str(fittedLRUnder.interceptVector))
print("-"*6)
trainingSummaryUnder = fittedLRUnder.summary

accuracy = trainingSummaryNormal.accuracy
falsePositiveRate = trainingSummaryNormal.weightedFalsePositiveRate
truePositiveRate = trainingSummaryNormal.weightedTruePositiveRate
fMeasure = trainingSummaryNormal.weightedFMeasure()
precision = trainingSummaryNormal.weightedPrecision
recall = trainingSummaryNormal.weightedRecall
print("Measurments of normal training: Accuracy: %s\nFalse Positive Rate: %s\nTrue Positive Rate:  %s\nPrecision: %s\nRecall: %s\nF-measure: %s"
      % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))

accuracy = trainingSummaryUnder.accuracy
falsePositiveRate = trainingSummaryUnder.weightedFalsePositiveRate
truePositiveRate = trainingSummaryUnder.weightedTruePositiveRate
fMeasure = trainingSummaryUnder.weightedFMeasure()
precision = trainingSummaryUnder.weightedPrecision
recall = trainingSummaryUnder.weightedRecall

print("Measurments of undersampling training: Accuracy: %s\nFalse Positive Rate: %s\nTrue Positive Rate:  %s\nPrecision: %s\nRecall: %s\nF-measure: %s"
      % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predicting

# COMMAND ----------

predictionsNormal = fittedLRNormal.transform(testNormal) # predict Normal
display(predictionsNormal)

# COMMAND ----------

predictionsUnder = fittedLRUnder.transform(testUnder) # predict Undersampled
display(predictionsUnder)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualizing
# MAGIC Visualizing the proportion of true labeling and wrong labeling

# COMMAND ----------

def calcAccuracy(a,b):
  if a==b:
    return 'True Prediction'
  return 'Wrong Prediction'

acc_udf = F.udf(calcAccuracy, StringType())

NormalDFAcc = predictionsNormal.withColumn('final_res', acc_udf(F.col('prediction'), F.col('label')))
display(NormalDFAcc.groupby('final_res').count())

# COMMAND ----------

UnderDFAcc = predictionsUnder.withColumn('final_res', acc_udf(F.col('prediction'), F.col('label')))
display(UnderDFAcc.groupby('final_res').count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Food For Thought
# MAGIC As we can see from both of the models' accuracy visualization, the undersampled model predicted poorly (almost 40% wrong predictions) whilst the normal model predicted overwhelmingly well (almost 0% wrong predictions).
# MAGIC <br>
# MAGIC At this point we'va formed an hypothesis: Training the model on the entire dataset made the model very biased towards the "No Trump" class. Moreover, we can conclude that since the proportion of "Trump" class out of the whole data is extremely small, even if the model were to classify every tweet as a "No Trump" it would have a neglectable precentage of wrong predictions.
# MAGIC <br>
# MAGIC On the other hand, training the model on the undersampled data showed us an ugly truth: Either the algorithm we chose does not fit for this task or the features we chose fo this task are not a good indications of tweets having "Trump" related hashtags.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluting the model based on test sets

# COMMAND ----------

resultsNormalEval = fittedLRNormal.evaluate(testNormal)
resultsUNderEval = fittedLRUnder.evaluate(testUnder) 

# COMMAND ----------

accuracy = resultsNormalEval.accuracy
falsePositiveRate = resultsNormalEval.weightedFalsePositiveRate
truePositiveRate = resultsNormalEval.weightedTruePositiveRate
fMeasure = resultsNormalEval.weightedFMeasure()
precision = resultsNormalEval.weightedPrecision
recall = resultsNormalEval.weightedRecall
print("Accuracy based on normal dataset: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
      % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))

accuracy = resultsUNderEval.accuracy
falsePositiveRate = resultsUNderEval.weightedFalsePositiveRate
truePositiveRate = resultsUNderEval.weightedTruePositiveRate
fMeasure = resultsUNderEval.weightedFMeasure()
precision = resultsUNderEval.weightedPrecision
recall = resultsUNderEval.weightedRecall
print("Accuracy based on undersampling: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
      % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))

# COMMAND ----------

# MAGIC %md
# MAGIC After evaluating both models on the test sets we can see that our hypothesis is likely correct. <BR>
# MAGIC As our hypothesis indicated, the model trained on the entire dataset is very biased and the undersampled model works poorly for this sort of task.