# Databricks notebook source
# MAGIC %md
# MAGIC #Extract Transform Load

# COMMAND ----------

# MAGIC %md
# MAGIC This notebook will present the actions we executed to extract raw tweets from Kafka, transform and enrich (hydrate) them using Tweepy, and load them to Elasticsearch. <BR>
# MAGIC Further explanations about date selection, index creation and data transformations is presented in the attached PDF.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Imports

# COMMAND ----------

import tweepy
from elasticsearch import Elasticsearch, helpers
from pyspark.sql.types import *
import pyspark.sql.functions as F
import pickle
import kafka
from pyspark import SparkConf

# COMMAND ----------

# MAGIC %md
# MAGIC ##Twitter Api Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC We requested Twitter for a "consumer key" which allows us to hydrate the tweets we extract from Kafka.

# COMMAND ----------

consumer_key = 'nT2y41biy35qNY8CsH95tBggi'
consumer_secret = 'b2axleaFed1GxDkSHU4dudPzmEez5fo3uZc6qvrUUwrLk5ttTE'
access_token = '1288512030165684224-CGyAOLR7MWKtL7qdze7qA9zvHwKlPw'
access_token_secret = 'Apnef4M0X0Y0PvuT5TT6c0kf7DjlqAVaSICpIawJ9pTFY'
consumer = kafka.KafkaConsumer(bootstrap_servers=["ddkafka.eastus.cloudapp.azure.com:9092"])

# COMMAND ----------

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Elasticsearch Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC Similar to what we've seen in the workshop, we create an Elasticsearch index only this time with extra settings to properly collect coordination data. <BR>
# MAGIC To maximize loading speed to the server, we used several notebooks. Each of them ETL different topics, thus the index name presented here. <BR>
# MAGIC As explained earlier, more information about that is presented in the PDF file.

# COMMAND ----------

spark.conf.set("spark.sql.session.timeZone", "UTC")

ES_HOST = 'dds2019s-1002.eastus.cloudapp.azure.com'
es = Elasticsearch([{'host': ES_HOST}], timeout=60000)

dbutils.fs.rm("/tmp/Dvir/Stream/", True)
dbutils.fs.mkdirs("/tmp/Dvir/Stream/")

index = 'trump_covid-19_tweets_11-03-2020_16-03-2020_2nd_try'
if es.indices.exists(index): # Delete if exists
  es.indices.delete(index=index)

settings = {
  "settings" : {
      "number_of_shards" : 1,
      "number_of_replicas": 0,
      "refresh_interval" : -1
  },
  "mappings" : {
    "properties" : {
      "created_at" : {
        "type" : "date"
      },
      "coordinates" : {
        "properties" : {
          "coordinates" : {
            "type" : "geo_point"
          }
        }
      }
    }
  }
}
es.indices.create(index=index, ignore=400, body=settings)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Extract From Kafka

# COMMAND ----------

raw_stream_df = spark.readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", "ddkafka.eastus.cloudapp.azure.com:9092") \
                .option("subscribe", "11-03-2020, 16-03-2020") \
                .option("startingOffsets", "earliest") \
                .option("maxOffsetsPerTrigger", "100")\
                .load()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Transform

# COMMAND ----------

schema = StructType() \
        .add("tweet_id", LongType(), False) \
        .add("user_id", LongType(), False) \
        .add("date", StringType(), True) \
        .add("keywords", ArrayType(StringType(), True), True) \
        .add("location", MapType(StringType(), StringType(), True), True)

# COMMAND ----------

json_df = raw_stream_df.selectExpr("CAST(value AS STRING)")\
                       .select(F.from_json(F.col("value"), schema= schema).alias('json'))\
                       .select("json.*")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Hydrate tweets, transform them and load to Elasticsearch

# COMMAND ----------

# MAGIC %md
# MAGIC Tuning the tweet schema to correctly receive the data we want.

# COMMAND ----------

tweet_schema = pickle.load(open('/dbfs/mnt/tweet_schema.pkl', 'rb'))
new_schema = StructType()
new_schema.add(StructField("created_at",DateType(),True))
index_list=[1, 4, 7, 22, 27, 28]
for i in index_list:
  new_schema.add(tweet_schema[i])

# COMMAND ----------

# MAGIC %md
# MAGIC To hydrate the tweets, we used Tweepy api. Twitter restrict the amount of tweets that can be hydrated in a period of time so we collect the data in batches of 100 tweets at a time and work on each batch seperately. <BR>
# MAGIC For each batch we use the twitter id attribute, extracted from Kafka, in the "statuses_lookup" function which returns the hydrated tweets. <BR>
# MAGIC Next we select and transform the data according to our plans and then load them to Elasticsearch.

# COMMAND ----------

def foreach_batch_function(df, epoch_id):
  # Transform and write batchDF
  tweets_id_list = [str(tweet.tweet_id) for tweet in df.collect()]
  tweets_list = api.statuses_lookup(id_=tweets_id_list, map=False)
  df_tweets = sqlContext.createDataFrame(tweets_list, schema=new_schema)
  df_tweets = df_tweets.select(F.col('created_at'), F.col('entities.hashtags.text').alias('hashtags'), 'coordinates', 'quote_count', 'reply_count', 'retweet_count', 'favorite_count') \
                       .withColumn('created_at', F.to_timestamp(F.col('created_at'), "EEE MMM dd HH:mm:ss ZZZZ yyyy"))
                       
  df_tweets.write \
    .format("org.elasticsearch.spark.sql") \
    .option("es.nodes.wan.only", "true") \
    .option("es.resource", index) \
    .option("es.nodes", ES_HOST) \
    .option("es.port","9200") \
    .option("es.nodes.client.only", "false") \
    .mode("append") \
    .save()
  pass

# COMMAND ----------

write_df = json_df.writeStream \
       .foreachBatch(foreach_batch_function) \
       .start()