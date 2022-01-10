import numpy as np
import os
import pandas as pd
import h5py
import tables
from itertools import chain
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.functions import col, size, create_map, lit, concat, element_at, when, isnan, count
from pyspark.sql.types import *
from statistics import mean
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import CountVectorizer, Tokenizer, IDF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.clustering import LDA
import pyLDAvis
from time import time
import numpy as np


spark = SparkSession.builder.master('yarn').config('spark.executor.memory', '16g').config('spark.driver.memory', '16g').appName('LR').getOrCreate()

spark.sparkContext.setLogLevel('ERROR')
sc = spark.sparkContext


out = open('hdfs:///user/averma/project/lr-output.txt', 'w')

print('Reading in Data...', file=out)
df = spark.read.load('hdfs:///user/averma/project/input/MSD_labeled.parquet')
df = df.withColumn("avg_bars_start",df.avg_bars_start.cast(DoubleType()))
df = df.withColumn("avg_beats_start",df.avg_beats_start.cast(DoubleType()))
# df.show()


# genre_labels = spark.read.option("header", "true").option("delimiter","\t").csv('hdfs:///user/averma/project/input/TU-wien-genre-labels.txt')
# df_labels = df.join(genre_labels,df.track_id == genre_labels.track_id, 'inner').drop(genre_labels.track_id)
# df_labels #3985 tracks have genre labels

# df.groupBy('genre').count().show()


# # Indexing String features

#key_string distinct values: LA,RE,DO#,MI,LA#,SOL#,DO,FA,FA#,SI,RE#,SOL
#mode_string distinct values: maj,min
#tonality distinct values: DO min,SOL# maj...

# df.select('tonality').distinct().collect()



#string indexer
str_idx_list= ['key_string','mode_string','tonality','genre',"artist_id","artist_name", "release","title","track_id"]
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in str_idx_list ]

pipeline = Pipeline(stages=indexers)
df_r = pipeline.fit(df).transform(df)

#drop string columns
df1 = df_r.drop('key_string','mode_string','tonality','genre',"artist_id","artist_name", "release", "title","track_id")
# df1.show()


# # TF Artist_terms


#tokenize ther artist terms
# tokenizer = Tokenizer(inputCol='artist_terms', outputCol='term_tokens')
# df1 = tokenizer.transform(df1)  #term tokens is an array of strings now

vector = CountVectorizer(inputCol='artist_terms', outputCol='count_vectors')
count_model = vector.fit(df1)
data_cv = count_model.transform(df1).select("count_vectors","artist_familiarity", "artist_hotttnesss", "duration","end_of_fade_in", "key", "loudness","mode", "start_of_fade_out", "tempo", "time_signature","density", "fadedness", "variability", "avg_bars_start", "avg_beats_start", "key_string_index","mode_string_index", "tonality_index","artist_id_index","artist_name_index", "release_index","title_index","track_id_index","genre_index")
tfidf = IDF(inputCol='count_vectors', outputCol='TF_features')
tfidf_model = tfidf.fit(data_cv)
df2 = tfidf_model.transform(data_cv).select("artist_familiarity", "artist_hotttnesss", "duration","end_of_fade_in", "key", "loudness","mode", "start_of_fade_out", "tempo", "time_signature","density", "fadedness", "variability", "avg_bars_start", "avg_beats_start", "key_string_index","mode_string_index", "tonality_index","artist_id_index","artist_name_index", "release_index","title_index","track_id_index",'TF_features',"genre_index")


# # Vector assembler & Logistic Regression

# except artist_id","artist_name", "release", "title","track_id", "artist_terms"
assembler = VectorAssembler().setInputCols(["artist_familiarity", "artist_hotttnesss","duration", "end_of_fade_in", "key", "loudness","mode", "start_of_fade_out", "tempo", "time_signature","density", "fadedness", "variability", "avg_bars_start", "avg_beats_start","key_string_index", "mode_string_index", "tonality_index","artist_id_index","artist_name_index", "release_index", "title_index","track_id_index"]).setOutputCol("features")
df_vec = assembler.transform(df1)#.select("genre_index", "features","artist_id","artist_name", \
#                                          "release", "title","track_id", "artist_terms")
df_vec = df_vec.withColumnRenamed('genre_index','label')

train_df,test_df = df_vec.randomSplit([0.8, 0.2])

print("Classifier: Logistic Regression", file=out)

folds=10
lr_model = LogisticRegression(labelCol="label", featuresCol="features",maxIter=20, regParam=0.05, elasticNetParam=0)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
param_grid = ParamGridBuilder().addGrid(lr_model.maxIter, [50,100]).addGrid(lr_model.regParam,[0.5,0.05,0.005,0.0005]).build()

cv_model_lr = CrossValidator(estimator=lr_model,estimatorParamMaps=param_grid,evaluator=evaluator,numFolds=folds)
print("Fitting model...", file=out)
cv_model_lr = cv_model_lr.fit(train_df)

best_lr_model = cv_model_lr.bestModel# Get the best model after cross validating
print("\nBest model selected from cross validation:", cv_model_lr.bestModel, file=out)

print("Evaluating on train and test data..", file=out)
train_pred_lr = best_lr_model.transform(train_df)
test_pred_lr = best_lr_model.transform(test_df)

accuracy_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",metricName='accuracy')
f1_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",metricName='f1') #default
train_accuracy = accuracy_evaluator.evaluate(train_pred_lr)
print("Accuracy on train data",train_accuracy, file=out)
train_f1 = f1_evaluator.evaluate(train_pred_lr)
print("F1 score on train data",train_f1, file=out)

test_accuracy = accuracy_evaluator.evaluate(test_pred_lr)
print("Accuracy on test data with known labels",test_accuracy, file=out)
test_f1 = f1_evaluator.evaluate(test_pred_lr)
print("F1 score on test data with known labels",test_f1, file=out)
