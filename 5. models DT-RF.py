#import numpy as np
#import os
#import pandas as pd
#import h5py
#import tables
#from itertools import chain
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
#from pyspark.sql.functions import col, size, create_map, lit, concat, element_at, when, isnan, count
from pyspark.sql.types import *
from statistics import mean
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import CountVectorizer, Tokenizer, IDF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.clustering import LDA
#import pyLDAvis
#from time import time
#import numpy as np



spark = SparkSession.builder.master('yarn').config('spark.executor.memory', '16g').config('spark.driver.memory', '16g').appName('DT').getOrCreate()

spark.sparkContext.setLogLevel('ERROR')
sc = spark.sparkContext


out = open('hdfs:///user/averma/project/dt-rf-output.txt', 'w')
print('Reading in Data...', file=out)
df = spark.read.load('hdfs:///user/averma/project/input/MSD_labeled.parquet')
df = df.withColumn("avg_bars_start",df.avg_bars_start.cast(DoubleType()))
df = df.withColumn("avg_beats_start",df.avg_beats_start.cast(DoubleType()))
# df.show()


# # Indexing String features
#key_string distinct values: LA,RE,DO#,MI,LA#,SOL#,DO,FA,FA#,SI,RE#,SOL
#mode_string distinct values: maj,min
#tonality distinct values: DO min,SOL# maj...

# df.select('tonality').distinct().collect()


str_idx_list= ['key_string','mode_string','tonality','genre',"artist_id","artist_name", "release", "title","track_id"]
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in str_idx_list ]

pipeline = Pipeline(stages=indexers)
df_r = pipeline.fit(df).transform(df)

#drop string columns
df1 = df_r.drop('key_string','mode_string','tonality','genre',"artist_id","artist_name", "release", "title","track_id","artist_terms","artist_id_index","artist_name_index")
df1.show()


# # TF Artist_terms
vector = CountVectorizer(inputCol='artist_terms', outputCol='count_vectors')
count_model = vector.fit(df1)
data_cv = count_model.transform(df1).select("count_vectors","artist_familiarity", "artist_hotttnesss", "duration", "end_of_fade_in", "key", "loudness","mode", "start_of_fade_out", "tempo", "time_signature","density", "fadedness", "variability", "avg_bars_start", "avg_beats_start", "key_string_index","mode_string_index", "tonality_index","artist_id_index","artist_name_index", "release_index","title_index","track_id_index","genre_index")
tfidf = IDF(inputCol='count_vectors', outputCol='TF_features')
tfidf_model = tfidf.fit(data_cv)
df_tf = tfidf_model.transform(data_cv).select("artist_familiarity", "artist_hotttnesss", "duration", "end_of_fade_in", "key", "loudness","mode", "start_of_fade_out", "tempo", "time_signature","density", "fadedness", "variability", "avg_bars_start", "avg_beats_start", "key_string_index","mode_string_index", "tonality_index","artist_id_index","artist_name_index", "release_index","title_index","track_id_index",'TF_features',"genre_index")


# # Vector assembler
# except artist_id","artist_name", "release", "title","track_id", "artist_terms"
assembler = VectorAssembler().setInputCols(["artist_familiarity", "artist_hotttnesss", "duration", "end_of_fade_in",
                                            "key", "loudness", "mode", "start_of_fade_out", "tempo", "time_signature",
                                            "density", "fadedness", "variability", "avg_bars_start", "avg_beats_start",
                                            "key_string_index", "mode_string_index", "tonality_index",
                                            "artist_name_index", "release_index"]).setOutputCol("features")

#  "title_index","track_id_index","artist_id_index","artist_name_index"
df_vec = assembler.transform(df1)#.select("genre_index")#, "features","artist_id","artist_name", \
#                                          "release", "title","track_id", "artist_terms")
# df_vec.printSchema()
df_vec = df_vec.withColumnRenamed('genre_index','label')

train_df,test_df = df_vec.randomSplit([0.8, 0.2])


# # Decision Tree
print("Classifier: Decision Trees", file=out)

folds=10
dt_model = DecisionTreeClassifier(labelCol="label", featuresCol="features",maxBins=7794)
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
param_grid = ParamGridBuilder().addGrid(dt_model.maxDepth, [10, 15, 20]).addGrid(dt_model.minInstancesPerNode, [25, 50, 75]).build()

cv_model_dt = CrossValidator(estimator=dt_model,estimatorParamMaps=param_grid,evaluator=evaluator,numFolds=folds)
print("Fitting model...", file=out)
cv_model_dt = cv_model_dt.fit(train_df)

best_dt_model = cv_model_dt.bestModel# Get the best model after cross validating
print("\nBest model selected from cross validation:", cv_model_dt.bestModel, file=out)

print("Evaluating on train and test data..", file=out)
train_pred_dt = best_dt_model.transform(train_df)
test_pred_dt = best_dt_model.transform(test_df)

accuracy_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",metricName='accuracy')
f1_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",metricName='f1') #default
train_accuracy = accuracy_evaluator.evaluate(train_pred_dt)
print("Accuracy on train data",train_accuracy, file=out)
train_f1 = f1_evaluator.evaluate(train_pred_dt)
print("F1 score on train data",train_f1, file=out)

test_accuracy = accuracy_evaluator.evaluate(test_pred_dt)
print("Accuracy on test data with known labels",test_accuracy, file=out)
test_f1 = f1_evaluator.evaluate(test_pred_dt)
print("F1 score on test data with known labels",test_f1, file=out)


# # Random Forest
print("\n\nClassifier: Random Forest", file=out)
folds=10
rf_model = RandomForestClassifier(labelCol="label",featuresCol="features")
evaluator = MulticlassClassificationEvaluator(labelCol="label",predictionCol="prediction", metricName="accuracy")
param_grid = ParamGridBuilder().addGrid(rf_model.maxDepth, [10, 15, 20]).addGrid(rf_model.numTrees, [10, 20, 30]).build()

cv_model_rf = CrossValidator(estimator=rf_model,estimatorParamMaps=param_grid,evaluator=evaluator,numFolds=folds)
print("Fitting Random Forest model and performing cross validation...", file=out)
cv_model_rf = cv_model_rf.fit(train_df)

best_rf_model = cv_model_rf.bestModel# Get the best model after cross validating
print("Best model selected from cross validation:\n", cv_model_rf.bestModel)

print("Evaluating on train and test data..", file=out)
train_pred_rf = best_rf_model.transform(train_df)
test_pred_rf = best_rf_model.transform(test_df)

accuracy_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",metricName='accuracy')
f1_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",metricName='f1') #default
train_accuracy = accuracy_evaluator.evaluate(train_pred_rf)
print("Accuracy on train data",train_accuracy, file=out)
train_f1 = f1_evaluator.evaluate(train_pred_rf)
print("F1 score on train data",train_f1, file=out)

test_accuracy = accuracy_evaluator.evaluate(test_pred_rf)
print("Accuracy on test data with known labels",test_accuracy, file=out)
test_f1 = f1_evaluator.evaluate(test_pred_rf)
print("F1 score on test data with known labels",test_f1, file=out)
