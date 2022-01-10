#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
findspark.init()


# In[2]:


import pandas as pd
import os
import glob
import h5py
import tables
from pyspark.sql import SparkSession
from pyspark.sql.types import *


# In[3]:


spark = SparkSession.builder.master('local').config('spark.executor.memory', '16g')     .config('spark.driver.memory', '16g').appName('Importing MSD').getOrCreate()

spark.sparkContext.setLogLevel('ERROR')
sc = spark.sparkContext


# In[4]:


def read_hdf5(path):
    store = pd.HDFStore(path, mode='r') # set up HDF5 store
    # get features from folders
    metadata = store.get("metadata/songs")
    analysis = store.get("analysis/songs")
    year = store.get("musicbrainz/songs")

    with h5py.File(path, 'r') as h5:
        # extract features
        artist_terms = pd.Series([[item.decode() for item in list(h5.get("metadata/artist_terms"))]], name="artist_terms")
        bars_start = pd.Series([[float(item) for item in list(h5.get("/analysis/bars_start"))]], name="bars_start")
        beats_start = pd.Series([[float(item) for item in list(h5.get("/analysis/beats_start"))]], name="beats_start")
        sections_start = pd.Series([[float(item) for item in list(h5.get("/analysis/sections_start"))]], name="sections_start")
        segments_start = pd.Series([[float(item) for item in list(h5.get("/analysis/segments_start"))]], name="segments_start")
        tatums_start = pd.Series([[float(item) for item in list(h5.get("/analysis/tatums_start"))]], name="tatums_start")

        # merge the features
        song = pd.concat([metadata, analysis, year, artist_terms, bars_start,
                          beats_start, sections_start, segments_start, tatums_start], axis=1, join='outer')

        # delete irrelevant columns
        # note that the genre column is completely empty since the dataset does not provide it
        song = song.drop(
            columns=["song_id", "genre", "idx_artist_terms", "idx_similar_artists", "audio_md5", "idx_bars_confidence",
                     "idx_bars_start", "idx_beats_confidence", "idx_beats_start", "idx_sections_confidence",
                     "idx_sections_start", "idx_segments_confidence", "idx_segments_loudness_max",
                     "idx_segments_loudness_max_time", "idx_segments_loudness_start", "idx_segments_pitches",
                     "idx_segments_start", "idx_segments_timbre", "idx_tatums_confidence", "idx_tatums_start",
                     "time_signature_confidence", "analyzer_version", "key_confidence", "mode_confidence",
                     "idx_artist_mbtags", "track_7digitalid", "release_7digitalid", "artist_playmeid", "artist_mbid",
                     "artist_7digitalid", "analysis_sample_rate"])

        return song.values.tolist()


# In[5]:


def import_data(path):
    r = read_hdf5(path)
    tables.file._open_files.close_all()
    return r


# In[6]:


schema = StructType([
    StructField("artist_familiarity", DoubleType(), True),
    StructField("artist_hotttnesss", DoubleType(), True),
    StructField("artist_id", StringType(), True),
    StructField("artist_latitude", DoubleType(), True),
    StructField("artist_location", StringType(), True),
    StructField("artist_longitude", DoubleType(), True),
    StructField("artist_name", StringType(), True),
    StructField("release", StringType(), True),
    StructField("song_hotttnesss", DoubleType(), True),
    StructField("title", StringType(), True),
    StructField("danceability", DoubleType(), True),
    StructField("duration", DoubleType(), True),
    StructField("end_of_fade_in", DoubleType(), True),
    StructField("energy", DoubleType(), True),
    StructField("key", LongType(), True),
    StructField("loudness", DoubleType(), True),
    StructField("mode", LongType(), True),
    StructField("start_of_fade_out", DoubleType(), True),
    StructField("tempo", DoubleType(), True),
    StructField("time_signature", LongType(), True),
    StructField("track_id", StringType(), True),
    StructField("year", LongType(), True),
    StructField("artist_terms", ArrayType(StringType()), True),
    StructField("bars_start", ArrayType(DoubleType()), True),
    StructField("beats_start", ArrayType(DoubleType()), True),
    StructField("sections_start", ArrayType(DoubleType()), True),
    StructField("segments_start", ArrayType(DoubleType()), True),
    StructField("tatums_start", ArrayType(DoubleType()), True)])


# In[108]:


print('Importing Data from HDF5...')
# print(os.walk('MSD'))
files = list(os.walk('MSD'))
# print(files)
files1 = files[3:]
# print(files1)
all_files=[]
for file_list in files1:
#     print ("file list",file_list)
    path=file_list[0]
    file_list2 = file_list[2]

    all_files.extend([os.path.join(path,f)for f in file_list2])
#     break
print(all_files)


# In[109]:


rdd = sc.parallelize(all_files).flatMap(lambda path: import_data(path))
rdd.collect()


# In[110]:


# Create dataframe
df = spark.createDataFrame(rdd, schema)
df.write.save('MSD_me.parquet')

