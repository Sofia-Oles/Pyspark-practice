from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

raw_df = spark.read.option("header", "true").option("encoding", "UTF-8").csv("data/movies_metadata.csv")
raw_df2 = spark.read.option("header", "true").csv("data/ratings_small.csv")

raw_df.show()

raw_df2.show()

# investigate schema in tree format
raw_df.printSchema()

# count rows in dfs
raw_df.count(), raw_df2.count()

# check unique users

from pyspark.sql.functions import countDistinct, when, count, col, isnull, avg, aggregate

raw_df2.select(countDistinct("userId")).show()

# missing values

raw_df.select([count(when(isnull(c), c)).alias(c) for c in raw_df.columns]).show()

# rows of raw_df2 (ratings) where MovieId is determined

raw_df2.filter(raw_df2.movieId == 5).show()

# 1 - MovieProfile

mv_id_name = raw_df.select(raw_df.id.cast('int').alias('movieId'), raw_df.title.cast('string').alias('movieName'))


# mv_id_name.count()

# basic filtering for null values and valid movieName

def filtering(input_df):
    return input_df.filter(input_df.movieName.rlike('[a-zA-Z]+'))  # one or more alpha symbols to clean numbers


def filtering2(input_df):
    return input_df.filter((input_df.movieName.startswith("[") == False) & (mv_id_name.movieId.isNotNull()) & (
        mv_id_name.movieName.isNotNull()))


df_m = mv_id_name.transform(filtering).transform(filtering2)
df_m.count()

df_m.show()

# select from raw_df2 dataframe

mv_id_rate = raw_df2.select(raw_df2.movieId.cast('int').alias('movieId'),
                            raw_df2.rating.cast('double').alias('movieRating'))

df_u = mv_id_rate.filter((mv_id_rate.movieId.isNotNull()) & (mv_id_rate.movieRating.isNotNull()))

# df_u.count() 26024289

# movie_id_name inner join movie_id_rating
mv_id_name_rate = df_m.join(df_u, df_m.movieId == df_u.movieId, 'inner') \
    .select(df_m.movieId.alias('movieId'), df_m.movieName.alias('movieName'), df_u.movieRating) \
    .groupBy('movieId', 'movieName') \
    .agg(avg(df_u.movieRating).alias('movieRating'))

mv_id_name_rate.show()

movie_id_genres = raw_df.select(raw_df.id.cast('int').alias('movieId'), raw_df.genres.alias('genres'))


def filter_gen_id(input_df):
    return input_df.filter((input_df.movieId.isNotNull()) & (input_df.genres.isNotNull()))


movie_id_genres = movie_id_genres.transform(filter_gen_id)
movie_id_genres.count()

import pyspark.sql.functions as f

final_df = movie_id_genres.select('movieId', f.get_json_object('genres', '$[*].name').alias('genres'))

final_df.show()

from pyspark.sql.functions import col, concat_ws, split, regexp_replace

movie_fin = final_df.withColumn("genres", regexp_replace(col("genres"), '[\\[\\]\\"]', ""))

movie_fin.show()

movie_profile = movie_fin.join(mv_id_name_rate, movie_fin.movieId == mv_id_name_rate.movieId) \
    .select(mv_id_name_rate.movieId, mv_id_name_rate.movieName, \
            mv_id_name_rate.movieRating, movie_fin.genres)

# movie_profile.count()

movie_profile.show()

# 2 - UserProfile

from pyspark.sql.functions import col, min as min_, max as max_, from_utc_timestamp, from_unixtime

raw_df2 = raw_df2.withColumn('timestamp', from_unixtime(raw_df2.timestamp).alias("timestamp"))

raw_df2.show()

from pyspark.sql.functions import from_unixtime, to_timestamp

df_user_marks = raw_df2.select(raw_df2.userId.alias('userId'), raw_df2.rating, raw_df2.timestamp) \
    .groupBy('userId') \
    .agg(count(raw_df2.rating).alias('numberOfMarks'), \
         avg(raw_df2.rating).alias('avgMark'), \
         min_(raw_df2.timestamp).alias('firstMark'), \
         max_(raw_df2.timestamp).alias('lastMark'))

df_user_marks.count()

df_user_marks.orderBy(df_user_marks.numberOfMarks.desc()).show()

# count years from today to firstMark
from pyspark.sql import functions as f
from pyspark.sql import types as t

df_min_max = df_user_marks.withColumn('yearsSpent', f.datediff(f.current_date(), df_user_marks.firstMark) / 365.25)

# check correctness
df_min_max.filter(df_min_max.userId == 10).show()

df_min_max_avg = df_min_max.withColumn('avgTimeBetweenMarks', f.datediff(df_min_max.lastMark, df_min_max.firstMark) / (
            df_min_max.numberOfMarks - 1))

# df_min_max_avg.orderBy(df_min_max_avg.avgTimeBetweenMarks.desc()).show()

# find favourite:

# 1 Find string of all genres of movies watched by user.

from pyspark.sql.functions import sum as sum_

fav_df = raw_df2.join(movie_profile, raw_df2.movieId == movie_profile.movieId) \
    .select(raw_df2.userId, movie_profile.genres) \
    .groupBy(raw_df2.userId) \
    .agg(f.concat_ws(",", f.collect_list(movie_profile.genres)).alias('allGenres'))

fav_df.show()

# 2 convert string splitted by comma to list
fav_df_list = fav_df.select(fav_df.userId, split(fav_df.allGenres, ",").alias("genresArray")).drop("allGenres")
fav_df_list.show()

fav_df_list.count()

temp = (fav_df_list.withColumn("Dist", f.array_distinct("genresArray"))
        .withColumn("Counts", f.expr("""transform(Dist,x->
                           aggregate(genresArray,0,(acc,y)-> IF (y=x, acc+1,acc))
                                      )"""))
        .withColumn("Map", f.arrays_zip("Dist", "Counts")
                    ))
out = temp.withColumn("favoriteGenre",
                      f.expr("""element_at(array_sort(Map,(first,second)->
         CASE WHEN first['Counts']>second['Counts'] THEN -1 ELSE 1 END),1)['Dist']"""))
out.show()

user_profile = df_min_max_avg.join(out, df_min_max_avg.userId == out.userId) \
    .select(df_min_max_avg.userId, df_min_max_avg.numberOfMarks, \
            df_min_max_avg.yearsSpent, df_min_max_avg.avgMark, \
            df_min_max_avg.avgTimeBetweenMarks, out.favoriteGenre)

user_profile.show()

user_profile.count()

# !pip install pyspark==<compatible-spark-version>
# !pyspark --packages io.delta:delta-core_2.12:1.1.0 --conf "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension" --conf "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog"

# user_df.write.format("delta").mode("append").save("/tmp/delta/UserProfile")
# user_df.write.format("delta").mode("append").saveAsTable("default.UserProfile")
