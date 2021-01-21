"""
Snippet for using regular expressions to extract potential candidate tweets

spark-submit --master yarn --queue umsi-drom --num-executors 200 --driver-memory 30g --executor-memory 8g --conf spark.executor.extrajavaoptions="-Xmx16084m" extract_regex_phrases.py | tee logs/extract_regex_phrases.txt
"""
import os
from os.path import join
from time import time
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import col,udf,lower
from datetime import datetime

import re

conf = SparkConf().setAppName('extractRegex')
conf.set("spark.sql.files.ignoreCorruptFiles", "true")
conf.set("spark.driver.maxResultSize","0")
sc = SparkContext.getOrCreate(conf)
sqlContext = SQLContext(sc)


"""
Step 1: Set phrases for each regex
"""
# Death
S_death = ['passed away', 'death', 'passing (?:of|away)', 'died', 'loss of my', 'killed',
           'murder', 'suicide',
           'lost (?:his|her|their) (?:life|lives)']
# Break up / divorce
S_breakup = ['broken? up', 'break\s?up', 'dumped', 'divorce', 'separated with',
                 'end(?:ed)?(?: to)? (?:my|our|the|a)(?: \w+)\{0,2\} relationship','parted ways']
# Fired from a job
S_job = ['\bfired\b', 'laid off', 'furlough', '\blay\s?off', 'jobless', 'unemployed',
                 'lost (?:my|the)(?: \w+)\{,2\} (?:job|position)', '\blet go\b']
# Crime / legal
S_crime = ['robbed','assault','\bstole','mugged','\brape','discriminat','offended','harass','\bstalk',
           'fraud','lynch','victim','involved in a','\babus(?:ed|ive)','threaten',
          'broke into','\bhacked','damaged','\bsued\b','\btowed','accident\b','stalk','fraud','arrest','beat(?:en)? up']


str_death = r'%s' % '|'.join(S_death)
str_breakup = r'%s' % '|'.join(S_breakup)
str_job = r'%s' % '|'.join(S_job)
str_crime = r'%s' % '|'.join(S_crime)

reg_death = re.compile(str_death)
reg_crime = re.compile(str_crime)
reg_breakup = re.compile(str_breakup)
reg_job = re.compile(str_job)

# function for creating file names based on each month
def floor(x):
    if x>=10:
        return 1
    else:
        return 0

"""
Step 2: read through cavium files between 2019-01-01 to 2020-07-01
"""
out_list = []
for filename in sorted(os.listdir('/hadoop-fuse/data/twitter/decahose/parquet')):
    twitter_dir = '/data/twitter/decahose/parquet'
    if filename.endswith('.parquet') & ('decahose.2019' in filename):
        try:
            mn = filename.split('decahose.2019')[1].split('-')[1]
        except:
            continue
        if int(mn)<7:
            out_list.append((twitter_dir,filename,'relationships-shocks/tweetsByRegex/2019/%s'%mn))

for year,x in list([(2020,x) for x in range(1,7)]):
    twitter_dir = '/data/twitter/decahose/2020/json'
    for i in range(1,3):
        ### additional measure to make sure that 2020-02p2-2020-05p1 are excluded
        if (x<=4):
            continue
        elif (x==5)&(i==1):
            continue

        # if (x*2+i >= 3) & (x*2+i <= 11):
        #     continue
        file_name = 'decahose.%d-%d%d*.p%d.bz2.json'%(year,floor(x),x%10,i)
        folder = 'tweetsByRegex/2020/%d'%x
        out_list.append((twitter_dir, file_name,'relationships-shocks/tweetsByRegex/%d/%d'%(year,x)))

# for filename in sorted(os.listdir('/hadoop-fuse/data/twitter/decahose/2020/json')):
#     twitter_dir = '/data/twitter/decahose/2020/'
#     if filename.endswith('.json') & ('decahose.2020-02' in filename):
#         out_list.append((twitter_dir,filename,'tweetsByRegex/2020/2_1'))

# 2020.01-03 is inevitably collected using .bz2 as .json during that direction misses out a lot of tweets
for year,x in list([(2020,x) for x in range(1,4)]):
    for i in range(1,3):
        twitter_dir = '/data/twitter/decahose/2020'
        file_name = 'decahose.%d-%02d*.p%d*.bz2'%(year,x,i)
        out_list.append((twitter_dir, file_name,'relationships-shocks/tweetsByRegex/%d/%d_2'%(year,x)))


# for year,x in list([(2019,x) for x in range(1,7)]):
#     for i in range(1,3):
#         twitter_dir = '/data/twitter/decahose/parquet'
#         # twitter_dir = '/var/twitter/decahose/raw'
#         file_name = 'decahose.%d-%d%d*.p%d*.parquet'%(year,floor(x),x%10,i)
#         # file_name = 'decahose.%d-%d%d*.p%d*.json'%(year,floor(x),x%10,i)
#         out_list.append((twitter_dir, file_name,'tweetsByRegex/%d/%d'%(year,x)))

# for year,x in list([(2019,x) for x in range(7,13)]):
#     for i in range(1,3):
#         twitter_dir = '/var/twitter/decahose/raw'
#         file_name = 'decahose.%d-%d%d*.p%d.bz2'%(year,floor(x),x%10,i)
#         out_list.append((twitter_dir, file_name,'tweetsByRegex/%d/%d'%(year,x)))

# 2019-7~2019-12
# for year,x in list([(2019,x) for x in range(7,13)]):
#     for i in range(1,3):
#         twitter_dir = '/var/twitter/decahose/raw'
#         file_name = 'decahose.%d-%d%d*.p%d.bz2'%(year,floor(x),x%10,i)
#         out_list.append((twitter_dir, file_name))

# 2020-1~2020-7

# for year,x in list([(2020,x) for x in range(5,7)]):
#     twitter_dir = '/data/twitter/decahose/2020/json'
#     for i in range(0,4):
#         file_name = 'decahose.%d-%d%d-%d*.bz2.json'%(year,floor(x),x%10,i)
#         out_list.append((twitter_dir, file_name))

def time_fn(dtime):
    return str(datetime.strftime(datetime.strptime(dtime, '%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d'))

# function for selecting event category
def reg_fn(text):
    if text:
        if reg_death.search(text):
            return 'death'
        elif reg_crime.search(text):
            return 'crime'
        elif reg_breakup.search(text):
            return 'breakup'
        elif reg_job.search(text):
            return 'job-loss'
        else:
            return None
    else:
        return None



time_udf = udf(time_fn)
reg_udf = udf(reg_fn)

# print(out_list[:2])
# import sys
start = time()

for twitter_dir,file_name,folder in out_list:

    print("Reading file ",file_name,int(time()-start))
    try:
        if file_name.endswith('parquet'):
            df = sqlContext.read.parquet(join(twitter_dir, file_name))
        else:
            df = sqlContext.read.json(join(twitter_dir,file_name))
        print("Loaded file ",file_name,int(time()-start))

        # filter tweets that contain a retweet
        df_rt = df.filter('retweeted_status is not null')
        df_rt = df_rt.select(col('retweeted_status.user.id_str').alias('uid'),
                             col('retweeted_status.user.screen_name').alias('name'),
                             col('retweeted_status.id_str').alias('tid'),
                             col('created_at').alias('timestamp'),
                             col('retweeted_status.created_at').alias('timestamp2'),
                             col('retweeted_status.text').alias('text'),
                             col('retweeted_status.extended_tweet.full_text').alias('text2'))
        df_qt = df.filter('quoted_status is not null')
        df_qt = df_qt.select(col('quoted_status.user.id_str').alias('uid'),
                             col('quoted_status.user.screen_name').alias('name'),
                             col('quoted_status.id_str').alias('tid'),
                             col('created_at').alias('timestamp'),
                             col('quoted_status.created_at').alias('timestamp2'),
                             col('quoted_status.text').alias('text'),
                             col('quoted_status.extended_tweet.full_text').alias('text2'))
        df_t = df.filter('retweeted_status is null')
        df_t = df_t.select(col('user.id_str').alias('uid'),
                             col('user.screen_name').alias('name'),
                             col('id_str').alias('tid'),
                             col('created_at').alias('timestamp'),
                             col('created_at').alias('timestamp2'),
                             col('text').alias('text'),
                             col('extended_tweet.full_text').alias('text2'))
        df_t = df_t.union(df_rt)
        df2 = df_t.union(df_qt)
        print("Combined retweets&quotes&original tweets for file ", file_name, int(time() - start))

        df2 = df2.drop_duplicates()
        print("Dropped duplicates for file ", file_name, int(time() - start))

        # check if tweets can be extended
        df21 = df2.filter('text2 is not null')
        df21 = df21.withColumn('text',df21['text2'])
        df22 = df2.filter('text2 is null')
        df2 = df21.union(df22)
        df2 = df2.select('uid','name','tid','timestamp','timestamp2','text')
        df2 = df2.withColumn('text',lower(df2['text']))
        print("Extended text for file ", file_name, int(time() - start))

        # regex separately for category
        df2 = df2.withColumn('event_category', reg_udf('text'))
        df2 = df2.na.drop(subset=['event_category'])
        print("Categorized events for file ", file_name, int(time() - start))

        df2.write.format('parquet').mode('append').save(folder)
        print("Saved file ", file_name,int(time()-start))

# df_reg.write.format('json').mode('append').option("compression",
        #                                                               "org.apache.hadoop.io.compress.GzipCodec").save(folder)
    except:
        print('Failed processing ',file_name,int(time()-start))
        continue
