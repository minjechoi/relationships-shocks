"""
Snippet for using regular expressions to extract potential candidate tweets

spark-submit --master yarn --queue umsi-drom --num-executors 200 --driver-memory 30g --executor-memory 8g --conf spark.executor.extrajavaoptions="-Xmx16084m" further_filter_regex.py | tee logs/further_filter_regex.txt
"""
import os
from os.path import join
from time import time
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.functions import col,udf,lower
from datetime import datetime

import re

conf = SparkConf().setAppName('filterRegex')
conf.set("spark.sql.files.ignoreCorruptFiles", "true")
conf.set("spark.driver.maxResultSize","0")
sc = SparkContext.getOrCreate(conf)
sqlContext = SQLContext(sc)

############################### Regex 1: Event category #####################

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

############################### Regex 2: ego-centered #######################
S_close = ['father', 'dad', 'mother', 'mom', 'mum', 'gran\w+','gram\w+','aunt(?:\w+)?',
           'buddy',  'pal', 'uncle', '\bson', 'daughter', 'cousin', 'niece',
           'nephew',
           'brother', 'sister', 'wife', 'husband', '(?:boy|girl)friend', '\bfriend',
           '(?:\w+)?mate', 'partner', 'fianc(?:\w+)', 'family', 'relative']
S_others = S_close + ['coworker','colleague','neighbor','boss',"friend's"]
S_nondating = list(set(S_close)-set(['wife','husband','(?:boy|girl)friend','partner','fianc(?:\w+)']))

############################### Regex 3: Recency ############################
S_recent = ['today', 'tonight', 'yesterday', '\bjust\b ', 'recent', '\bnow\b',
            '(last|this|on|at|in the) (morning|afternoon|evening|night|monday|tuesday|wednesday|thursday|\
            friday|saturday|sunday|weekend|week\b)','(days|week|(hour|minute)s?) (ago|since)']
S_nonrecent = ['(weeks|month|year|yr)', 'a while', 'some time', 'anniversar',
               'the day', 'when \w+ was', 'dream','\b(in|at|around) [0-9]{1,4}\b']

str_death = r'%s' % '|'.join(S_death)
str_breakup = r'%s' % '|'.join(S_breakup)
str_job = r'%s' % '|'.join(S_job)
str_crime = r'%s' % '|'.join(S_crime)

str_me = r"\bi\b|\bive\b|\bim\b|\bmy\b|\bme\b|\bmine\b|\bour\b|\bwe\b" # words indicating "me"
str_others = r'%s|\bhis\b|\bhim\b|\bher\b|\bhe\b|\bshe\b|their|they|them'%('|'.join(S_others)) # words indicating others
str_close = r'%s'%'|'.join(S_close)
str_nondating = r'%s'%'|'.join(S_nondating)
str_nondating = r'(my|our)( \w+)? %s' % ('|'.join(S_nondating))

str_recent = r'%s' % '|'.join(S_recent)
str_nonrecent = r'%s' % '|'.join(S_nonrecent)

# get a list of the tweet directories to consider: 2015-2020
out_list = []


# function for creating file names based on each month
def floor(x):
    if x>=10:
        return 1
    else:
        return 0

print(len(out_list))
# str_events = r'passed away|death of|died|passing (of|away)|open to( \w+) (job|opportun|offer)'


def time_fn(dtime):
    return str(datetime.strftime(datetime.strptime(dtime, '%a %b %d %H:%M:%S +0000 %Y'), '%Y-%m-%d'))

reg_death = re.compile(str_death)
reg_crime = re.compile(str_crime)
reg_breakup = re.compile(str_breakup)
reg_job = re.compile(str_job)
reg_me = re.compile(str_me)
reg_close = re.compile(str_close)
reg_others = re.compile(str_others)
reg_nondating = re.compile(str_nondating)
reg_recent = re.compile(str_recent)
reg_nonrecent = re.compile(str_nonrecent)

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

def reg2_fn(text,event):
    if event in ['job-loss','crime']:
        if reg_me.search(text):
            if reg_others.search(text):
                return False
            else:
                return True
        else:
            return False
    elif event in ['death','health']:
        if reg_me.search(text):
            if reg_close.search(text):
                return True
            else:
                return False
        else:
            return False
    elif event=='breakup':
        if reg_nondating.search(text):
            return False
        else:
            if reg_me.search(text):
                return True
            else:
                return False
    else:
        return False

# a regex function for recency
def reg3_fn(text):
    if text:
        if reg_recent.search(text):
            if reg_nonrecent.search(text):
                return False
            else:
                return True
        else:
            return False
    else:
        return False

time_udf = udf(time_fn)
reg_udf = udf(reg_fn)
reg2_udf = udf(reg2_fn)
reg3_udf = udf(reg3_fn)

start = time()

df = sqlContext.read.parquet('relationships-shocks/tweetsByRegex/*/*/*.parquet')
print("Read files!",int(time()-start))

df = df.na.drop(subset=['text'])
df = df.na.drop(subset=['timestamp'])
df = df.na.drop(subset=['timestamp2'])
df = df.drop_duplicates()

df = df.withColumn('event_category', reg_udf('text')) # re-do this because there was a bug in the previous code resulting in 0 breakup tweets

df = df.withColumn('is_egocentric', reg2_udf('text','event_category'))
df = df.withColumn('is_recent', reg3_udf('text'))
df2 = df.filter(df.is_egocentric==True)
df3 = df2.filter(df2.is_recent==True)
df3 = df3.drop_duplicates()
for cat in ['death','health','job-loss','breakup','crime']:
    df3.filter(df3.event_category == cat).coalesce(1).write.json('relationships-shocks/shock_tweets_filtered_2/reg3/%s' % cat)
