"""
A complex pipeline for building a shock classifier
"""

import os
import shutil
from os.path import join
import re
import bz2,gzip
import ujson as json
from time import time
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from random import sample
from nltk.tokenize import TweetTokenizer

# list of regular expressions


# get file directory
tweet_dir = '/twitter-turbo/decahose/raw/'

# extract tweets from twitter-turbo based on regular expressions
def extractInitialTweetsByRegex(file_name):
    save_dir = '/shared/0/projects/relationships/working-dir/life-events/initial-tweets'
    start = time()

    print("Starting ",file_name)

    """
    Negative
    Death of a close member
    Separation from a partner (break up / divorce)
    Fired from a job
    Injury / sickness
    Affected by a crime (legal problem)
    Natural disaster
    Financial problems
    Problems in house
    General distress
    Sudden event

    Positive
    Acceptance / admission / award
    Job offer / promotion
    Unexpected pregnancy
    """

    S_events = []
    # Death
    S_events.extend(['passed away','death of','passing (of|away)','died','loss of'])
    # Break up / divorce
    S_events.extend(['broke up', 'break\s?up', 'dumped', 'divorce',
                     'end(ed)?( to)? (my|our|the)( \w+)? relationship'])
    # Fired from a job
    S_events.extend(['fired','laid off','furloughed','layoff','jobless','unemployed',
                     'lost (my|his|her) (job|position)','kicked out'])
    S_events.extend(['(seeking|looking|searching)( \w+){,3} (job|offer|position)',
                     'open to( \w+){,2} (job|opportun|offer)'])
    # Injury / sickness
    S_events.extend(['(hurt|broke) (my|his|her)( \w+){,2}(head|arm|leg|toe|ankle|back|knee|bone|spine|hip|shoulder|foot|feet|hand|finger)'])
    S_events.extend(['suffering from','diagnosed with','tested positive','(am|is|has been|have been) (fighting|infected)',
                     '\b(in|at)( \w+){,2} (hospital|emergency room|er\b)','is under treatment','is being treated',
                     'critical (condition|state)','hospitalized','hospitalization'])
    # Crime / legal
    S_events.extend(['(is|been|was|got) (\w+ ){,1}(hit by a|robbed|attacked|assaulted|stolen|mugged|raped|discriminated|offended|harrassed)',
                     '(robbed|attacked|assaulted|stole|mugged|raped|discriminated|harrassed) (my|his|her)'])
    # Natural disaster

    # Financial problem
    # S_events.extend(['low on (funds|money)','financial (crisis|difficult)','in a (difficult|bad|horrible|terrible) situation'])
    # Problems in house
    # S_events.extend(['(house|room|kitchen|home|bathroom) is (flooded|out of electricity|destroyed|on fire|burn|robbed)'])
    # General distress
    S_events.extend(['(please|pls|plz) help','help (me|us|our)\b','any (help|assistance|support)',
                     '(need of|request for|requesting|ask(ing)? for|appreciate)( \w+){,3} (help|assistance|support)'])
    # Sudden events
    S_events.extend(['my( \w+){,2} (unexpected|sudden|unforeseen)',
                     '(unexpected|sudden|unforeseen)( \w+){1,2} of my','urgent','asap','emergency'])


    # General cases
    S_events.extend(['(^just|i just) (found out|was informed|heard|came to know|discovered)'])

    # Acceptance / admission / promotion

    S_events.extend(['(accepted|admission|acceptance|admittance|admitted|job|offer|position|internship|fellowship|promotion|promote|scholarship) (at|from|by|to)',
                     '\b(am|will be|was) awarded','achieved','achievement','i won','(will be|am) appointed as',
                     '\b(will receive|am receiving|(\w+ )?received)','accept(ing)? (an|the) (\w+ )?offer',
                     'job offer','(will be|was|am) offered','offered me a'])

    # public announcement
    S_events.extend(['to (announce|report|inform|notify|say that)\b'])


    # S_events.extend(['birthday','bday','b-day']) # birthday
    # S_events.extend(['hired','fired','laid off','furloughed','job offer',
    #                  'offered a','(job|position|offer|internship|fellowship) (at|from)',
    #                  'working (for|as)','promoted (to|as)',
    #                  '(searching|seeking|looking) for a (job|position)',
    #                  '(accepted|rejected|turned down|admitted) (at|from|by|to)']) # job-related
    # S_events.extend(['(got|getting) (wed|married|engaged)','newlywed',
    #                  'marrying','marriage','engagement','groom','bride','fianc']) # wedding/engagement
    # S_events.extend(['award','prize','runners(\s|-)?up',
    #                  '(first|second|third|1st|2nd|3rd) place']) # awards
    # # S_events.extend([]) # sports
    # S_events.extend(['anniversary']) # anniversary
    # # S_events.extend([]) # give birth
    # S_events.extend(['graduating','graduation']) # graduate
    # # S_events.extend([]) # admission
    # S_events.extend(['my interview']) # interview
    # S_events.extend(['my relocation','(be|am) moving to','relocat(ing|ed) to']) # moving
    # # S_events.extend([]) # travel
    # S_events.extend(['diagnosed with','suffering from'])
    #
    # S_events.extend(['to (inform|announce)','announcing my','annnouncement of',
    #                  ]) # informing
    # S_events.extend(['my (sudden|unexpected|unforseen|abrupt)','lost (my|her|his|our)',
    #                  'loss of (my|her|his|our)',
    #                  '(sudden|unexpected|unforseen|abrupt) \w+ of my']) # unexpected events

    # Possible replies towards such situations
    S_replies =['(wish|hope) (you|this|it|he|she)','sorry (for|to)','condol','aww','too bad','sucks','terrible',
                '(feel|get) (better|well)','rest in peace','rip','r.i.p','reach out','cheer up','good luck',
                     'sad to','cheer up','proud of','oh no','get over','awful','pray','fingers? cross']

    reg_events = re.compile(r'%s'%'|'.join(['('+x+')' for x in S_events]))
    reg_replies = re.compile(r'%s'%'|'.join(['('+x+')' for x in S_replies]))

    out_event = []
    out_response = []

    with bz2.open(join(tweet_dir,file_name)) as f:
        # pbar = tqdm(enumerate(f))
        for i,line in enumerate(f):
            # if i>100000:
            #     break

            # pbar.set_description('%d lines so far... %d tweets / %d responses collected'\
            #                      %(i,len(out_event),len(out_response)))
            try:
                line = line.decode('utf-8')
                obj = json.loads(line)
                if obj['lang']!='en':
                    continue

                # if it is a quote or a reply
                if (obj['in_reply_to_status_id_str']!=None) or ('quoted_status' in obj):
                    response_text=obj['text']
                    res = reg_replies.findall(response_text.lower())
                    if res:
                        res = [x for x in res[0] if x]
                        obj['extracted']=res
                        out_response.append(obj)

                # otherwise, it's the original text
                else:
                    if 'quoted_status' in obj:
                        text = obj['quoted_status']['text']
                    else:
                        text = obj['text']
                    res = reg_events.findall(text.lower())
                    if res:
                        res = [x for x in res[0] if x]
                        obj['extracted']=res
                        out_event.append(obj)
            except:

                continue

    with gzip.open(join(save_dir,'event_'+file_name.replace('.bz2','.gz')),'wt') as f:
        for obj in out_event:
            f.write(json.dumps(obj)+'\n')

    with gzip.open(join(save_dir,'response_'+file_name.replace('.bz2','.gz')),'wt') as f:
        for obj in out_response:
            f.write(json.dumps(obj)+'\n')
    print("Finished %s (%d lines/%d mins): %d events, %d responses"%(file_name,i,int((time()-start)/60),len(out_event),len(out_response)))
    return

# using combinations of regular expressions, get different category levels (will be using this only once)
def regexByCategoryLevel(category,level):
    """
    A function for selecting samples and the level of precision to extract from our filtered tweets.
    Saves into a gzip file, organized by event category and level of precision
    :param category: str, which life event to extract. Determines the regular expressions
    :param level: int, 1 (the same category used for filtering tweets),
                2 (additional constraints to make sure it's related to "me" or someone around me if death),
                3 (additional constraints to make sure it has happened within a week)
    :return: None
    """

    # for creating a list of regular expressions related to recent events


    # for creating a list of regular expressions related to people around me
    S_people = ['father', 'dad', 'mother', 'mom', 'mum', 'grand\w+', 'gran', 'aunt(\w+)?', 'uncle', '\bson\b',
                'daughter',
                'cousin', 'niece', 'nephew', 'brother', 'sister', 'wife', 'husband', '(boy|girl)friend',
                'friend', 'mate', 'partner', 'fianc(\w+)']  # level 2
    str_people = '(' + '|'.join(S_people) + ')'
    str_people = 'my( \w+){,2} %s( @\w+)?' % str_people

    # level 1: basic regular expressions, still ensures a high number
    if level==1:
        if category=='death':
            S_death = ['passed away', 'death', 'passing (of|away)', 'died', 'loss of']
            reg_event = re.compile(r'%s'%('|'.join(S_death)))
            # reg_event = re.compile(r'(passed away|(death|loss) of|passing (of|away)|died|loss of)')
        elif category=='job-loss':
            S_job = ['\bfired\b', 'laid off', 'furlough', '\blay\s?off', 'jobless', 'unemployed',
                     '(lost my|seeking|looking|searching|open to)( \w+)\{,2\} (job|opportun|offer|position)']
            reg_event = re.compile(r'%s'%('|'.join(S_job)))
        elif category=='crime':
            S_crime = ['robbed', 'assaulted', '\bstole', 'mugged', 'raped', 'discriminated', 'offended', 'harrassed',
                       'broke into', 'hacked', 'damaged', '\bsued\b', '\btowed']
            reg_event = re.compile(r'%s'%('|'.join(S_crime)))
        elif category=='separation':
            S_breakup = ['broke up', 'break\s?up', 'dumped', 'divorce',
                         'end(ed)?( to)? (my|our|the)( \w+)? relationship']
            reg_event = re.compile(r'%s'%('|'.join(S_breakup)))

    # level 2: exclude phrases of events happening to others than "me" (except death)
    elif level==2:
        S_close = ['father', 'dad', 'mother', 'mom', 'mum', 'grand\w+', 'gran', 'aunt(\w+)?',
                   'buddy', 'pal', 'uncle', 'son\b', 'daughter', 'cousin', 'niece', 'nephew',
                   'brother', 'sister', 'wife', 'husband', '(boy|girl)friend', 'friend',
                   '(\w+)?mate', 'partner', 'fianc(\w+)', 'family']
        S_others = S_close + ['coworker', 'colleague', 'neighbor', 'boss', "friend's"]
        str_me = r"\bi\b|\bmy\b|\bme\b|\bmine\b|\bour\b"  # words indicating "me"
        str_others = r'%s|\bhis\b|\bhim\b|\bher\b|\bhe\b|\bshe\b|their|they|them' % (
            '|'.join(S_others))  # words indicating others
        str_close = r'(my|our)( \w+)? %s' % ('|'.join(S_close))
        if category in ['job-loss','crime']:
            reg_pos = re.compile(str_me)
            reg_neg = re.compile(str_others)
        elif category in ['death','health']:
            reg_pos = re.compile(str_close)
            reg_neg = None
        elif category=='separation':
            S_others = ['father', 'dad', 'mother', 'mom', 'mum', 'grand\w+', 'gran', 'aunt(\w+)?',
                       'buddy', 'pal', 'uncle', 'son\b', 'daughter', 'cousin', 'niece', 'nephew',
                       'brother', 'sister', '\bfriend',
                       '(\w+)?mate', 'family']
            reg_pos = re.compile(str_me)
            reg_neg = re.compile(r'%s'%('|'.join(S_others)))

    # level 3: phrases to ensure recency
    elif level==3:
        S_recent = ['today', 'tonight', 'yesterday', '\bjust\b ', 'recent', '\bnow\b',
                    '(last|this|on) (morning|afternoon|evening|night|monday|tuesday|wednesday|thursday|\
                    friday|saturday|sunday|weekend|week\b)', '(days|week|(hour|minute)s?) ago']
        S_nonrecent = ['(weeks|month|year|yr)', 'a while', 'some time', 'since',
                       'the day', 'when \w+ was', 'when i', 'dream', '\b(in|at|around) [0-9]{1,4}\b']+['ditched','kicked out']
        reg_pos = re.compile(r'%s' % '|'.join(S_recent))
        reg_neg = re.compile(r'%s' % '|'.join(S_nonrecent))

    out = []

    # filtering phrases
    n_in = 0 # samples you start with

    # only for level 1, we extract directly from our tweet dump instead of filtered files
    if level==1:
        S = set()  # to ensure that each tweet is included only once
        load_dir = '/shared/0/projects/relationships/working-dir/life-events/initial-tweets'
        files = sorted([x for x in os.listdir(load_dir) if 'event' in x])
        pbar = tqdm(files)
        for i,filename in enumerate(pbar):
            with gzip.open(join(load_dir, filename)) as f:
                for i, line in enumerate(f):
                    if i%100==0:
                        pbar.set_description('%s:%d/%d'%(filename,len(out),n_in))
                    n_in+=1
                    obj = json.loads(line.decode('utf-8'))
                    # 1) ensure we're looking at the original tweet form
                    while ('quoted_status' in obj) or ('retweeted_status') in obj:
                        if 'quoted_status' in obj:
                            obj = obj['quoted_status']
                        if 'retweeted_status' in obj:
                            obj = obj['retweeted_status']
                    if 'extended_tweet' in obj:
                        if 'full_text' in obj['extended_tweet']:
                            if obj['extended_tweet']['full_text']:
                                obj['text']=obj['extended_tweet']['full_text']
                    tid = obj['id_str']
                    # 2) remove duplicates so that each tweet appears only once
                    if tid in S:
                        continue
                    else:
                        S.add(tid)
                    text = obj['text'].lower()
                    if reg_event.search(text):
                        out.append(obj)
    else: # levels 2 and 3
        load_file = '/shared/0/projects/relationships/working-dir/life-events/filtered-tweets/reg.%d.%s.json.gz' % (level-1, category)
        with gzip.open(load_file) as f:
            for line in f:
                n_in+=1
                obj = json.loads(line.decode('utf-8'))
                text = obj['text'].lower()
                if reg_pos.search(text):
                    if reg_neg:
                        if reg_neg.search(text):
                            continue
                        else:
                            out.append(obj)
                    else:
                        out.append(obj)

    save_file = '/shared/0/projects/relationships/working-dir/life-events/filtered-tweets/reg.%d.%s.json.gz' % (level,category)

    with gzip.open(save_file,'wt') as f:
        for obj in out:
            f.write(json.dumps(obj)+'\n')

    print("%s\tLevel %d\t%d->%d"%(category,level,n_in,len(out)))
    return


# # a code for sampling a dataframe of tweets at any time, to be labeled
# def getUnlabeledSamples(typ,level,category,n_samples=100):
#     """
#     :param typ: the prefix ('reg' or 'cls')
#     :param level: the number level
#     :param category: the type of event
#     :param n_samples: the number of samples to get, set to 100 by default
#     :return:
#     """
#     # save samples containing info
#     load_file = '/shared/0/projects/relationships/working-dir/life-events/filtered-tweets/%s.%d.%s.json.gz' % (typ,level,category)
#     save_file = '/shared/0/projects/relationships/working-dir/life-events/labeled-tweets/%s.%d.%s.df.tsv'%(typ,level,category)
#
#     # read the file
#     out = []
#     tokenize = TweetTokenizer().tokenize
#
#     S_tid = [] # to remove duplicate tweet ids
#     df_dir = '/shared/0/projects/relationships/working-dir/life-events/train-data/'
#     for file in os.listdir(df_dir):
#         if '.df.tsv' in file:
#             df = pd.read_csv(join(df_dir,file),sep='\t')
#             S_tid.extend([x[1:] for x in df['tweet_id'].values])
#     S_tid = set(S_tid)
#
#     with gzip.open(load_file) as f:
#         for ln,_ in enumerate(f):
#             continue
#     with gzip.open(load_file) as f:
#         if n_samples<ln:
#             interval = int(ln/(n_samples+20))
#             bit = np.random.randint(0,interval-1)
#         else:
#             interval = 1
#             bit = 0
#         for ln2,line in enumerate(f):
#             if len(out)>=n_samples:
#                 break
#             if ln2%interval==bit:
#                 obj = json.loads(line.decode('utf-8'))
#                 if obj['id_str'] in S_tid:
#                     continue
#                 text = obj['text']
#                 text = ' '.join(tokenize(text))
#                 out.append(('t'+obj['id_str'],'u'+obj['user']['id_str'],category,text,0))
#
#     # save at end
#     df = pd.DataFrame(out,columns=['tweet_id','user_id','category','text','is_shock'])
#     df.to_csv(save_file,sep='\t',index=False)
#     print(typ,category,level,'%d/%d samples to label!'%(min(n_samples,ln),ln))
#     return

# a code for sampling a dataframe of tweets at any time, to be labeled
# code updated for cavium files
def getUnlabeledSamples(typ,level,category,n_samples=100):
    """
    :param typ: the prefix ('reg' or 'cls')
    :param level: the number level
    :param category: the type of event
    :param n_samples: the number of samples to get, set to 100 by default
    :return:
    """
    # save samples containing info
    load_file = '/shared/0/projects/relationships/working-dir/life-events/filtered-tweets-cavium/reg3.%s.json' % (category)
    save_file = '/shared/0/projects/relationships/working-dir/life-events/labeled-tweets/%s.%d.%s.df.tsv'%(typ,level,category)

    # read the file
    out = []
    tokenize = TweetTokenizer().tokenize

    S_uid = [] # to remove duplicate users
    df_dir = '/shared/0/projects/relationships/working-dir/life-events/train-data/%s'%category
    for file in os.listdir(df_dir):
        if '.df.tsv' in file:
            df = pd.read_csv(join(df_dir,file),sep='\t')
            S_uid.extend([x[1:] for x in df['user_id'].values])
    S_uid = set(S_uid)

    with open(load_file) as f:
        for ln,_ in enumerate(f):
            continue
    with open(load_file) as f:
        for ln2,line in enumerate(f):
            obj = json.loads(line)
            if obj['uid'] in S_uid:
                continue
            else:
                S_uid.add(obj['uid'])
            text = obj['text']
            text = ' '.join(tokenize(text))
            out.append(('t'+obj['tid'],'u'+obj['uid'],category,obj['timestamp'],text,0))
    from random import sample
    out = sample(out,n_samples)

    # save at end
    df = pd.DataFrame(out,columns=['tweet_id','user_id','category','date','text','is_shock'])
    df.to_csv(save_file,sep='\t',index=False)
    print(typ,category,level,'%d/%d samples to label!'%(min(n_samples,ln),ln))
    return

# a code for creating a training dataset out of the labeled samples
def labeledSamplesToTrainData(typ, category,level=None):
    """

    :param typ: whether "reg" or "cls". If "reg" then levels are not needed, as we will create train/test/valid sets out of all available lablels.
    :param level: level, only needed if typ=cls
    :return:
    """

    load_dir = '/shared/0/projects/relationships/working-dir/life-events/labeled-tweets/'
    save_dir = '/shared/0/projects/relationships/working-dir/life-events/train-data/%s/'%category

    # if regular expressions, use all labeled samples to create a training/test/validation set
    if typ=='train':
        df_all = pd.DataFrame([], columns=['tweet_id', 'user_id', 'category', 'text', 'is_shock'])
        files = [join(load_dir,'reg.3.%s.df.tsv'%category)]
        # files = sorted([x for x in os.listdir(load_dir) if 'reg.' in x])
    elif typ=='cls':
        df_all = pd.DataFrame([], columns=['tweet_id', 'user_id', 'category', 'text', 'is_shock'])
        files = [join(load_dir,'cls.%d.%s.boundary.df.tsv'%(level,category))]
        # files = sorted([x for x in os.listdir(load_dir) if 'cls.%d.'%level in x])

    # # else, then the classified tweets for this round are set as additional training data only
    # elif typ.startswith('cls'):
    #     df_all = pd.DataFrame([], columns=['tweet_id', 'user_id', 'category', 'text', 'pred_score', 'is_shock'])
    #     files = sorted([join(load_dir,x) for x in os.listdir(load_dir) if ('cls'+str(level) in x) and ('boundary' in x)])
    # elif typ=='95':
    #     df_all = pd.DataFrame([], columns=['tweet_id', 'user_id', 'category', 'text', 'pred_score', 'is_shock'])
    #     files = sorted([join(load_dir,x) for x in os.listdir(load_dir) if ('.95.' in x)])
    else:

        print("Error: typ should either be 'train' or 'cls'")
        return

    # load to df_all, and move the files
    for file in files:
        print("read ",file)
        df_tmp = pd.read_csv(join(load_dir,file),sep='\t')
        df_tmp['file_name']=file
        df_all = df_all.append(df_tmp)
        print(len(df_all),' lines after adding ',file)

    # if level:
    #     move_dir = join(load_dir,'%s.%d'%(typ,level))
    # else:
    #     move_dir = join(load_dir,'%s'%(typ))
    # if not os.path.exists(move_dir):
    #     os.makedirs(move_dir)
    # for file in files:
    #     shutil.move(join(load_dir,file),join(move_dir,file))

    # shuffle
    df_all = df_all.sample(frac=1,replace=False)

    # if 'reg', split to train/test/val, else, only add more training samples
    if typ=='train':
        idx1 = 400
        idx2 = 600
        df_tr = df_all[:idx1]
        df_v = df_all[idx1:idx2]
        df_t = df_all[idx2:]
        if not os.path.exists(join(save_dir)):
            os.makedirs(join(save_dir))
        df_tr.to_csv(join(save_dir,'reg.3.train.df.tsv'),sep='\t',index=False)
        df_v.to_csv(join(save_dir,'reg.3.val.df.tsv'),sep='\t',index=False)
        df_t.to_csv(join(save_dir,'reg.3.test.df.tsv'),sep='\t',index=False)
        print(len(df_tr),len(df_v),len(df_t))
    elif typ=='cls':
        df_all.to_csv(join(save_dir,'cls.%d.%s.train.df.tsv'%(level,category)),sep='\t',index=False)
    return

# def saveTweetsWithHighScores()

# a code for

if __name__=='__main__':
    # extract tweets according to regex
    # files = [x for x in sorted(os.listdir(tweet_dir)) if '2020' in x]
    # try:
    #     pool = Pool(12)
    #     pool.map(extractInitialTweetsByRegex,files)
    # finally:
    #     pool.close()

    # use regular expressions to extract different levels

    # for cat in ['job-loss']:
    # for cat in ['death', 'job-loss', 'separation','crime']:
    #     for level in range(2,4):
    #         regexByCategoryLevel(cat,level)

    # from initial-tweets, split them into event categories
    # for level in range(3,4):
    #     for cat in ['crime']:
    #     # for cat in ['death','job-loss','breakup','crime']:
    #         getUnlabeledSamples('reg',level,cat,n_samples=500)
    #     labeledSamplesToTrainData(typ='cls',level=8,category='crime')
    # manual labeling process


    # convert samples to training data
    # for cat in ['death','job-loss','breakup','crime']:
    #     labeledSamplesToTrainData(typ='train',level=1,category=cat)
    # labeledSamplesToTrainData(typ='cls',level=5,category='crime')
    # labeledSamplesToTrainData(typ='cls',level=5,category='breakup')
    # labeledSamplesToTrainData(typ='cls',level=5,category='death')
    labeledSamplesToTrainData(typ='cls',level=5,category='job-loss')
    # labeledSamplesToTrainData(typ='cls',level=2,category='separation')
    # labeledSamplesToTrainData(typ='cls',level=5)

    # train classifier on first dataset
    # run training code from run_shock_classifier.py

    #
