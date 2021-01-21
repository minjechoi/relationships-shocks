# Python scripts for training and improving classifiers that infer whether a tweet is a shock tweet
- Applies BERT classifiers + active learning

## Order of processing
1. pipeline_shock_classifier.py
    - contains various subfunctions that you have to run one at a time
    - samples training set / produces additional tweets to label
2. run_shock_classifier.py
    - contains model to run
    - need to iterate with additionally labeled data over time
   
## Data stored
Main directory: ```/shared/2/projects/relationships-shocks/working-dir'''
- annotation_comparison: a comparison of the tweets annotated by 3 separate people
- labeled-tweets: a list of the tweets that were manually labeled as the process of active learning
- train-data: a list of the actual train/test/validation datasets used for training the classifier
- predicted-tweets: a list of the predicted tweets from our dataset using the best classifiers + pre-labeled results