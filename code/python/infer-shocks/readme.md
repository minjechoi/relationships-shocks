# Python scripts for training and improving classifiers that infer whether a tweet is a shock tweet
- Applies BERT classifiers + active learning

## Order of processing
1. pipeline_shock_classifier.py
    - contains various subfunctions that you have to run one at a time
    - samples training set / produces additional tweets to label
2. run_shock_classifier.py
    - contains model to run
    - need to iterate with additionally labeled data over time