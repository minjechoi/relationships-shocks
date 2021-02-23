"""
Handles with miscellaneous functions for processing the files
"""

import os
from os.path import join

def clean_predicted_tweets():
    """
    The predicted tweets from classifiers may have a number of problems (e.g., duplicate)
    :return:
    """
    return