# Pyspark scripts for extracting shocks from Cavium Twitter dataset

## Order of processing
1. extract_regex_phrases.py
    - From the range 2019.01-2020.07, filters all tweets containing phrases which are candidates of a shock
2. further_filter_regex.py
    - Using two rounds of regular expressions, further filters tweets so that there is a higher possibility that they contain shock events