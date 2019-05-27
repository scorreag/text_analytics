# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 17:19:01 2019

@author: Admin
"""

"""
Top 5 parts of speech POS in the job descriptions
"""
import os
import re
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
nltk.download('tagsets')
nltk.download('stopwords')

folder = r"C:\..."
os.chdir(folder)


# Import the file. The file was previously cut to 2500 rows
salary = pd.read_csv("out.csv")
salary = salary.drop(['Unnamed: 0'], axis = 1)

# Create the body of text
#text_1 = list(salary["FullDescription"])
text = salary["FullDescription"].tolist()

#print(text[2499])

# Tokenize the text and create a flat list out of the tokens list of lists. Tokenize by sentences and words
sent_tokens = [nltk.sent_tokenize(i) for i in text] # converts to list of sentences 
flat_sent = [x for y in sent_tokens for x in y]
tokens = [word_tokenize(i) for i in text]
flat_tokens = [x for y in tokens for x in y]
a = flat_tokens[:5]
# Put all words in lower case to ensure each word is counted only once
lower_tokens = [t.lower() for t in flat_tokens]

# Create the pattern to extract only words and Extract all words from lower_tokens to avoid punctuation marks
all_words = r"[A-Za-z]\w+"
only_words = re.findall(all_words, str(lower_tokens)) # These are the official tokens

# Assign the part of speech to each token and determine the frequency distribution of the tags
pos = nltk.pos_tag(only_words)
tag_fd = nltk.FreqDist(tag for (word, tag) in pos)
print(tag_fd.most_common(5))

# Plot the frequency distribution
tag_fd.plot(20, cumulative = False)
plt.show()

"""
Top 5 parts of speech POS in the job descriptions after excluding stopwords
"""
# Create a list of stopwords and remove them from the tokens list (only_words)
stop_words = set(stopwords.words("english"))
no_stops = [t for t in only_words if t not in stop_words]

# Assing the POS to each token in the new list and determine the FreqDist of the tags
pos_1 = nltk.pos_tag(no_stops)
tag_fd_1 = nltk.FreqDist(tag for (word, tag) in pos_1)
print(tag_fd_1.most_common(5))

# Plot the frequency distribution
tag_fd_1.plot(20, cumulative = False)
plt.show()
    
# Calculate the frequency of words, extract the most common 100, and get only the frequencies
word_freq = nltk.FreqDist(word for word in only_words)
common_100 = word_freq.most_common(100)
counts = np.array([i for (x, i) in common_100])

# Plot Zipf's law
# Create an object containing the ranks
ranks = np.arange(1, len(common_100)+1)

# Create an object containing the frequencies. The indices line allows for the frequencies to 
# be inverted to make the plot consistent
indices = np.argsort(-counts)
frequencies = counts[indices]

# Run the correlation coeficients 
correlation = np.corrcoef(ranks, [math.log(c) for c in frequencies])
print(correlation)


# Create the frequency according to the theoretical Zipf. Convert the frequency to an 
# array and invert it before ploting it 
f_t_1 = []
for i in range(29128, 29229):
    for j in range(1, 101):
        f_t_1.append(i/j)
f_t_2 = np.array(f_t_1)
frequencies_2 = f_t_2[indices]

# Plot Zipf's law applied to the text
plt.loglog(ranks, frequencies, marker = ".", label = "Job Description Zipf's Law")
plt.loglog(ranks, frequencies_2, marker = ".", label = "Theoretical Zipf's Law")
plt.title("Zipf plot for job offers corpus tokens")
plt.xlabel("Log(Rank of the word)")
plt.ylabel("Log(Frequency of the word)")
plt.legend(loc = 'upper right')
plt.grid(True)
plt.show()


# Lemmatize the text. As stop words have been remove already, we can apply lemmatization to
# the no_stops list. All required packages have been imported at the beginning of the script

# Create the lemmatizer and apply lemmatization on the no_stops list
lem = WordNetLemmatizer()
lemmatized = [lem.lemmatize(t) for t in no_stops]

# Determine the frequency distribution of the 10 most common words after lemmatization
tag_fd_2 = nltk.FreqDist(word for word in lemmatized)
print(tag_fd_2.most_common(10))














