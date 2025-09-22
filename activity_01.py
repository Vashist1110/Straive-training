#Cleaning the text
#1. create a text file and take the text from it
#2. convert the letters into the lowercase
#3. remove the punctuations like ,.!?
import string
from collections import Counter
from idlelib.iomenu import encoding

import matplotlib.pyplot as plt

text = open('amazon_review.txt',encoding='utf-8').read()
lower_case = text.lower()
cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
#print(cleaned_text)

#Tokenization
# splitting text into words
tokenized_words = cleaned_text.split()
print(tokenized_words)


stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
              "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
              "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these",
              "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do",
              "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
              "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
              "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
              "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
              "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
              "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


final_words = []
for word in tokenized_words:
    if word not in stop_words:
        final_words.append(word)

print(final_words)

emotion_list = []
with open('sentiment.txt', 'r',encoding='utf-8') as file:
    for line in file:
        cleaned_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = cleaned_line.split(':')
       # print("word :" + word + " " +"Emotion :" + emotion)

        if word in final_words:
            emotion_list.append(emotion)
print(emotion_list)
w = Counter(emotion_list)
print(w)


# Plotting the emotions on the graph

fig, ax1 = plt.subplots()
ax1.bar(w.keys(), w.values())
fig.autofmt_xdate()
plt.savefig('graph.png')
plt.show()

# # sentiment analysis
# positive_words = ["good", "great", "excellent", "amazing", "love", "fantastic","happy","satisfied", "best" ,"awesome"]
# negative_words = ["bad", "poor", "terrible", "worst", "hate", "awful", "horrible", "sad", "unsatisfied"]
#
# with open("amazon_review.txt","r",encoding="utf-8")