import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import swifter
import string
import spacy
import re
import collections
from tqdm import tqdm

nlp = spacy.load('en')
punctuation = string.punctuation
stop_words = nlp.Defaults.stop_words

df = pd.read_csv('train.csv')

df['Neutrality'] = (df['Sentiment'] - 2).abs()
df['Positivity'] = np.where(df['Sentiment'] >= 2, 1, 0)

#TODO:
#exclamation marks, polatiry/subjectivity(textblob), extra spaces, grammar/spelling errors(textblob and levenshtein distance)
df['char_count'] = df['Phrase'].swifter.apply(len)
df['word_count'] = df['Phrase'].swifter.apply(lambda x: len(x.split()))
df['punctuation_count'] = df['Phrase'].swifter.apply(lambda x: len(''.join(p for p in x if p in punctuation)))
df['all_caps_word_count'] = df['Phrase'].swifter.apply(lambda x: len([word for word in x.split() if word.isupper()]))
df['stopword_count'] = df['Phrase'].swifter.apply(lambda x: len([word for word in x.split() if word.lower() in stop_words]))
df['avg_word_length'] = df['char_count'] / df['word_count']
df['capital_char_count'] = df['Phrase'].astype(str).str.findall(r'[A-Z]').str.len()
df['non_all_caps_words_capital_char_count'] = df['capital_char_count'] - 3 * df['all_caps_word_count']
df['first_char_uppercase'] = df['Phrase'].swifter.apply(lambda x: x[0].isupper()).astype(int)
print('Basic features done')

#lists to store spacy data in, used to make columns later on
cleaned_phrases = []
parts_of_speech = []
non_entity_counts = []
phrase_vecs = []

table = str.maketrans({}.fromkeys(string.punctuation))
for doc in tqdm(nlp.pipe(df['Phrase'].astype('unicode').values, batch_size=500,
                        n_threads=3)):
    if doc.is_parsed:
        entities = [n.text for n in doc.ents]
        full_string = ' '.join([n.text for n in doc])
        for entity in entities:
            full_string = full_string.replace(entity, '')
        cleaned_phrases.append(' '.join([n.lemma_ for n in doc])) #keep in stopwords, looks like doesn't matter
        parts_of_speech.append([n.pos_ for n in doc])
        non_entity_counts.append(len(full_string.translate(table).split()))
        phrase_vecs.append(doc.vector)
    else:
        cleaned_phrases.append(None)
        parts_of_speech.append(None)
        non_entity_counts.append(None)
        phrase_vecs.append(None)

print('Spacy loop done')

df['Cleaned Phrase'] = cleaned_phrases
df['Parts of Speech'] = parts_of_speech
df['Non entity word count'] = non_entity_counts
df['Entity word ratio'] = df['Non entity word count'] / df['word_count']

from textblob import TextBlob
df['Textblob_subjectivity'] = df['Cleaned Phrase'].swifter.apply(lambda tweet: TextBlob(tweet).subjectivity)
df['Textblob_polarity'] = df['Cleaned Phrase'].swifter.apply(lambda tweet: TextBlob(tweet).polarity)
#df['Textblob_corrected_spelling'] = df['Cleaned Phrase'].swifter.apply(lambda tweet: TextBlob(tweet).correct())
#df['spelling_errors'] = df.swifter.apply(num_edits, axis=1)

#convert parts of speech from list of lists to pandas dataframe
mlb = MultiLabelBinarizer()
df_parts_of_speech = pd.DataFrame(mlb.fit_transform(df['Parts of Speech']), columns=mlb.classes_, index=df.index)
df = df.drop(['Parts of Speech'], axis=1)

#convert word vector from list of numpy arrays to pandas dataframe
cols_word_vec = ['word_vec_' + str(n) for n in range(1,129)]
df_phrase_vectors = pd.DataFrame(data=phrase_vecs, columns=cols_word_vec)

#create pandas dataframe that contains count vectorized representation of phrase
word_vec = CountVectorizer(stop_words=None, analyzer='word', min_df = 0.0005, #ignore terms that appear in less than .05% of the documents
                      ngram_range=(1, 3)) #consider words and 2 word combinations
tfidf_matrix = word_vec.fit_transform(df['Cleaned Phrase'])
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=word_vec.get_feature_names())
print('Tf-idf done')

#combine all the dataframes into one
df = pd.concat([df, df_parts_of_speech], axis=1, sort=False)
df = pd.concat([df, df_phrase_vectors], axis=1, sort=False)
df = pd.concat([df, df_tfidf], axis=1, sort=False)
print('Combining dataframes done')

#check out frequency of phrases
#df_freq = df['Cleaned Phrase'].str.split(expand=True).stack().value_counts()
#df_freq.to_csv('frequencies.csv', index=False)

#write to csv
df.to_csv('train_cleaned.csv', index=False)
