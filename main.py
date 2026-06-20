import pandas as pd
import string
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('IMDBDataset.csv')

all_punctuation = string.punctuation
exclude_words = ['not', 'no', 'never', 'neither', 'nor']
stop = [word for word in stopwords.words('english') if word not in exclude_words]
stemmer = SnowballStemmer('english')

df['review'] = df['review'].str.replace(r'<br\s*/?>', '', regex=True)
df['review'] = df['review'].str.replace(f'[{all_punctuation}]', '', regex=True)
df['review'] = df['review'].str.lower()
df['review'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df['review'] = df['review'].apply(lambda x: [stemmer.stem(y) for y in x.split()])
df['review'] = df['review'].apply(lambda x: ' '.join([stemmer.stem(y) for y in x.split()]))

train_df, test_df = train_test_split(df, test_size=0.25, random_state=42, shuffle=True, stratify=df['sentiment'])

vectorizer = TfidfVectorizer()
train_tfidf = vectorizer.fit_transform(train_df['review'])
test_tfidf = vectorizer.fit_transform(test_df['review'])

sentiment_train = train_df['sentiment']
sentiment_test = test_df['sentiment']


print(df.head())