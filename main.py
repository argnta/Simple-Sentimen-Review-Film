import pandas as pd
import string
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer

all_punctuation = string.punctuation
exclude_words = ['not', 'no', 'never', 'neither', 'nor']
stemmer = SnowballStemmer('english')

stop = [word for word in stopwords.words('english') if word not in exclude_words]

df = pd.read_csv('IMDBDataset.csv')
df['review'] = df['review'].str.replace(r'<br\s*/?>', '', regex=True)
df['review'] = df['review'].str.replace(f'[{all_punctuation}]', '', regex=True)
df['review'] = df['review'].str.lower()
df['review'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
df['review'] = df['review'].apply(lambda x: [stemmer.stem(y) for y in x.split()])

print(df.head())