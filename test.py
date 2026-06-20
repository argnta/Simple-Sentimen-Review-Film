import pickle
import string
import re  
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer

with open('model_sentimen.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

all_punctuation = string.punctuation
exclude_words = ['not', 'no', 'never', 'neither', 'nor']
stop = [word for word in stopwords.words('english') if word not in exclude_words]
stemmer = SnowballStemmer('english')

def bersihkan_teks(teks):
    teks = re.sub(r'<br\s*/?>', '', teks)                
    teks = re.sub(f'[{all_punctuation}]', '', teks)     
    teks = teks.lower()                               
    teks = re.sub(r'\b10\b', 'ratingten', teks)         
    teks = re.sub(r'\b100\b', 'ratinghundred', teks)         
   
    kata_bersih = [word for word in teks.split() if word not in stop]
    kata_stemmed = [stemmer.stem(y) for y in kata_bersih]
    
    return ' '.join(kata_stemmed)

print("=" * 50)
print("  APLIKASI ANALISIS SENTIMEN REVIEW FILM (IMDB)  ")
print("  Ketik 'exit' pada input review untuk keluar.   ")
print("=" * 50 + "\n")

while True:
    user_input = input("Masukkan Review Film: ")
    
    if user_input.strip().lower() == 'exit':
        print("\nBye!!")
        break
    
    if not user_input.strip():
        print("Input tidak boleh kosong\n")
        continue

    input_bersih = bersihkan_teks(user_input)
    
    teks_input_list = [input_bersih] 

    vektor_user = vectorizer.transform(teks_input_list)
    prediksi_user = model.predict(vektor_user)[0] 

    print("-" * 50)
    print(f"Hasil Analisis: SENTIMEN {prediksi_user.upper()}")
    print("-" * 50 + "\n")