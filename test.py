import sys
import pickle

with open('model_sentimen.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

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

    teks_input_list = [user_input] 

    vektor_user = vectorizer.transform(teks_input_list)

    prediksi_user = model.predict(vektor_user)[0] 

    print("-" * 50)
    print(f"Hasil Analisis: SENTIMEN {prediksi_user.upper()}")
    print("-" * 50 + "\n")