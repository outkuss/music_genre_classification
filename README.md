import os
import difflib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def turkish_to_english(text):
    # Türkçe karakterleri İngilizce karakterlere çeviren bir çeviri sözlüğü
    turkish_to_english_mapping = {
        'ı': 'i',
        'ğ': 'g',
        'ü': 'u',
        'ş': 's',
        'i': 'i',
        'ö': 'o',
        'ç': 'c',
        'İ': 'I',
        'Ğ': 'G',
        'Ü': 'U',
        'Ş': 'S',
        'Ö': 'O',
        'Ç': 'C'
    }

    # Her bir karakteri çeviri sözlüğüne göre değiştir
    translated_text = ''.join(turkish_to_english_mapping.get(char, char) for char in text)

    return translated_text

def load_turkish_lyrics_dataset(file_path):
    try:
        # Metin dosyasını oku
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # Her bir satırı parçalara ayır ve bir veri çerçevesi oluştur
        data = {"lyrics": [], "genre": []}
        for line in lines:
            # Satırı virgülle ayırarak lyrics ve genre olarak ekleyelim (örnek bir format)
            parts = line.strip().split(",")
            if len(parts) >= 2:  # En azından iki alanın olması gerekir
                lyrics = turkish_to_english(parts[0].strip())
                genre = parts[1].strip()
                data["lyrics"].append(lyrics)
                data["genre"].append(genre)

        # Veri çerçevesini oluştur
        dataset = pd.DataFrame(data)
        return dataset

    except FileNotFoundError:
        print(f"Dosya bulunamadı: {file_path}")
        return None
    except Exception as e:
        print(f"Bir hata oluştu: {e}")
        return None

def max_length(X_train_seq, X_test_seq):
    return max(max(len(seq) for seq in X_train_seq), max(len(seq) for seq in X_test_seq))

def preprocess_data(texts, labels):
    # Verileri eğitim ve test setlerine bölmek
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # LabelEncoder kullanarak etiketleri sayısal değerlere dönüştürme
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Metin verilerini sayısal dizilere dönüştürme
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # Dizileri aynı uzunluğa getirme (padding)
    max_len = max_length(X_train_seq, X_test_seq)
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

    # Etiketleri kategorik hale getirme
    y_train_categorical = to_categorical(y_train_encoded)
    y_test_categorical = to_categorical(y_test_encoded)

    return X_train_padded, y_train_categorical, X_test_padded, y_test_categorical, label_encoder, tokenizer, max_len

def build_and_train_cnn(X_train, y_train, X_test, y_test, vocab_size, max_len):
    # CNN modelini oluşturma
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=100, input_length=max_len))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))

    # Modeli derleme
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Modeli eğitme
    model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), verbose=2)

    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    # Test seti üzerinde modelin performansını değerlendir
    y_pred = np.argmax(model.predict(X_test), axis=-1)

    # Doğruluk (Accuracy) metriği
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Doğruluk (Accuracy): {accuracy}")

    # Hassasiyet (Precision), Geri Çağırma (Recall), F1 Skoru
    classification_report = metrics.classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print("Sınıflandırma Raporu:")
    print(classification_report)

    # Karışıklık Matrisi
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    print("Karışıklık Matrisi:")
    print(confusion_matrix)

    return accuracy

def main():
    dataset = load_turkish_lyrics_dataset("turkce_rock.txt")

    X_train, y_train, X_test, y_test, label_encoder, tokenizer, max_len = preprocess_data(dataset["lyrics"], dataset["genre"])

    model = build_and_train_cnn(X_train, y_train, X_test, y_test, vocab_size=len(tokenizer.word_index), max_len=max_len)

    # Modelin performansını değerlendir
    evaluate_model(model, X_test, y_test, label_encoder)

    given_lyrics = str(input("Şarkı Sözlerini Giriniz: "))
    given_lyrics = turkish_to_english(given_lyrics)
    given_lyrics_vectorized = tokenizer.texts_to_sequences([given_lyrics])
    given_lyrics_padded = pad_sequences(given_lyrics_vectorized, maxlen=max_len, padding='post')
    given_lyrics_reshaped = np.expand_dims(given_lyrics_padded, axis=-1)

    # Verilen şarkı türünü tahmin et
    predicted_genre = label_encoder.inverse_transform(np.argmax(model.predict(given_lyrics_reshaped), axis=-1))

    print(f"Bu parçanın tahmini türü: {predicted_genre}")

if __name__ == "__main__":
    main()
