from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import requests
import pandas as pd
from textblob import TextBlob
from googletrans import Translator

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from textblob import TextBlob
from googletrans import Translator

# Kripto Koin haber getirtme
URL_kriptokoin = "https://kriptokoin.com/haberler/"
page_kriptokoin = requests.get(URL_kriptokoin)
soup_kriptokoin = BeautifulSoup(page_kriptokoin.content, "html.parser")
news_kriptokoin = soup_kriptokoin.find_all("h2", class_="kanews-post-headline")

#Kripto Para Haber Haberler
URL_kriptoparahaber = "https://kriptoparahaber.com/kategori/haberler/"
page_kriptoparahaber = requests.get(URL_kriptoparahaber)
soup_kriptoparahaber = BeautifulSoup(page_kriptoparahaber.content, "html.parser")
news_kriptoparahaber = soup_kriptoparahaber.find_all("h2" , class_= "kanews-post-headline truncate truncate-2")

#Koin Bülteni Haber
URL_koinbulteni = ("https://koinbulteni.com/haberler")
page_koinbulteni = requests.get(URL_koinbulteni)
soup_koinbulteni = BeautifulSoup(page_koinbulteni.content, "html.parser")
new_kriptobulteni = soup_koinbulteni.find_all("h4", class_="sc_blogger_item_title entry-title")


allnews = []
allnews += news_kriptokoin
allnews += news_kriptoparahaber
allnews += new_kriptobulteni


translator = Translator()

data_list = []

for y in allnews:
    haber = y.text.strip()
    translated = translator.translate(str(haber), dest='en').text
    blob = TextBlob(translated)
    duygu_skoru = blob.sentiment.polarity

    if duygu_skoru < 0:
        duygu = "Olumsuz"
    elif duygu_skoru > 0:
        duygu = "Olumlu"
    else:
        duygu = "Nötr"

    data_list.append([haber, duygu_skoru, duygu])

# Eski DataFrame'i oku (eğer daha önce oluşturulmuşsa)
# Liste üzerinden DataFrame oluşturma
dataframe = pd.DataFrame(data_list, columns=["Başlık", "Duygu Skoru", "Duygu"])

# Etiketleme
dataframe['Etiket'] = dataframe['Duygu'].apply(lambda x: 1 if x == "Olumlu" else (0 if x == "Nötr" else -1))

# Eğer datalist.csv dosyası mevcutsa, mevcut verilere yeni verileri ekleyin; aksi halde dosyayı oluşturun.
try:
    existing_data = pd.read_csv('datalist.csv', index_col=0)
    updated_data = pd.concat([existing_data, dataframe], ignore_index=True)
    updated_data.to_csv('datalist.csv', index=False)
    print("Veriler dosyaya başarıyla eklendi.")
except FileNotFoundError:
    dataframe.to_csv('datalist.csv', index=False)
    print("Dosya bulunamadığından dolayı yeni dosya oluşturuldu.")


# Verileri hazırla
haber_basliklari = dataframe["Başlık"]
duygu_skorlari = dataframe["Duygu Skoru"]

# Matplotlib kullanarak görselleştirme
plt.figure(figsize=(10, 6))
plt.plot(haber_basliklari, duygu_skorlari, marker='o', linestyle='', color='b')
plt.title('Haber Duygu Skorları', fontsize=12)
plt.xlabel('Haber Başlıkları' , fontsize=10)
plt.ylabel('Duygu Skoru',  fontsize=10)
plt.xticks(rotation=45, ha='right')  # Başlıkları yatayda döndürme
plt.tight_layout()
plt.show()
plt.subplots_adjust(bottom=0.2, top=0.9)

def translate_and_analyze(text):
    translator = Translator()
    translated_text = translator.translate(text, dest='en').text
    blob = TextBlob(translated_text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Veri kümesine çeviri ve duygu analizi sonuçlarını ekleyin
dataframe['Translated_Title'] = dataframe['Başlık'].apply(translate_and_analyze)

# Veri kümesini eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(dataframe['Translated_Title'], dataframe['Duygu'], test_size=0.2, random_state=42)

# Modeli oluşturun ve eğitin
model = make_pipeline(CountVectorizer(max_features=5000, ngram_range=(1, 2)), MultinomialNB())
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapın
y_pred = model.predict(X_test)

# Modelin performansını değerlendirin
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Model Accuracy: {accuracy}')
print('Classification Report:\n', classification_rep)