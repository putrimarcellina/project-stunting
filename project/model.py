import pickle
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Membaca File
df = pd.read_excel("dataset_stunting.xlsx")  # Membaca data dari file excel

# Menghapus kolom yang tidak diinginkan
df = df.drop(columns=['Unnamed: 7', 'Unnamed: 8'])

# Menghapus baris yang memiliki nilai null dan menyimpan hasilnya ke DataFrame baru
df_cleaned = df.dropna()

# Memeriksa nilai yang hilang di setiap kolom setelah penghapusan
missing_values_after = df_cleaned.isnull().sum()

# Memisahkan fitur dan target
X = df_cleaned[['berat', 'tinggi', 'lingkar_kepala', 'usia_ukur', 'jenis_kelamin']]  # Fitur
y = df_cleaned['label']  # Target

# Membagi data menjadi training set dan testing set (80% untuk training, 20% untuk testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model menggunakan Random Forest
model = RandomForestClassifier(random_state=42)

# Latih model menggunakan data pelatihan
model.fit(X_train, y_train)

# # Memprediksi label untuk data testing
# y_pred = model.predict(X_test)

# Menyimpan model
pickle.dump(model, open("model.pkl", "wb"))

# # Mengevaluasi model
# print('Akurasi:', accuracy_score(y_test, y_pred))
# print('Laporan Klasifikasi:\n', classification_report(y_test, y_pred))