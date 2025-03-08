# Preprocessing-data-dan-Data-Visualization-UCIML
> dataset
https://archive.ics.uci.edu/dataset/880/support2 

> Kelas 2023C  

```
Marshanda Claudia Iswahyono(23031554014)

Zulma Nayla Ifaada(23031554063) 

Gatiari Dwi Panefi (23031554110)

Metha Nailis Sa'adah(23031554159)
```

## Data Collection 
```
!pip install ucimlrepo
```

Mengambil data X dan Y lalu melihat maksud dari masing-masing variable
```
from ucimlrepo import fetch_ucirepo
import pandas as pd
# fetch dataset
support2 = fetch_ucirepo(id=880)

# data (as pandas dataframes)
X = support2.data.features
y = support2.data.targets

# metadata
print(support2.metadata)
# variable information
variables_df= pd.DataFrame(support2.variables)
pd.set_option('display.max_colwidth', None)
variables_df
```
Mendefinisikan dataset dengan mengambil variable data x pada dataset asli

```
import pandas as pd

dataset = pd.DataFrame(X)
dataset.head()
```
melihat informasi datatypes pada dataset
```
dataset.info()
```
## Inconsistency Data

Melihat shape dataset 
```
dataset.shape
```
Melihat apakah pada dataset ada data yang duplikat
```
dataset.duplicated().sum()
```
Process encoding tanpa mengganti missing value dengan -1
```
import pandas as pd
import numpy as np

cat_columns = dataset.select_dtypes(include=['object']).columns
cat_columns = cat_columns.drop('income', errors='ignore')  # Hapus 'income' dari encoding kategori lainnya

mappings = {}  # Dictionary untuk menyimpan mapping kategori

# Encoding data kategorikal selain fitur 'income'
for col in cat_columns:
    dataset[col] = dataset[col].astype('category', copy=False)  # Ubah ke kategori tanpa menghapus NaN
    mappings[col] = dict(enumerate(dataset[col].cat.categories))  # Simpan mapping
    dataset[col] = dataset[col].cat.codes.replace(-1, np.nan)  # Ganti -1 menjadi NaN dan overwrite kolom lama

# Midpoint Encoding khusus untuk income
income_mapping = {
    '$11-$25k': 18000,
    '$25-$50k': 37500,
    '>$50k': 75000,
    'under $11k': 5500
}

if 'income' in dataset.columns:  # Cek apakah income ada dalam dataset sebelum mengubahnya
    dataset['income'] = dataset['income'].map(income_mapping)
    mappings['income'] = income_mapping  # Tambahkan ke dictionary untuk dicetak bersama

# Print mapping kategori
for col, mapping in mappings.items():
    print(f"Mapping for {col}:")
    for key, value in mapping.items():
        print(f"    {key}: '{value}'")
    print("\n")

```
## EDA (Exploratory Data Analysis)
> Distribusi Numerik dan Kategorikal


```
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Calculate the number of rows needed
num_cols = 5  # Number of columns in the grid
num_plots = len(dataset.columns)
num_rows = math.ceil(num_plots / num_cols)

# Create a figure and axes with the desired grid layout
fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 5 * num_rows))
fig.tight_layout(pad=3.0)  # Add padding between subplots

# Flatten the axes array for easier iteration
axes = axes.ravel()

# Iterate through columns and plot
for i, col in enumerate(dataset.columns):
    ax = axes[i]  # Get the current subplot axis

    if dataset[col].dtype == 'object':
        sns.countplot(x=col, data=dataset, ax=ax)
        ax.set_title(f'Distribusi Kategorikal - {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frekuensi')
    else:
        sns.histplot(dataset[col], kde=True, bins=30, ax=ax)
        ax.set_title(f'Distribusi Numerik - {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frekuensi')

# Hide any unused subplots
for j in range(i + 1, num_plots):
    axes[j].axis('off')

plt.show()
```

> Unique Value

untuk mengetahui nilai yang unik dari setiap kolom 
```
import pandas as pd

def print_unique_values_in_dataframe(dataset):
    for column in dataset.columns:
        unique_values = dataset[column].unique()
        print(f"Column '{column}':\n{unique_values}\n")

# Assuming 'dataset' is your DataFrame, call the function:
print_unique_values_in_dataframe(dataset)

```
## Handling Missing Value
> cek missing value

```
dataset.isna().sum()
```
> cek rows yang menyebabkan kolom memiliki 1 missing value   

dilakukan untuk mengetahui apakah rows tersebut, unik atau hanya satu rows saja yang meneybebkan  

```
import pandas as pd
from IPython.display import display
pd.set_option('display.max_columns', None)                                      # Melihat semua fitur
missing_features = dataset.isna().sum()
missing_features = missing_features[missing_features == 1].index.tolist()       # Melihat fitur yang memuat 1 missing value

rows_to_drop = set()
# Menampilkan semua baris yang memiliki missing values pada fitur tersebut
for feature in missing_features:
    missing_rows = dataset[dataset[feature].isna()]
    print(f"\nMissing value in feature: {feature}")
    display(missing_rows)  # Display the DataFrame with the missing value
    rows_to_drop.update(missing_rows.index) 
```
> Grafik presentase missing value per kolom 
```
import pandas as pd
import matplotlib.pyplot as plt

mv = dataset.isna().sum()
percent_null = dataset.isna().mean() * 100

cek_missing_value = pd.concat([mv, percent_null], axis=1)
cek_missing_value.columns = ['Jumlah Missing', 'Persentase Missing']
cek_missing_value = cek_missing_value[cek_missing_value['Jumlah Missing'] > 0]

cek_missing_value = cek_missing_value.dropna()
cek_missing_value = cek_missing_value.sort_values(by='Persentase Missing', ascending=True)

if not cek_missing_value.empty:
    plt.figure(figsize=(25, 10))
    plt.barh(cek_missing_value.index, cek_missing_value['Persentase Missing'], color='skyblue')
    plt.xlabel('Persentase Missing (%)')
    plt.ylabel('Kolom')
    plt.title('Persentase Missing Value per Kolom')
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    for index, value in enumerate(cek_missing_value['Persentase Missing']):
        plt.text(value + 1, index, f"{value:.2f}%", va='center', fontsize=10)

    plt.show()
else:
    print("Tidak ada missing value dalam dataset.")

plt.show()
```
> Mengisi missing value dengan Regressi
Memilih hanya kolom numerik saja, menghitung nilai korelasi masing-masing fitur. Kemudian mengecek kolom mana yang memiliki missing value dan membandingkan fitur-fitur lainnya yang  berkorelasi untuk mengisi missing value tersebut. Mengisi missing value dengan median yang telah disesuaikan oleh median, apabila masih belum teratasi maka diisi dengan modus. 
```
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Pilih hanya kolom numerik dari 'dataset'
numerik_dataset = dataset.select_dtypes(include=['number'])

# Hitung matriks korelasi berdasarkan data numerik
matriks_korelasi = numerik_dataset.corr()

# Identifikasi kolom yang memiliki nilai yang hilang
nilai_hilang = dataset.isnull().sum()
kolom_hilang = nilai_hilang[nilai_hilang > 0].index

# Temukan fitur yang paling berkorelasi dengan setiap kolom yang memiliki nilai hilang
fitur_korelasi_teratas = matriks_korelasi[kolom_hilang.intersection(numerik_dataset.columns)]
korelasi = fitur_korelasi_teratas.apply(lambda x: x.abs().sort_values(ascending=False).index[1], axis=0)

# Mengisi nilai yang hilang menggunakan regresi linear
for target_kolom, referensi_kolom in korelasi.items():
    if target_kolom in dataset.columns and referensi_kolom in dataset.columns:
        # Pilih data yang tidak memiliki nilai kosong pada kedua kolom
        mask_valid = dataset[[target_kolom, referensi_kolom]].notnull().all(axis=1)
        X_train = dataset.loc[mask_valid, [referensi_kolom]].values.reshape(-1, 1)
        y_train = dataset.loc[mask_valid, target_kolom].values

        # Latih model regresi linear
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prediksi nilai yang hilang dengan menangani NaN dalam data uji
        mask_hilang = dataset[target_kolom].isnull()
        if mask_hilang.sum() > 0:
            X_test = dataset.loc[mask_hilang, [referensi_kolom]].values.reshape(-1, 1)

            # Gantilah NaN dalam X_test dengan median dari referensi_kolom
            X_test = np.nan_to_num(X_test, nan=np.nanmedian(dataset[referensi_kolom]))

            dataset.loc[mask_hilang, target_kolom] = model.predict(X_test)

# Mengisi sisa nilai numerik yang hilang dengan median
for kolom in kolom_hilang:
    if dataset[kolom].isnull().sum() > 0 and pd.api.types.is_numeric_dtype(dataset[kolom]) and not pd.api.types.is_categorical_dtype(dataset[kolom]):
        dataset[kolom].fillna(dataset[kolom].median(), inplace=True)

# Mengisi nilai kategori yang hilang dengan modus (nilai yang paling sering muncul)
kolom_kategorikal = dataset.select_dtypes(include=['object', 'category']).columns
for kolom in kolom_kategorikal:
    if dataset[kolom].isnull().sum() > 0:
        dataset[kolom].fillna(dataset[kolom].mode()[0], inplace=True)

# clean = dataset.copy()

```
> check missing value
re-check untuk melihat apakah semua sudah terisi, jika belum maka akan dilakukan perbaikan dalam proses pengisian missing value 

```
dataset.isna().sum()

```
## Modeling 
mendefinisikan kolom numerik untuk proses lebih lanjut 
```
numerik = dataset.select_dtypes(include=['int64', 'float64', 'int8'])
numerik
```


```
import seaborn as sns
import matplotlib.pyplot as plt


columns_to_impute = numerik.columns[numerik.isna().any()].tolist()

for column in columns_to_impute:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=column, hue="death", kde=True, bins=30)
    plt.title(f"Distribusi {column} berdasarkan Kategori Target (death)")
    plt.xlabel(column)
    plt.ylabel("Frekuensi")
    plt.legend(title="Death", labels=["0 - Tidak", "1 - Ya"])
    plt.show()
```

> Permodelan dengan KNN

Menyalin dataset numerik agar tidak berdampak pada dataset asli.Memisahkan kolom target dan fitur. Untuk presentase data testing kami mengambil proporsi 80:20 dengan rincian 80% data untuk training dan 20% untuk testing. Kami melakukan standarisasi fitur untuk menilai kelaykan fitur tersebut untuk model. Model knn dikombinasikan dengn hyperparameter tuning dengan grid search, kemudian mengambil hasil terbaik untuk dijadikan model. 

```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Salin dataset numerik
apt = numerik.copy()

# Memisahkan fitur dan target
X = apt.iloc[:, :-1].values  # Semua kolom kecuali kolom target
y = apt.iloc[:, -1].values   # Kolom target

# Mengubah target kontinu menjadi kategori berdasarkan median (binarisasi manual)
median_y = np.median(y)
y = np.where(y >= median_y, 1, 0)  # 1 jika >= median, 0 jika < median

# Membagi dataset (stratify untuk menjaga keseimbangan label)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Standarisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inisialisasi model KNN
knn = KNeighborsClassifier()

# Definisi parameter untuk tuning
param_dist = {
    'n_neighbors': np.arange(3,20),
    'weights': ['uniform', 'distance'],  # Bobot untuk tetangga
    'metric': ['euclidean', 'manhattan', 'minkowski'],  # Metode perhitungan jarak
    'p': [1, 2]  # Minkowski (p=1 Manhattan, p=2 Euclidean)
}

# Grid Search dengan Cross Validation (cv=5)
rndm = RandomizedSearchCV(knn, param_dist, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
rndm.fit(X_train, y_train)

# Gunakan model terbaik berdasarkan hasil tuning
best_knn = rndm.best_estimator_

# Prediksi pada data uji
y_pred = best_knn.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)

# Output hasil tuning
print(f'Best Params: {rndm.best_params_}')
print(f'Best Accuracy (CV): {rndm.best_score_:.2f}')
print(f'Final Accuracy on Test Set: {accuracy:.2f}')
```

