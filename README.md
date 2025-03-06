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


```

```

```

```

```

```

```

```

```

```

```

```
