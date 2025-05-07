# Gerekli kütüphaneleri içe aktar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Verileri oku
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Eğitim verisinin ilk 5 satırını ve boyutunu göster
print(train_data.head())   
print(train_data.shape)

# Çok fazla eksik veri içeren kolonları veri setinden çıkar
train_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)

# Sayısal kolonlardaki eksik verileri doldur
train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].mean())  # Ortalama ile doldurma
train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(0)                                  # Eksik taş kaplama alanını 0 ile doldurma
train_data['GarageYrBlt'] = train_data['GarageYrBlt'].fillna(0)                                # Eksik garaj yılı bilgilerini 0 ile doldurma

# Kategorik kolonlardaki eksik verileri 'None' ile doldur
for col in ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
            'BsmtFinType1', 'BsmtFinType2', 'GarageType', 
            'GarageFinish', 'GarageQual', 'GarageCond']:
    train_data[col] = train_data[col].fillna('None')

# 'Electrical' kolonunda eksik olan değeri en sık görülen değer ile doldur
train_data['Electrical'] = train_data['Electrical'].fillna(train_data['Electrical'].mode()[0])

# Eksik verileri kontrol etmek için kolon bazlı eksik değer sayılarını yazdır
print(train_data.isnull().sum().sort_values(ascending=False))
print(train_data.isnull().sum().sort_values(ascending=False))

train_data['FireplaceQu'] = train_data['FireplaceQu'].fillna('None')
print(train_data.isnull().sum().sum())
