# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values   #verisetinden yeni veriseti oluşturduk -1 ile sondakini dahil etmedik
y = dataset.iloc[:,3].values #sondakini ayrı eksen olarak aldık

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
#fit yanlnızca hesaplar transform uygular fit_transform ikisini birlikte yapar
imputer = imputer.fit(x[:,1:3])  #x içinde nan olan 2 ve 3. satırı bul 1:3 demek 2 ve 3. satır demek 4. dahil değil demek
x[:, 1:3] = imputer.transform(x[:, 1:3]) #xteki nanları meanler ile doldurü

#encoding categorical data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

#xteki ülkeleri dummy variable ile encode ediyoruz
#bunun amacı her ülkenin eşit ağırlığa sahip olmasını sağlamak
#ann tarafına geçtiğimizde ispanyanın isminden dolayı katsayısı 3 olmasın diye yani
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])],remainder = 'passthrough')
x = np.array(ct.fit_transform(x),dtype=np.float)

#y değerlerini de bu komutla encode ediyoruz
y = LabelEncoder().fit_transform(y)


#split training and test
#genelde 0.2 0.25 0.3 zor ihtimal 0.4 kullanılır
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)


#feature scaling
#age ile salary aynı scalede değil bunlar sorun çıkarabiliyor
#standardization yada normalization
"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""




