#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline

"""
standartScaler
DecisionTree
Pipeline(standardScaler,DecisionTree)
"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

import warnings
warnings.filterwarnings("ignore")
#load and EDA
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df = df.drop(["id"], axis=1)

df.info()
describe = df.describe()

#stroke hedef değişken.
plt.figure()
sns.countplot(x = "stroke" ,data=df)
plt.title("Distribution of Stroke Class")
plt.show()
"""
0 inme yok 1 inme var
4800 ->0
250 ->1
Dengesiz veri seti

kcy:tüm sonuçlara 0 de Acc:4800/5100 =0.94 yanıltıcı sonuç
yanılmamak için: CM ,f1 score
dengesiz veri seti çözümü: Stroke (1) sayısını artırmak lazım , veri toplama
                           down sampling(0) sayısını azalt ,veri kaybı olur.
                           
"""
#Missing Value : DecisionTreeRegressor
df.isnull().sum()
"""
bmi:201
"""

DT_bmi_pipe = Pipeline(steps = [
    ("scale",StandardScaler()),#veri standartlaştırmak için standard scaler
    ("dtr",DecisionTreeRegressor()) #karar ağacı regesyon modeli
    ])

X = df[["gender","age","bmi"]].copy()
#gender sütununda bulunan değerleri sayısal değerlere dönüştürüyoruz.
#male ->0, female->1, other ->-1
X.gender = X.gender.replace ({"Male":0,"Female":1,"Other":-1}).astype(np.uint8)

#bmi değeri eksik olan nan olan satırları ayırç
missing = X[X.bmi.isna()]
#bmi değeri eksik olmayan verileri ayıralım
X = X[~X.bmi.isna()] #eksik olan değerlerin tersini ~ ile almış olduk.
y = X.pop("bmi")

#Modeli eksik olmayan veriler ile eğit.
DT_bmi_pipe.fit(X,y)
#eksik bmi değerlerini tahmin edelim.Tahmin yapılırken gender ve age kullanılabilecek.
predicted_bmi = pd.Series(DT_bmi_pipe.predict(missing[["gender","age"]]),index = missing.index)
df.loc[missing.index,"bmi"] =predicted_bmi

#Model Prediction:encoding,training and testing
df["gender"] = df["gender"].replace({"Male" : 0,"Female" :1 ,"Other" : -1}).astype(np.uint8)
df["Residence_type"] = df["Residence_type"].replace({"Rural" : 0,"Urban" :1}).astype(np.uint8)
df["work_type"] = df["work_type"].replace({"Private" : 0,"Self-employed" :1 ,"Govt_job" : 2 ,"children" : -1, "Never_worked":-2}).astype(np.uint8)

X= df[["gender","age","hypertension","heart_disease","work_type","avg_glucose_level","bmi"]]
y =df["stroke"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)

logreg_pipe = Pipeline(steps =[("scale",StandardScaler()),("LR",LogisticRegression())])
#model training 
logreg_pipe.fit(X_train,y_train)
#modelin testi
y_pred = logreg_pipe.predict(X_test)

print("Accuracy: ",accuracy_score(y_test,y_pred))
#Accuracy:  0.9452054794520548
print("cm : \n",confusion_matrix(y_test,y_pred))
"""
cm : 
 [[483   0]
 [ 28   0]]
"""
print("Classification Report :\n",classification_report(y_test,y_pred))
#Model Kaydetme ve Geri Yükleme ,gerçek hasta testi(olasılık göster- %90 ->1 ,%10 ->0)
"""
import joblib 
#modeli kaydetme

joblib.dump(logreg_pipe,"log_reg_model.pkl")
"""
#MODEL YÜKLEME
loaded_log_reg_pipe = joblib.load("log_reg_model.pkl")

# Yeni hasta verisi tahmin etme 
new_patient_data = pd.DataFrame({
    "gender": [1],
    "age": [45],
    "hypertension": [1],
    "heart_disease": [0],
    "work_type": [0],
    "avg_glucose_level": [70],
    "bmi": [25]
})


#tahmin
new_patient_data_result = loaded_log_reg_pipe.predict(new_patient_data)

#tahmin olasılıksal
new_patient_data_result_probability = loaded_log_reg_pipe.predict_proba(new_patient_data)

#%90 0 sınıfına %10 ihtimalle 1 sınıfına ait.













