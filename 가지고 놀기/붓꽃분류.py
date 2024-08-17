#로지스틱 회귀를 사용하여 붓꽃을 분류하고 소프트맥스 함수를 사용하여 확률을 알아봅시다.

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax

iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

x = df[iris.feature_names]
y = df['species']

train_input, test_input, train_target, test_target = train_test_split(x, y,test_size=0.2, random_state=42)

ss = StandardScaler()
ss.fit(train_input)

train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


lr = LogisticRegression(C=10, max_iter=1000)
lr.fit(train_scaled, train_target)


decision = lr.decision_function(test_scaled[:5])
proba = softmax(decision, axis=1)


print("예측 클래스 : ", lr.predict(test_scaled[:5]))
print("확률 : ", np.round(proba, decimals=3))