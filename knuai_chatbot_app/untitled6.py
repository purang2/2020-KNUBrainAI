# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:43:24 2020

@author: puran
"""

# [KNN 알고리즘]
# 농구선수 포지션 예측 실습을 변형해서 대경권 train 돌려보기

import pandas as pd
#데이터 시각화에 필요한 라이브러리 2개
import matplotlib.pyplot as plt
import seaborn as sns
#사이킷런의 train_test_split을 사용하면 
#코드 한줄로 손쉽게 학습데이터/테스트데이터 나눌 수 있다.
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
#테스트를 위한 라이브러리 임포트
from sklearn.metrics import accuracy_score

%matplotlib inline
#샘플 데이터 수집

#df=pd.read_csv("https://raw.githubusercontent.com/wikibook/machine-learning/2.0/data/csv/basketball_stat.csv")
df=pd.read_csv("train.csv")

#수집한 데이터 샘플 확인하기
df.head() 

#현재 데이터에서 포지션 개수 확인
 #df.Pos.value_counts()

#Seaborn의 2차원 그래프를 사용한 데이터 시각화

df["주행거리"] = df["주행거리"].str.replace(pat='km', repl='', regex=False)
df["주행거리"] = df["주행거리"].str.replace(pat='천', repl='000', regex=False)
df["주행거리"] = df["주행거리"].str.replace(pat='만', repl='0000', regex=False)

#ml ->1마일 to 1.609344 킬로미터

#df["주행거리"].str.replace(pat='ml',repl='',regex=False) 
#df["주행거리"]=pd.to_numeric(df["주행거리"])


    
for index in range(11769):
    if(df["주행거리"][index].find("ml")!=-1):
        df["주행거리"][index]=df["주행거리"][index].replace("ml","")        
        df["주행거리"][index]= int(df["주행거리"][index]) *1.609344   
    elif(df["주행거리"][index].find("등록")!=-1):
        df["주행거리"][index] =str(0)
      
#all to float        
df["주행거리"]=pd.to_numeric(df["주행거리"])  
      

df["연월"]=df["연월"].str.slice(start=0,stop=5)


for index in range(11769):
    read_year, read_month = df["연월"][index].split('/')
    read_year=int(read_year)
    read_month=int(read_month)
    if(read_year>30):
        read_year = read_year -72
    else:
        read_year = read_year +100 - 72
    
    df["연월"][index] = read_year*12 + read_month    




#신차가격 없는 갯수 : 2225개/11769개 중
print(df["신차가(만원)"].isnull().sum())

##전략 

## 1차) 신차가격 없는거 버리고 돌리기
## 2차) 적절해 신차가격을 예측해서 다 돌리기




'''
sns.lmplot('연식','신차가(만원)',data=df, fit_reg=False, #x축 y축 데이터, 노라인
           scatter_kws={"s": 150}, #좌표 상의 점의 크기
           markers=["o","x"],
           hue="가격(만원)") #예측값

#타이틀
#plt.title('STL and 2P in 2d plane')


sns.lmplot('주행거리','연식',data=df, fit_reg=False, #x축 y축 데이터, 노라인
           scatter_kws={"s": 150}, #좌표 상의 점의 크기
           markers=["o","x"],
           hue="가격(만원)") #예측값

'''

'''
#타이틀
#plt.title('BLK and 3P in 2d plane')

#분별력이 없는 특징(feature)을 제거함
#df.drop(['2P','AST','STL'],axis=1,inplace=True)

#데이터프레임 조회
 #df.head()

#다듬어진 데이터에서 20%를 테스트 데이터로 분류
train, test = train_test_split(df,test_size=0.2)

#다듬어진 데이터 개수
#train.shape[0]
#test.shape[0]

#최적의 k를 찾기 위해 교차 검증을 수행할 k의 범위를 3부터 학습 데이터의 절반까지 지정
max_k_range= train.shape[0] //2
k_list=[]
for i in range(3,max_k_range,2):
    k_list.append(i)

cross_validation_scores = []
x_train = train[['3P','BLK','TRB']]
y_train = train[['Pos']]


#교차 검증(10-fold)을 각 k를 대상으로 수행해 검증 결과를 저장
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train.values.ravel(), 
                             cv=10,scoring='accuracy')
    cross_validation_scores.append(scores.mean())

'''
'''
#k에 따른 정확도를 시각화
plt.plot(k_list, cross_validation_scores)
plt.xlabel('the number of k')
plt.ylabel('Accuracy')
plt.show()
'''

'''
#가장 예측률이 높은 k를 선정
k= k_list[cross_validation_scores.index(max(cross_validation_scores))]
#print("The best number of k : "+str(k))

knn=KNeighborsClassifier(n_neighbors=k)

x_train[['3P','BLK','TRB']]
y_train[['Pos']]

#kNN 모델학습
knn.fit(x_train,y_train.values.ravel())

#테스트 데이터에서 분류를 위해 사용될 속성을 지정
x_test = test[['3P','BLK','TRB']]

#선수 포지션에 대한 정답을 지정
y_test = test[['Pos']]

#테스트 시작
pred=knn.predict(x_test)

#모델 예측 정확도(accuracy) 출력
print("accuracy : " + str(accuracy_score(y_test.values.ravel(),pred)) )

#예측 값 눈으로 보기
comparison = pd.DataFrame({'prediction':pred, 'ground_truth':y_test.values.ravel()})

print(comparison)


'''