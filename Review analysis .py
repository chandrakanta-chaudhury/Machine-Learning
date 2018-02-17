@author: chandrakanta chaudhury
"""

import numpy as np

from urllib.request import  urlopen as ureq

from bs4 import BeautifulSoup

import pandas as pd

import re
import nltk
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer 
nltk.download('stopwords')

containerlist=[]
i=1
for i in range(25):      # Number of pages plus one 
    test11= "https://www.trustpilot.com/review/www.sonetel.com?page="+(str(i))
    uclient = ureq(test11)
    page_html = uclient.read()
    uclient.close()
    page_soup= BeautifulSoup(page_html,"html.parser")
    container = page_soup.find_all("div", {"class" :"review-body"})    
    contain=container[0:-1]    
    containerlist.extend(contain)
    i=i+1
#    print(len(contain))
    
print (len(containerlist))
len(contain)

 

labels=['review1','review2','review3','review4','review5','review6','review7','review8','review9','review10',
        'review11','review12','review13','review14','review15','review16','review17','review18','review19','review20',
        'review21','review22','review23','review24','review25','review26','review27','review28','review29','review30','review31']
df=pd.DataFrame.from_records(containerlist,columns =labels)

 
df.shape

#cleaning texts

corpus =[]

for i in range(0,466):
        review11=re.sub('[^a-zA-Z]', ' ',df['review1'][i].strip())
        review11=review11.lower()
        review11 = review11.split()
    #review11 = [word for word in review11 if not word in set(stopwords.words('english'))]
        ps= PorterStemmer()
        review11=[ps.stem(word) for word in review11 if not word in set(stopwords.words('english'))]
        review11 = ' '.join(review11)
        corpus.append(review11)



#getting review ratings from users 
user_rating=[]
#s=1
for s in range(24):
   test11= "https://www.trustpilot.com/review/www.sonetel.com?page="+(str(s))
   uclient = ureq(test11)
   page_html = uclient.read()
   uclient.close()
   for i in range(0,20):
       page_soup= BeautifulSoup(page_html,"html.parser")
       ratings1 = page_soup.find_all("div", {"class":"social-share-network social-share-network--facebook"})
       ratings2 = page_soup.find_all("div", {"class":"social-share-network social-share-network--google"})
       ratings3 = page_soup.find_all("div", {"class":"social-share-network social-share-network--twitter"})
        
        #ratings = page_soup.find_all("div", {"data-status" :" "})
            
       ratingno1=ratings1[i]
       ratingno2=ratings2[i]
       ratingno3=ratings3[i]
       resstring1=str(ratingno1)
       resstring2=str(ratingno2)
       resstring3=str(ratingno3)
       i=i+1
       try :
          check_rating1 =int(re.search(r'\d+',resstring1).group())
       except AttributeError :
      #       print ("bypass no ratings")
          check_rating1=0
        
       try :
          check_rating2 =int(re.search(r'\d+',resstring2).group())
       except AttributeError :
     #       print ("bypass no ratings")
         check_rating2=0
            
       try :
         check_rating3 =int(re.search(r'\d+',resstring3).group())
       except AttributeError :
    #        print ("bypass no ratings")
         check_rating3 =0
        
       Actual_rating=max(check_rating1,check_rating2,check_rating3)
        
       user_rating.append(Actual_rating)
   s=s+1


#creating bag of words
from sklearn.feature_extraction.text  import CountVectorizer
cv=CountVectorizer(max_features=1000)
x=cv.fit_transform(corpus).toarray()

y=((pd.DataFrame(user_rating).values)/5)
#fixing label issue , so convert to string type 
y=np.asarray(y,dtype=str)

y=y[0:466]

#label encoding rating 0 -1 
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)
le.classes_

#split data into train and test
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#feature scaling 

from sklearn.preprocessing  import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#gaussian Baive bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

classifier.fit(x_train,y_train)

ypred =classifier.predict(x_test)

from sklearn.metrics import  confusion_matrix
cm=confusion_matrix(y_test,ypred)

#gaussion naive gives less accuracy - 32%

from sklearn.svm import SVC
svcclassfier = SVC(kernel = 'rbf', random_state =0)
svcclassfier.fit(x_train,y_train)
ypred2 =svcclassfier.predict(x_test)
cm=confusion_matrix(y_test,ypred2)

#svM classifier gives  46.15%

#KNN 35.04%
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)

neigh.fit(x_train,y_train)

ypred1 =neigh.predict(x_test)

cm=confusion_matrix(y_test,ypred1)

#XGB classfier

from xgboost  import XGBClassifier
xgbclassifier =XGBClassifier()
xgbclassifier.fit(x_train,y_train)
ypred1 =xgbclassifier.predict(x_test)
cm=confusion_matrix(y_test,ypred1)


from sklearn.model_selection import  cross_val_score
accuracy = cross_val_score(estimator=xgbclassifier,X=x_train,y=y_train,cv=10)
accuracy.mean()
