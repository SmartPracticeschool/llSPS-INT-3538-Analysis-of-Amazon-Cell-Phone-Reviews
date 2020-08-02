import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
global classifier,graph

from flask import Flask , request, render_template 
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

graph =  tf.get_default_graph()

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('project.html')

@app.route('/login',methods = ['POST'])
def login():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    import re
    dataset=pd.read_csv(r'C:\Users\Washifa\Desktop\final.csv',engine='python')
    dataset.isnull().sum()
    dataset['body'] = dataset['body'].fillna('').apply(str)
    dataset['name'] = dataset['name'].fillna('').apply(str)
    dataset['title'] = dataset['title'].fillna('').apply(str)
    dataset['helpfulVotes'] = dataset['helpfulVotes'].fillna('').apply(str)
    c=[]
    for i in range(0, 67965):
        review=re.sub('[^a-zA-Z]','',dataset["body"][i])
        review=review.lower()
        review=review.split()
        review=[word for word in review if not word in set(stopwords.words('english'))]
        ps=PorterStemmer()
        review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review=' '.join(review)
        c.append(review)
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    dataset['helpfulVotes']=le.fit_transform(dataset['helpfulVotes'])
    from sklearn.feature_extraction.text import CountVectorizer
    cv=CountVectorizer(max_features=1500)
    x=cv.fit_transform(c).toarray()
    y=dataset.iloc[:,-1].values

    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    x_train=x_train.astype(float)
    y_train=y_train.astype(float)
    import tensorflow as tf
    from tensorflow.python.keras.layers import Input, Dense
    from tensorflow.keras import Sequential
    classifier= Sequential()
    classifier.add(Dense(units =1500,kernel_initializer ='uniform' , activation ='relu'))
    classifier.add(Dense(units =100,kernel_initializer ='uniform' , activation ='relu'))
    classifier.add(Dense(units = 1 ,kernel_initializer ='uniform' , activation ='relu'))
    classifier.compile(optimizer='adam',loss="binary_crossentropy", metrics=["accuracy"])
    classifier.fit(x_train,y_train, batch_size=5000,epochs=10)
    name=request.form['name']
    rating=request.form['rating']
    
    title=request.form['review']
    body=request.form['detailed review']
    total={"name":name,"rating":rating,"title":title,"body":body}
    
    d=[]
    for i in range(0,1):
        review=re.sub('[^a-zA-Z]','  ',total['body'])
        review=review.lower()
        review=review.split()
        review=[word for word in review if not word in set(stopwords.words('english'))]
        ps=PorterStemmer()
        review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review=' '.join(review)
        d.append(review)
    from sklearn.feature_extraction.text import CountVectorizer
    cv=CountVectorizer(max_features=1500)
    x=cv.fit_transform(d).toarray()
    x.resize((1500,1500),refcheck=False)
    y_pred=classifier.predict(np.array(x))
    y=y_pred[0][0]
    
   
    
       
    return render_template('project.html',abc=y)
if __name__ == '__main__':
    app.run(debug = False)
        
        
