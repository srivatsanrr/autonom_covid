import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import joblib
from firebase_admin import firestore
import qrcode
from sklearn import preprocessing
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier as rf
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
import json
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from google.cloud.exceptions import NotFound
import firebase_admin
from firebase_admin import credentials

def data_preprocessor(path):
    # Reading data from file
    dat=pd.read_excel(path)
    # remove unnecessary cols
    names=dat['Aadhaar'].values
    mobs=dat['Mobile']
    email=dat['Email'].values
    dat=dat.drop(["Name","Insurance", "salary", "people_ID", 'Aadhaar', 'Mobile', 'Email'],  axis=1)
    # Label Encoding
    dat["Region"] = dat["Region"].astype('category').cat.codes
    dat["Gender"] = dat["Gender"].astype('category').cat.codes
    dat["Designation"] = dat["Designation"].astype('category').cat.codes
    dat["Married"] = dat["Married"].astype('category').cat.codes
    dat["Mode_transport"] = dat["Mode_transport"].astype('category').cat.codes
    dat["Occupation"] = dat["Occupation"].astype('category').cat.codes
    dat["Married"] = dat["Married"].astype('category').cat.codes
    dat["comorbidity"] = dat["comorbidity"].astype('category').cat.codes
    dat["Pulmonary score"] = dat["Pulmonary score"].astype('category').cat.codes
    dat["cardiological pressure"] = dat["cardiological pressure"].astype('category').cat.codes
    
        # Missing value handling using mean and forward fill methods
    dat['Children']=dat['Children'].fillna(int(np.mean(dat['Children'])))
    dat['Diuresis']=dat['Diuresis'].fillna(int(np.mean(dat['Diuresis'])))
    dat['d-dimer']=dat['d-dimer'].fillna(int(np.mean(dat['d-dimer'])))
    dat['Heart rate']=dat['Heart rate'].fillna(int(np.mean(dat['Heart rate'])))
    dat['Platelets']=dat['Platelets'].fillna(int(np.mean(dat['Platelets'])))
    dat['HDL cholesterol']=dat['HDL cholesterol'].fillna(int(np.mean(dat['HDL cholesterol'])))
    dat['HBB']=dat['HBB'].fillna(int(np.mean(dat['HBB'])))
    dat['FT/month']=dat['FT/month'].fillna(int(np.mean(dat['FT/month'])))
    dat['comorbidity'].fillna('ffill', inplace=True)
    dat['cardiological pressure'].fillna('ffill', inplace=True)
    # print(dat.head())
    # PCA
    ncomp=6
    pca_cov = PCA(n_components=ncomp)
    principalComponents = pca_cov.fit_transform(dat.iloc[:,9:])
    principaldf=pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2','pc3','pc4','pc5','pc6'])
    X=principaldf.values
    # For single sample inference
    if(X.shape[0]==1):
        X=X.reshape(1,-1)
    return([X, names, mobs, email])


def inference(X,names,mobs,email,model): 
    clf = joblib.load(model)
    y=clf.predict(X)
    values = ['High', 'Low', 'Mid']
    y_labels= [values[i] for i in y]
    names=[str(i) for i in names]
    mob=[str(i) for i in mobs]

    l=len(names)
    indices=[i for i in range(len(names))]
##  Index: AAdhar number, Mobile Number, Risk Factor
    ret_dict=dict(zip(indices,zip(names,mob,y_labels)))
    ret_dict.update( {'length' : l})
    ret_json= json.dumps(ret_dict)
    ret1=tuple(zip(email,y_labels))
    return([ret1,ret_json])

def qrgen(uid, mob, health_flag):
   ret_dict={uid:[mob,health_flag]}
   ret_json= json.dumps(ret_dict)
   print(ret_json)
   retstr=str(ret_json)
   img=qrcode.make(retstr)
   return img


'''def main(path, model_path='model_sar.pkl'):
    (X,names)=data_preprocessor(path)
    return inference(X,names, model_path)'''
def add_risk_score(ret1):
    db = firestore.client()
    # (emails,y_labs)=ret1

    for (email,y_lab) in ret1:
      # docs=db.collection(str(email)).stream()
      try:
        doc_ref=db.collection(email).document()  
        if(doc_ref!=None):
          doc_ref.set({u'Status':y_lab})
      except NotFound:
        continue

@app.route('/main', methods=['GET'])
def main():
    cred = credentials.Certificate("samhar-21151-firebase-adminsdk-w4vxj-35492734bd.json")
    firebase_admin.initialize_app(cred)
    (X,names, mobs, email)=data_preprocessor('Train_Mobile.xlsx')
    [ret1, retjson]=inference(X,names, mobs, email,'model_sar.pkl')
    add_risk_score(ret1)
    return retjson



if __name__ == "__main__":
	app.run(host='192.168.43.129', debug=True)
