from math import cos, asin, sqrt
from opencage.geocoder import OpenCageGeocode
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from google.cloud.exceptions import NotFound
import firebase_admin
from firebase_admin import credentials, firestore
cred = credentials.Certificate("samhar-21151-firebase-adminsdk-w4vxj-3d5cbb7790.json")
firebase_admin.initialize_app(cred)
path='Train_pincode.xlsx' # Change accordingly
def amIHome(myLoc,refLoc):
  (lat1, lon1, lat2, lon2)=(myLoc[0],myLoc[1], refLoc[0],refLoc[1])
  p = 0.017453292519943295
  a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p))/2
  dist=12742 * asin(sqrt(a))
  if dist<=5:
    flg=True
  else:
    flg=False
  return flg

def findMyHome(pincode='612104'): 
  key='858c794be631439582229839e2816bd4'
  geocoder = OpenCageGeocode(key)
  results = geocoder.geocode(pincode)
  refLoc=[results[0]['geometry']['lat'],results[0]['geometry']['lng']]
  return refLoc

def getPincodeEmail(email):
  dat=pd.read_excel(path)
  pincode=str(dat[dat['Email']==email]['Pincode'].values[0])
  return pincode

def getPincodeAadhaar(uid):
  dat=pd.read_excel(path)
  pincode=str(dat[dat['Aadhaar']==email]['Pincode'].values[0])
  return pincode

def addIfHome(email, flg):
  db = firestore.client()
  try:
    doc_ref=db.collection(email).document('doc')
    if(doc_ref!=None):
      doc_ref.update({u'isHome':flg})
  except NotFound:
    pass


