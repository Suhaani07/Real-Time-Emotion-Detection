import cv2
from keras.models import model_from_json
import numpy as np

#from keras_preprocessing.image import load_img
#Loading our model
json_file = open('emotiondetector-74%.json','r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights('emotiondetector-74%.h5')
#using default cv2 haarcascade classifier for face detection
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

#extract feature to classify single realtime frame
def extract_features(image):
    feature=np.array(image)
    feature=feature.reshape(1,48,48,1)
    return feature/255.0

#starting webcam
webcam=cv2.VideoCapture(0)

#labels for emotions
labels={0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Sad',5:'Surprise',6:'Neutral'}

#detecting faces and classifying them
while True:
    #reading frames from webcam
    (_,frame)=webcam.read()
    #converting to grayscale
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #detecting faces
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    #predicting emotions
    for (x,y,w,h) in faces:
        #extracting face from frame
        face=gray[y:y+h,x:x+w]
        #drawing rectangle around face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #resizing face to 48x48 pixels
        face=cv2.resize(face,(48,48))
        #extracting features from face
        features=extract_features(face)
        #predicting emotion from extracted features
        prediction=model.predict(features)
        #label
        prediction_labels=labels[prediction.argmax()]
        #displaying emotion
        cv2.putText(frame,prediction_labels,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('Emotion Detector',frame)
    #press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break