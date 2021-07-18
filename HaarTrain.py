import os
import numpy as np
from PIL import Image
import cv2
import pickle
Base_dir=os.path.dirname(os.path.abspath(__file__))
img_dir=os.path.join(Base_dir,"ImgOur")
face_cascade=cv2.CascadeClassifier('src\cascades\data\haarcascade_frontalface_alt2.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()

current_id=0
label_id={}
y_labels=[]
x_train=[]

for root,dirs,files in os.walk(img_dir):
    for file in files:
        if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("bmp"):
            path=os.path.join(root,file)
            label=os.path.basename(root).replace(" ", "-").lower()
            #print(label,path)
            if not label in label_id:
                label_id[label]=current_id
                current_id+=1
                id_=label_id[label]
               # print(label_id)
            # y.labels.append(label)
            #x-train.append(path)
            pill_image=Image.open(path).convert("L")
            image_array=np.array(pill_image,"uint8")
           # print(image_array)
            faces=face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)
            for (x,y,w,h)  in faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
#print(y_labels)
#print(x_train)
                with open("labels.pickle",'wb') as f:
                    pickle.dump(label_id,f)
recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainner.xml")