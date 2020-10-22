import os
from flask import Flask, flash, request, redirect, url_for,render_template, Response,send_file
from werkzeug.utils import secure_filename
from Training_model import *
import numpy as np 

#initialiser flask
app = Flask(__name__,template_folder='templates',static_folder='C://Users/soukaina/Desktop/Object detection API/static')
#Creer le chemin pour enregistrer les images reçues 
app.config["IMAGE_UPLOADS"] = "C://Users/soukaina/Desktop/Object detection API/static/uploads"
#creation du endpoint 
@app.route("/")
def get_img():
    return render_template('index.html')

@app.route("/",methods=['GET','POST'])
def detect_object():
    if request.method == "POST":
        if request.files:
            #recevoir l'image inseree
            image = request.files["image"]
            #sécuriser le fichier 
            filename = secure_filename(image.filename)
            # enregistrer l'image dans le dossier /images/uploads
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
            #lire l'image
            image_ = cv2.imread("C://Users/soukaina/Desktop/Object detection API/static/uploads/"+filename)
            #appliquer la fonction de détection d'objets sur l'image
            get_prediction(image_,net,labels)
            #fonction de detection de la plaque d'immatriculation 
            res=getContours(image_)
            
            #enregistrer l'output dans le dossier /images/detections
            file_ = 'C://Users/soukaina/Desktop/Object detection API/static/detections/'+filename
            #afficher l'output
            cv2.imwrite(file_, res)

            
    return render_template("output.html", display_detection = filename, fname = filename)  


# exécuter l'API 
if __name__ == '__main__':
   app.run(port=4000, debug=True)