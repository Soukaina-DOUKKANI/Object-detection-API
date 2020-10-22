#importation des bibliotheques 
import cv2
import numpy as np
import pytesseract 
#importation de Tesseract OCR pour la lecture du texte 
pytesseract.pytesseract.tesseract_cmd='C://Program Files (x86)/Tesseract-OCR/tesseract.exe'
#Importation de YOLO et coco.names
yolov3_weights="YOLO_algorithm/yolov3.weights"
yolov3_cfg="YOLO_algorithm/yolov3.cfg"
labels="YOLO_algorithm/coco.names"

# importation de l'algorithme YOLO
net = cv2.dnn.readNet(yolov3_weights, yolov3_cfg)
# obtention des classes d'objets
classes = []
with open(labels, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# obtention des couches de sortie de l'algorithme
layer_names = net.getLayerNames()
#obtention de l'indice de chaque couche
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#fonction de detection d'objets
def get_prediction(image,net,labels):
    (H, W) = image.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                conf=confidence
                classIDs.append(classID)

    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.1)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in idxs:
            x, y, w, h = boxes[i]
            # obtention du nom de l'objet
            label = str(classes[classIDs[i]])
            # bounding box pour entourer l'objet
            cv2.rectangle(image, (x, y), (x + w, y + h), (255,0,0), 3)
            # insérer le nom et le niveau de confiance  de l'objet
            cv2.putText(image, label, (x, y + 30), font, 1.5, (255,0,0), 2)
            cv2.putText(image, f' {int(conf*100)}%', (x, y + h), font, 1.5, (255,0,0), 2)

    return image


#fonction de detection de la plaque
def getContours(image):
    # utilisation de la fonction findContours() qui cherche les contours des objets
    # RETR_EXTERNAL:  méthode de récupération des contours d'objets
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    # appliquer le filtre de canny pour détecter les coutours
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        # chercher la surface du contour
        area = cv2.contourArea(cnt)
        # filtrer l'image recue pour garder les zones qui peuvent contenir un objet
        if area > 500:
            # calculer le perimètre du contour
            peri = cv2.arcLength(cnt, True)
            # chercher les coins des contours afin de déterminer la forme de l'objet (ex: un triangle a trois coins)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # calculer le nombre de coins de l'objet
            #print(len(approx))
            # obtention des coins des objets
            objCor = len(approx)
            # determiner les coordonnees
            x, y, w, h = cv2.boundingRect(approx)
            # détection de l'objet 'plaque d'immatriculation
            # la plaque est sous forme d'un rectangle ayant 4 coins
            if objCor == 4 and w>2*h:
                # width est différente de height
                longlarg = w / h
                if longlarg != 1:
                    objectType = "plaque d'immatriculation"
            else:
                continue
                       
            plaque = image[y:y + h, x:x + w]
            # bounding box pour détecter la plaque
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
            #lire le contenu de la plaque
            content=pytesseract.image_to_string(plaque)
            print("le contenu de la plaque est:",content)
            
            # nommer l'objet
            cv2.putText(image,objectType, (x , y -h//4 ), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 0))

    return image