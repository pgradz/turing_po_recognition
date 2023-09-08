import cv2
import pickle
from sklearn.preprocessing import Normalizer
from vgg import extract_face, VGG_embeeding
import pathlib
from keras_vggface.vggface import VGGFace


output_dir = pathlib.Path.cwd().joinpath('output')
model_dir = pathlib.Path.cwd().joinpath('model')
opencv_dir = pathlib.Path.cwd().joinpath('opencv_frames')

po_model_filename = str(model_dir.joinpath('PO_recognition.sav'))
po_model = pickle.load(open(po_model_filename, 'rb'))
model = VGGFace(model= "resnet50" , include_top=False, input_shape=(224, 224, 3), pooling= "avg" )
out_encoder = pickle.load(open(str(model_dir.joinpath('out_encoder')), 'rb'))

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cam = cv2.VideoCapture(0)
cv2.namedWindow("PO recognition")
img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        break
    k = cv2.waitKey(1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        img = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        pass
        # SPACE pressed
        # if we want to save captures
        # img_name = str(opencv_dir.joinpath("opencv_frame_{}.png".format(img_counter)))
        # cv2.imwrite(img_name, frame)
        # print("{} written!".format(img_name))
        # image = Image.open(img_name)
    try:
        face = extract_face(frame[y:y+h,x:x+w]) 
        embedding = VGG_embeeding(face)
        yhat_class = po_model.predict(embedding)
#       yhat_prob = svm_model.predict_proba(embedding)
        predict_names = out_encoder.inverse_transform(yhat_class)[0]
        print(predict_names)
        cv2.putText(frame, predict_names, 
                    (x+5,y-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (0, 255, 255), 
                    2, 
                    cv2.LINE_4)
    
    except IndexError:
        pass

    cv2.imshow('video', frame)
    
    img_counter += 1


cam.release()

cv2.destroyAllWindows()
