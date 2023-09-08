import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageFont, ImageDraw
from sklearn.preprocessing import Normalizer
from vgg import extract_face, VGG_embeeding
import pathlib
from keras_vggface.vggface import VGGFace

# provide location to a file
file = '' 
pictures_dir = pathlib.Path.cwd().joinpath('pictures')
pictures_res = pathlib.Path.cwd().joinpath('pictures_boxed')
output_dir = pathlib.Path.cwd().joinpath('output')
model_dir = pathlib.Path.cwd().joinpath('model')

po_model_filename = str(model_dir.joinpath('PO_recognition.sav'))
po_model = pickle.load(open(po_model_filename, 'rb'))
model = VGGFace(model= "resnet50" , include_top=False, input_shape=(224, 224, 3), pooling= "avg" )
out_encoder = pickle.load(open(str(model_dir.joinpath('out_encoder')), 'rb'))



if __name__ == '__main__':

    image =  Image.open(file)
    face,  y1, y2, x1, x2 = extract_face(image)
    embedding = VGG_embeeding(face)
    yhat_class = po_model.predict(embedding)
    yhat_prob = po_model.predict_proba(embedding)
    predict_names = out_encoder.inverse_transform(yhat_class)[0]
    print(predict_names)
    image =  Image.open(file)
    
    font = ImageFont.truetype("Arial", 30)
    draw = ImageDraw.Draw(image)
    draw.text((x1,y1-35), text=predict_names, font=font, fill="green")
    draw.rectangle(((x1, y1), (x2, y2)), fill=None, outline='green', width = 5)
    image.show()
 
