from mtcnn.mtcnn import MTCNN
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
from numpy import asarray
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import pathlib
from numpy import savez_compressed
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle


pictures_dir = pathlib.Path.cwd().joinpath('pictures')
pictures_res = pathlib.Path.cwd().joinpath('pictures_boxed')
model_dir = pathlib.Path.cwd().joinpath('model')

model = VGGFace(model= "resnet50" , include_top=False, input_shape=(224, 224, 3), pooling= "avg" )


def extract_face(image):
    '''
    extracts face from the image
    :param image
    :type:
    '''
    
    pixels = asarray(image)
    plt.axis("off")
    plt.imshow(pixels)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]["box"]
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = asarray(image)
    return face_array, y1, y2, x1, x2


def load_faces(directory):
    """
    :param directory from which to import images
    :type: pathlib.Path

    :return: a list of arrays with extracted faces
    :rtype: list
    """
    faces = list()
    # enumerate files
    for filename in directory.glob('*.jpeg'):
        image =  Image.open(filename)
        face,_,_,_,_, = extract_face(image)
        if face is None:
            continue
        # store
        faces.append(face)
    return faces


def load_dataset(directory):
    '''
    :param pathlib.Path
    traverse a directory and loads images from subdirectories using their names as labels
    '''
    X, y = list(), list()
    # enumerate folders, on per class
    for subdirectory in directory.iterdir():
        # load all faces in the subdirectory
        faces = load_faces(subdirectory)
        # create labels
        labels = [subdirectory.name for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdirectory.name))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)


def get_train_test_embeedings():
    ''' provides training and test numpy array with vgg embeedings'''

    train_x, train_y = load_dataset(pictures_dir.joinpath('train'))
    print(train_x.shape, train_y.shape)
    test_x, test_y = load_dataset(pictures_dir.joinpath('test'))
    # save arrays to one file in compressed format
    print("dataset loaded!")
    print("creating embeddings...")
    # # convert each face in the train set to an embedding
    new_train_x = list()
    for face_pixels in train_x:
        embedding = VGG_embeeding(face_pixels)
        new_train_x.append(embedding)
    new_train_x = asarray(new_train_x)
    print(new_train_x.shape)
    # convert each face in the test set to an embedding
    new_test_x = list()
    for face_pixels in test_x:
        embedding = VGG_embeeding(face_pixels)
        new_test_x.append(embedding)
    new_test_x = asarray(new_test_x)
    print(new_test_x.shape)
    new_train_x = np.squeeze(new_train_x)
    new_test_x = np.squeeze(new_test_x)

    return new_train_x, train_y, new_test_x, test_y

def VGG_embeeding(face_pixels):
    # scale pixel values
    samples = asarray(face_pixels, "float32")
    samples = preprocess_input(samples, version=2)
    samples = np.expand_dims(samples, 0)
    embeddings = model.predict(samples)

    return embeddings


if __name__ == '__main__':

    new_train_x, train_y, new_test_x, test_y = get_train_test_embeedings()

    out_encoder = LabelEncoder()
    out_encoder.fit(train_y)
    pickle.dump(out_encoder, open(str(model_dir.joinpath('out_encoder')), 'wb'))
    train_y = out_encoder.transform(train_y)
    test_y = out_encoder.transform(test_y)
    # fit model
    model_svc = SVC(kernel='linear', probability=True)
    model_svc.fit(new_train_x, train_y)

    yhat_train = model_svc.predict(new_train_x)
    yhat_test = model_svc.predict(new_test_x)
    # score
    score_train = accuracy_score(train_y, yhat_train)
    score_test = accuracy_score(test_y, yhat_test)
    # summarize
    print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))

    filename = str(model_dir.joinpath('PO_recognition.sav'))
    pickle.dump(model_svc, open(filename, 'wb'))