from flask import render_template, request, Flask
from keras.models import Model, load_model
from werkzeug.utils import secure_filename
import urllib.request
import datetime
import pickle
import cv2
import numpy as np
import os
import pandas as pd
from scipy import spatial
import re
import json

upload_folder = './static/uploads'
allowed_extensions = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

image_dimension = 200

app = Flask(__name__)  # create instance of Flask class
app.config['UPLOAD_FOLDER'] = upload_folder

feature_extractor = load_model('./static/models/feature_extractor.h5')
feature_extractor._make_predict_function()
# self.model = load_model('./static/models/feature_extractor.h5')
# self.model._make_predict_function()
# self.graph = tf.get_default_graph()

corpus_features = pickle.load(open('static/models/corpus_features.pickle', 'rb'))
corpus_metadata = pd.read_json('./static/models/corpus_metadata.json')
corpus_filenames = pickle.load(open('./static/models/corpus_filenames.pickle', 'rb'))

# with urllib.request.urlopen('http://artwork.ninja/static/models/corpus_features_000000_004999.pickle') as response:
#     with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
#         shutil.copyfileobj(response, tmp_file)
# corpus_features = pickle.load(open(tmp_file.name, 'rb'))


def image_to_3d_tensor(image_path):
    im = cv2.resize(cv2.imread(image_path), (image_dimension, image_dimension)).astype(np.float32)
    return im/255

def image_to_4d_tensor(image_path):
    im = image_to_3d_tensor(image_path)
    im = im.reshape(1, image_dimension, image_dimension, 3)
    return im

def compare_test_image(test_image_file, metric='cosine'):
    if os.path.isfile(test_image_file):
        print('File exists')
    else:
        print('No file')
    # Predict feature values for test image
    test_tensor = image_to_4d_tensor(test_image_file)
    print('Extracting features from test file...')
    test_features = feature_extractor.predict(test_tensor)
    print('Features extracted')
    # Images need to be converted to '2d' for use in distance.cdist.
    # First dimension is number of images (1 for test, corpus size for corpus).
    # Second dimension is width x height x number of color channels.
    test_features_2d = np.reshape(test_features, (test_features.shape[0], test_features.shape[1] * test_features.shape[2] * test_features.shape[3]))
    corpus_features_2d = np.reshape(corpus_features, (corpus_features.shape[0], corpus_features.shape[1] * corpus_features.shape[2] * corpus_features.shape[3]))
    distance_array = spatial.distance.cdist(test_features_2d, corpus_features_2d, metric=metric)
    print(type(distance_array))
    shift = 0
    if np.min(distance_array) < .001: 
        shift = 1
    print('corpus filenames length:', len(corpus_filenames))
    distancelist = distance_array.argsort()[0][0+shift:5+shift].tolist()
    print(distancelist)
    top5ids = [corpus_filenames[index][:-4] for index in distancelist]
    top5list = []
    print('Closest matches')
    for index, value in enumerate(top5ids):
        artwork = {}
        artwork['order'] = index
        artwork['image_url'] = corpus_metadata.loc[corpus_metadata['id'] == top5ids[index], 'image_url'].item()
        artwork['artist'] = corpus_metadata.loc[corpus_metadata['id'] == top5ids[index], 'artist'].item()
        artwork['title'] = corpus_metadata.loc[corpus_metadata['id'] == top5ids[index], 'title'].item()
        artwork['date'] = corpus_metadata.loc[corpus_metadata['id'] == top5ids[index], 'date'].item()
        artwork['medium'] = corpus_metadata.loc[corpus_metadata['id'] == top5ids[index], 'medium'].item()
        artwork['page_url'] = corpus_metadata.loc[corpus_metadata['id'] == top5ids[index], 'page_url'].item()
        artwork['resubmit'] = 'placeholder'
        top5list.append(artwork)
    return top5list


@app.route("/", methods=["POST", "GET"])
def viz_page():
    with open("static/html/index.html", 'r') as viz_file:
        return viz_file.read()


@app.route("/image", methods=["POST", "GET"])
def image():
    if request.method == 'POST':
        f = request.files['file']
        test_path_and_file = './static/uploads/' + ''.join(str(secure_filename(f.filename)).strip().split())
        print(test_path_and_file)
        # test_path_and_file = './static/uploads/upload.jpg'
        f.save(test_path_and_file)
        # Code goes here to function to put file through model and find similar images
        top5list = compare_test_image(test_path_and_file)
        print('Got here')
        return render_template('results.html',
            original_image = test_path_and_file, 
            rec0_image = top5list[0]['image_url'],
            rec0_artist = top5list[0]['artist'],
            rec0_title = top5list[0]['title'],
            rec0_date = top5list[0]['date'],
            rec0_medium = top5list[0]['medium'],
            rec0_page = top5list[0]['page_url'],
            rec0_resubmit = 'resubmit',
            rec1_image = top5list[1]['image_url'],
            rec1_artist = top5list[1]['artist'],
            rec1_title = top5list[1]['title'],
            rec1_date = top5list[1]['date'],
            rec1_medium = top5list[1]['medium'],
            rec1_page = top5list[1]['page_url'],
            rec1_resubmit = 'resubmit',
            rec2_image = top5list[2]['image_url'],
            rec2_artist = top5list[2]['artist'],
            rec2_title = top5list[2]['title'],
            rec2_date = top5list[2]['date'],
            rec2_medium = top5list[2]['medium'],
            rec2_page = top5list[2]['page_url'],
            rec2_resubmit = 'resubmit',
            rec3_image = top5list[3]['image_url'],
            rec3_artist = top5list[3]['artist'],
            rec3_title = top5list[3]['title'],
            rec3_date = top5list[3]['date'],
            rec3_medium = top5list[3]['medium'],
            rec3_page = top5list[3]['page_url'],
            rec3_resubmit = 'resubmit',
            rec4_image = top5list[4]['image_url'],
            rec4_artist = top5list[4]['artist'],
            rec4_title = top5list[4]['title'],
            rec4_date = top5list[4]['date'],
            rec4_medium = top5list[4]['medium'],
            rec4_page = top5list[4]['page_url'],
            rec4_resubmit = 'resubmit',
        )
        # return 'Local image uploaded successfully'
        # myfile = requests.args.get('myfile')
    else:
        return 'Not POST ... unsuccessful'


@app.route("/url", methods=["POST", "GET"])
def url():
    if request.method == 'POST':
        url = request.form.get('myurl')
        test_filename = str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")) + ' ' + url.split('/')[-1]
        test_path_and_file = './static/uploads/' + test_filename
        urllib.request.urlretrieve(url, test_path_and_file)

        # Code goes here to function to put file through model and find similar images
        top5list = compare_test_image(test_path_and_file)

        return render_template('results.html', 
            original_image = url,
            rec0_image = top5list[0]['image_url'],
            rec0_artist = top5list[0]['artist'],
            rec0_title = top5list[0]['title'],
            rec0_date = top5list[0]['date'],
            rec0_medium = top5list[0]['medium'],
            rec0_page = top5list[0]['page_url'],
            rec0_resubmit = 'resubmit',
            rec1_image = top5list[1]['image_url'],
            rec1_artist = top5list[1]['artist'],
            rec1_title = top5list[1]['title'],
            rec1_date = top5list[1]['date'],
            rec1_medium = top5list[1]['medium'],
            rec1_page = top5list[1]['page_url'],
            rec1_resubmit = 'resubmit',
            rec2_image = top5list[2]['image_url'],
            rec2_artist = top5list[2]['artist'],
            rec2_title = top5list[2]['title'],
            rec2_date = top5list[2]['date'],
            rec2_medium = top5list[2]['medium'],
            rec2_page = top5list[2]['page_url'],
            rec2_resubmit = 'resubmit',
            rec3_image = top5list[3]['image_url'],
            rec3_artist = top5list[3]['artist'],
            rec3_title = top5list[3]['title'],
            rec3_date = top5list[3]['date'],
            rec3_medium = top5list[3]['medium'],
            rec3_page = top5list[3]['page_url'],
            rec3_resubmit = 'resubmit',
            rec4_image = top5list[4]['image_url'],
            rec4_artist = top5list[4]['artist'],
            rec4_title = top5list[4]['title'],
            rec4_date = top5list[4]['date'],
            rec4_medium = top5list[4]['medium'],
            rec4_page = top5list[4]['page_url'],
            rec4_resubmit = 'resubmit',
        )

    else:
        return 'Not POST'


if __name__ == '__main__':
    app.debug = True
    app.run()
