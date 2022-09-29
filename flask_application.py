import os
import time
import random
import warnings
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## Import TensorFlow modules
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

## Import Flask, and its extras
from flask import Flask, request

## Set warning and log levels
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR') # Suppress TensorFlow logging (2)
warnings.filterwarnings('ignore') # Suppress Matplotlib warnings

## Enable GPU dynamic memory allocation
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

## Define data and model paths
WORKING_DIR = os.getcwd()
PATH_TO_MODEL_DIR = os.path.join(WORKING_DIR, 'exported-models', 'my_model')
PATH_TO_LABELS = os.path.join(WORKING_DIR, 'annotations', 'label_map.pbtxt')
PATH_TO_SAVED_MODEL = os.path.join(PATH_TO_MODEL_DIR, 'saved_model')
#TEST_IMAGES_PATH = os.path.join(WORKING_DIR, 'images', 'test')
TEST_IMAGES_PATH = '/mnt/input'

## Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

## Load label map data
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)



def detect_image(image_path):
    """
    """
    ## Print processing message to user
    print(f'Running inference for {image_path}')

    ## Extract filename from input file
    file_name = os.path.basename(image_path)
    
    ## Open input image, convert to NumPy array, then to a tensor
    image = Image.open(image_path)
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(image_np)
    
    ## The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    ## Call the detect function on input tensor
    detections = detect_fn(input_tensor)

    ## Extract number of detections from detections
    num_detections = int(detections.pop('num_detections'))
    
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    ##
    image_np_with_detections = image_np.copy()

    ## Draw bounding-boxes around data points
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.30,
        agnostic_mode=False
    )

    ## Generate output path for plot
    output_file_path = os.path.join('/mnt/output', file_name)

    ## Save file
    plt.figure()
    plt.imshow(image_np_with_detections)
    plt.savefig(output_file_path, bbox_inches='tight')


'''
## Loop through three random images in input dir
for f in os.scandir(TEST_IMAGES_PATH):
    ## Extract input image path from DirEntry object
    image_path = f.path
    
    ## Run detect image
    detect_image(image_path)
'''


## Create a Flask application instance
app = Flask(__name__)

## Define welcome route 
@app.route('/')
def hello_route():
    """
    This route serves a welcome message to the caller. It provides nothing, and 
    should be used only to see if the API is running.
    """
    return "Welcome to CAE TensorFlow Predict API", 200

## Define predict route
@app.route('/predict',  methods=['POST'])
def predict_route():
    """
    Flask route to run the TensorFlow 
    """
    ## Get dict object containing form fields   
    form_dict = request.get_json() if request.is_json else request.form

    ## Extract values from form_dict obj
    input_dataset_name = form_dict.get('input_dataset')
    output_dataset_name = form_dict.get('output_dataset')

    ## Parameter check-point to make sure they were given and not none 
    if not all([input_dataset_name, output_dataset_name]):
        return 'Invalid parameters', 400

    ## Join dataset names to i/o paths
    input_path = os.path.join('/mnt/input', input_dataset_name)

    ## Call predict function
    detect_image(input_path)

    ## Return success message
    return 'done', 202

## Run flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7008)