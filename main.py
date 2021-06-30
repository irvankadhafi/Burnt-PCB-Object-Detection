import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# loading image
cap = cv2.VideoCapture('/home/irvan/test.mp4')  # or cap = cv2.VideoCapture("<video-path>")
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0


def load_model():
    # Lokasi Labelmap dan Frozen saved model
    PATH_TO_LABELS = os.path.join('model', 'labelmap.pbtxt')
    PATH_TO_SAVED_MODEL = os.path.join('model', 'saved_model')

    global category_index
    global detection_model

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    detection_model = tf.saved_model.load(str(PATH_TO_SAVED_MODEL))
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.8,
                                           tf.uint8)
        print(detection_masks_reframed)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def run_inference(model, cap):
    while cap.isOpened():
        global frame_id
        frame_id += 1
        ret, image_np = cap.read()
        # Actual detection.
        output_dict = run_inference_for_single_image(model, image_np)
        # print(output_dict['detection_classes'],'\n')
        # print(output_dict['detection_scores'].size,'\n')
        # if scores.size != 0:
        #   new_scores =  np.extract(output_dict['detection_scores']>0.1,output_dict['detection_scores'])
        #   output_dict['detection_scores'] = new_scores
        # Visualization of the results of a detection.
        if True:
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=8)

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(image_np, "FPS:" + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 1)
        cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    load_model()
    run_inference(detection_model, cap)
