import numpy as np
import os
import tensorflow.compat.v1 as tf
import cv2

# Disable TensorFlow 2.x behavior
tf.disable_v2_behavior()

# Paths
PATH_TO_CKPT = os.path.join('pknic_trained_model', 'exported_model', 'frozen_inference_graph.pb')
NUM_CLASSES = 1

# Load TensorFlow model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Manually define label names instead of label_map_util
category_index = {1: {"name": "ID Card"}}

def draw_bounding_boxes(image, boxes, classes, scores, threshold=0.5):
    """Draw bounding boxes on the image based on detection results."""
    h, w, _ = image.shape
    for i in range(len(boxes)):
        if scores[i] > threshold:
            y_min, x_min, y_max, x_max = boxes[i]
            x_min, x_max = int(x_min * w), int(x_max * w)
            y_min, y_max = int(y_min * h), int(y_max * h)
            
            # Draw rectangle and label
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"{category_index[int(classes[i])]['name']} ({scores[i]:.2f})"
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def detect_id_card(image_path):
    """Detects an ID card in the given image and draws bounding boxes."""
    
    image_np = cv2.imread(image_path)
    if image_np is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    image_np_expanded = np.expand_dims(image_np, axis=0)

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            # Model input/output tensors
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Run detection
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # Draw bounding boxes on the image
            image_with_boxes = draw_bounding_boxes(image_np, np.squeeze(boxes), np.squeeze(classes), np.squeeze(scores))

            # Save and show result
            output_path = "output.jpg"
            cv2.imwrite(output_path, image_with_boxes)
            print(f"Detection complete. Output saved to {output_path}")
            cv2.imshow("ID Card Detector", image_with_boxes)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Run detection
image_path = r"C:\Users\mahaz.abbasi\Downloads\download (1).jpg"  # Change to your image path
detect_id_card(image_path)
