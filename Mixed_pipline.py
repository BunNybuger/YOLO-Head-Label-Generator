# Import statements

import os
import cv2
import matplotlib.pyplot as plt
from retinaface import RetinaFace
from IPython.display import display
# import ipywidgets as widgets
import torch

# Constants
YOLO_CLASS_INDEX = 2
IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']
DEFAULT_BLUR_STRENGTH = 25
PERSON_MODEL_PATH = './1-models/Person_Detector.pt'
HEAD_MODEL_PATH = './1-models/Head_detector_fromPersonBox_yolov5.pt'

# Configuration
INPUT_PATH = './2-dataset/test'
OUTPUT_PATH = './4-output/blurred_test'
OUTPUT_PATH_NEW = OUTPUT_PATH + '_NewMethod'
VISUALIZE = True
BLUR = False

# Helper functions

# Set the working directory to .py file
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def de_identify_rectangle(image, rectangle, blur=False, blur_strength=DEFAULT_BLUR_STRENGTH):
    """De-identify the given rectangle region in the image."""
    x1, y1, x2, y2 = rectangle
    if blur:
        roi = image[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
        image[y1:y2, x1:x2] = blurred_roi
    else:
        image[y1:y2, x1:x2] = 0  # Set the rectangle region to black (zeros)

    return image

def is_image_file(filename):
    """Check if the given filename is an image file."""
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)

# Main functions

def process_dataset(input_path, output_path, yolo_class_index, visualize=True, blur=False):
    """Process the dataset by detecting faces and saving yolo.txt files and blurred images."""
    os.makedirs(output_path, exist_ok=True)
    retina_failed = []

    for root, _, files in os.walk(input_path):
        for filename in files:
            if is_image_file(filename):
                image_path = os.path.join(root, filename)
                resp = RetinaFace.detect_faces(image_path)

                if isinstance(resp, dict):
                    rel_image_path = os.path.relpath(image_path, input_path)
                    txt_output_path = os.path.join(output_path, os.path.splitext(rel_image_path)[0] + '.txt')
                    os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)

                    image = cv2.imread(image_path)
                    img_height, img_width, _ = image.shape

                    with open(txt_output_path, 'w') as f:
                        for key in resp:
                            facial_area = resp[key]['facial_area']
                            x_center = (facial_area[0] + facial_area[2]) / (2 * img_width)
                            y_center = (facial_area[1] + facial_area[3]) / (2 * img_height)
                            width = (facial_area[2] - facial_area[0]) / img_width
                            height = (facial_area[3] - facial_area[1]) / img_height
                            f.write(f"{yolo_class_index} {x_center} {y_center} {width} {height}\n")

                    if visualize:
                        output_image_path = os.path.join(output_path, rel_image_path)
                        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

                        for key in resp:
                            facial_area = resp[key]['facial_area']
                            image = de_identify_rectangle(image, facial_area, blur=blur)

                        cv2.imwrite(output_image_path, image)
                        display_images(image_path, image)

                else:
                    retina_failed.append(image_path)

    return retina_failed

def load_model(model_path, model_name='custom'):
    """Load the YOLO model."""
    return torch.hub.load('ultralytics/yolov5', model_name, path=model_path)

def process_imagelist_yolo(images, output_path, person_model, head_model, yolo_class_index, visualize=True, blur=False):
    """Process a list of images using YOLO models to detect persons and heads, save yolo.txt files, and create blurred images."""
    os.makedirs(output_path, exist_ok=True)

    for image_path in images:
        image = cv2.imread(image_path)
        img_height, img_width, _ = image.shape

        # Step 1: Detect persons using the first model
        results_person = person_model(image)
        persons = results_person.pandas().xyxy[0]
        person_boxes = persons[persons['name'] == 'person'][['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()

        # Step 2: Detect heads using the second model
        head_boxes = []
        head_yolo = []
        for box in person_boxes:
            [x1, y1, x2, y2] = [int(x) for x in box]
            person_region = image[y1:y2, x1:x2]
            results_head = head_model(person_region)
            heads = results_head.pandas().xyxy[0]
            if 'head' in heads['name'].values:
                head = heads[heads['name'] == 'head'][['xmin', 'ymin', 'xmax', 'ymax']].values[0]
                head_boxes.append([x1 + head[0], y1 + head[1], x1 + head[2], y1 + head[3]])
                x_center = (x1 + head[0] + x1 + head[2]) / (2 * img_width)
                y_center = (y1 + head[1] + y1 + head[3]) / (2 * img_height)
                width = (head[2] - head[0]) / img_width
                height = (head[3] - head[1]) / img_height
                head_yolo.append([x_center, y_center, width, height])

        # Step 3: Save YOLO format .txt file with head bounding boxes
        rel_image_path = os.path.relpath(image_path, INPUT_PATH)
        txt_output_path = os.path.join(output_path, os.path.splitext(rel_image_path)[0] + '.txt')
        os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)

        with open(txt_output_path, 'w') as f:
            for box in head_yolo:
                f.write(f"{yolo_class_index} {box[0]} {box[1]} {box[2]} {box[3]}\n")

        # Step 4: Save the new dataset with blurred faces
        if visualize:
            output_image_path = os.path.join(output_path, rel_image_path)
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

            for box in head_boxes:
                box = [int(x) for x in box]
                image = de_identify_rectangle(image, box, blur=blur)

            cv2.imwrite(output_image_path, image)
            display_images(image_path, image)

def display_images(original_image_path, processed_image):
    """Display the original and processed images side by side."""
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(original_image_path), cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.title('Processed Image')
    plt.axis('off')
    display(plt.gcf())

if __name__ == "__main__":
    # Process the dataset using the first method
    retina_failed = process_dataset(INPUT_PATH, OUTPUT_PATH, YOLO_CLASS_INDEX, VISUALIZE, BLUR)

    if len(retina_failed) != 0:
        print(f"First method finished, yolo txt files are in: {OUTPUT_PATH} \n")
        print(f"Retina Could not detect faces in {len(retina_failed)} Images.\n")
        print(f"***************\n\n\nRunning second method for {len(retina_failed)} Images failed in Retina\n\n\n***************")
        # Load YOLO models
        person_model = load_model(PERSON_MODEL_PATH)
        head_model = load_model(HEAD_MODEL_PATH)

        # Process the images using the second method
        process_imagelist_yolo(retina_failed, OUTPUT_PATH_NEW, person_model, head_model, YOLO_CLASS_INDEX, VISUALIZE, BLUR)
        print(f"***************\nyolo.txt files and blurred images using the 2nd method are saved in {OUTPUT_PATH_NEW}\n***************")

    else:
        print(f"First method processed all images! done! \nyolo txt files are in: {OUTPUT_PATH}")