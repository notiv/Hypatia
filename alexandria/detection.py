import os
import pandas as pd
import numpy as np
import cv2
import glob

from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches

_WHITE = (255, 255, 255)
img = None
img0 = None
outputs = None

_DATA_FOLDER = "data"
_IMG_FOLDER = "static/upload/*/"

# Load names of classes and get random colors
def load_classes(filename="coco.names"):
    with open(os.path.join(_DATA_FOLDER, filename)) as infile:
        classes = infile.read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
    return classes, colors


def load_model(cfg='yolov3.cfg', model='yolov3.weights'):
    cfg_file = os.path.join(_DATA_FOLDER, cfg)
    model_file = os.path.join(_DATA_FOLDER, model)
    net = cv2.dnn.readNetFromDarknet(cfg_file, model_file)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    return net


def get_number_layers(net):
    ln = net.getLayerNames()
    return [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# def load_images(glob_string="*.jp*"):
#     img_paths = glob.glob(_IMG_FOLDER+"/"+glob_string)
#     return img_paths, list(map(lambda x: cv2.imread(x), img_paths))

def load_images(glob_string="*.jp*"):
    img_paths = [f for f in glob.glob(_IMG_FOLDER+"/"+glob_string) if 'thumbnail' not in f]
    return img_paths, list(map(lambda x: cv2.imread(x), img_paths))


def detect(img, net):
    blob = cv2.dnn.blobFromImage(
        img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    ln = get_number_layers(net)
    # combine the 3 output groups into 1 (10647, 85)
    # large objects (507, 85)
    # medium objects (2028, 85)
    # small objects (8112, 85)
    return np.vstack(net.forward(ln))


def get_boxes(img, outputs, conf, classes_names, colors):
    H, W = img.shape[:2]
    boxes = []
    confidences = []
    classIDs = []
    positions = []
    for output in outputs:
        scores = output[5:]
        classID = np.argmax(scores)
        if classes_names[classID] != "book":
            continue
        confidence = scores[classID]
        if confidence > conf:
            x, y, w, h = output[:4] * np.array([W, H, W, H])
            p0 = int(x - w//2), int(y - h//2)
            boxes.append([*p0, int(w), int(h)])
            confidences.append(float(confidence))
            classIDs.append(classID)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
    Position = namedtuple(
        'Position',
        ['x_slice', 'y_slice', 'args4rectangle', 'args4rectangle_cv'])
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            pos = Position(
                x_slice=(x, x+w),
                y_slice=(y, y+h),
                args4rectangle=((x, y), w, h, color),
                args4rectangle_cv=((x, y), (x+w, y+h))
            )
            positions.append(pos)
    return positions


def show_img_rectangles(img, box_position):
    # Create figure and axes
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(img)
    for boxes in box_position:
        xy, width, height, color = boxes.args4rectangle
        # Create a Rectangle patch
        ax.add_patch(
            patches.Rectangle(xy, width, height, edgecolor="red", fill=False))
    # Add the patch to the Axes
    plt.show()


def save_img_rectangles(img, box_position, outfile):
    # Create figure and axes
    fig, ax = plt.subplots()
    # Display the image
    ax.imshow(img)
    for boxes in box_position:
        xy, width, height, color = boxes.args4rectangle
        # Create a Rectangle patch
        ax.add_patch(
            patches.Rectangle(xy, width, height, edgecolor="red", fill=False))
    # Add the patch to the Axes
    plt.savefig(outfile)

def save_img_rectangles_cv(img, box_position, outfile):
    for boxes in box_position:
        xy, width, height, _ = boxes.args4rectangle
        img = cv2.rectangle(img, xy, tuple(sum(x) for x in zip(xy, (width, height))),
            color=(0, 255, 0), thickness=2)

    cv2.imwrite(outfile, img)



if __name__ == "__main__":
    # user input
    confidence = 0.6
    classes, colors = load_classes("coco.names")
    model = load_model(cfg='yolov3.cfg', model='yolov3.weights')
    images_paths, images_list = load_images()
    outputs_list = [detect(i, model) for i in images_list]
    boxes_positions = [get_boxes(i, o, confidence, classes, colors)
                       for i, o in zip(images_list, outputs_list)]
    for i, b in zip(images_list, boxes_positions):
        show_img_rectangles(i, b)
