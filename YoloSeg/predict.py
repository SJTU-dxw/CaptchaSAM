import glob
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

test_filenames = glob.glob("Microsoft_test/*.jpg")
model = YOLO('Microsoft.pt')

is_Amazon = False
is_Microsoft = True

show_flag = True
correct_pred = 0
for filename in test_filenames:
    result = model(filename, verbose=False)[0]
    pred_boxes = result.boxes
    characters = []
    characters_boxes = []
    confs = []
    for d in reversed(pred_boxes):
        c = int(d.cls)
        name = result.names[c]
        box = d.xyxy.squeeze()

        characters.append(name)
        characters_boxes.append(box)
        confs.append(d.conf.item())

    if is_Microsoft:
        # for Microsoft
        boxes_middle = [(boxes[1] + boxes[3]).item() / 2. for boxes in characters_boxes]
        kmeans = KMeans(n_clusters=2, n_init="auto")
        pred = kmeans.fit_predict(np.array(boxes_middle).reshape(-1, 1))

        centers = kmeans.cluster_centers_.reshape(-1)
        horizontal_position = np.argsort(np.argsort(centers))
        horizontal_position = horizontal_position[pred]

        boxes_position = np.array([(boxes[0] + boxes[2]).item() / 2. for boxes in characters_boxes])
        sorted_positions_layer = np.argsort(np.argsort(boxes_position))

        sorted_positions_layer = sorted_positions_layer + horizontal_position * 10
    else:
        # normal captcha
        boxes_position = np.array([(boxes[0] + boxes[2]).item() / 2. for boxes in characters_boxes])
        sorted_positions_layer = np.argsort(np.argsort(boxes_position))

    sorted_positions_layer = np.argsort(sorted_positions_layer)

    pred_characters = []
    for position in sorted_positions_layer:
        if confs[position] > 0.5:
            pred_characters.append(characters[position])
    pred_characters = "".join(pred_characters).lower()

    # for Amazon
    if is_Amazon:
        pred_characters = pred_characters[-6:]

    correct_characters = (filename.split("/")[-1].split(".")[0].replace("-", "")).lower()
    if pred_characters == correct_characters:
        correct_pred += 1
    else:
        img = cv2.imread(filename)
        plt.imshow(img)
        plt.title("pred: " + pred_characters + " correct:" + correct_characters)
        plt.show()

        res_plotted = result.plot()
        plt.imshow(res_plotted)
        plt.show()

        print(filename, "pred: ", pred_characters, "correct:", correct_characters)

    if show_flag:
        res_plotted = result.plot()
        plt.imshow(res_plotted)
        plt.show()

        show_flag = False

print("Acc:", correct_pred, "/", len(test_filenames), "=", correct_pred / len(test_filenames))
