import numpy as np
import cv2
import math
from itertools import combinations

video_stream = cv2.VideoCapture(0)
def dist1(x, y):
    dist = math.sqrt((x**2)+ (y**2))
    return (dist)

while True:
    ret, current_frame = video_stream.read()
    img_to_detect = current_frame
    img_height = img_to_detect.shape[0]
    img_width = img_to_detect.shape[1]

    img_blob = cv2.dnn.blobFromImage(img_to_detect, 0.003922, (416,416), swapRB=True, crop=False)
    class_labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                    "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
                    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball",
                    "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket",
                    "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                    "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair",
                    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                    "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
                    "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"]

    class_colors = ["0,255,0", "0,0,255", "255,0,0", "255,255,0", "0,255,255"]
    class_colors = [np.array(color.split(",").astype("int") for color in class_colors)]
    class_colors = np.array(class_colors)
    class_colors = np.tile(class_colors,(16,1))

    yolo_model = cv2.dnn.readNetFromDarknet('C:/Users/hp/Desktop/Yolo/yolov3.cfg','C:/Users/hp/Desktop/Yolo/yolov3.weights')


    yolo_layers = yolo_model.getLayerNames()
    yolo_output_layer = [yolo_layers[yolo_layer[0] - 1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]

    yolo_model.setInput(img_blob)

    objection_detection_layers = yolo_model.forward(yolo_output_layer)

    class_ids_list = []
    boxes_list = []
    confidences_list = []
    person_dict = dict()
    for objection_detection_layer in objection_detection_layers:
        for detection in objection_detection_layer:
            all_scores = detection[5:]
            predicted_class_id = np.argmax(all_scores)
            predicted_confidence = all_scores[predicted_class_id]

            if predicted_confidence > 0.20:
                bounding_box = detection[0:4]*np.array([img_width, img_height, img_width, img_height])
                (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                start_x_pt = int(box_center_x_pt - (box_width / 2))
                start_y_pt = int(box_center_y_pt - (box_height / 2))

                class_ids_list.append(predicted_class_id)
                confidences_list.append(float(predicted_confidence))
                boxes_list.append([start_x_pt , start_y_pt, int(box_width), int(box_height)])

    max_values_ids = cv2.dnn.NMSBoxes(boxes_list, confidences_list, 0.5, 0.4)

    for max_values_id in max_values_ids:
        max_class_id = max_values_id[0]
        if (class_labels[class_ids_list[max_class_id]]=="person"):
            box = boxes_list[max_class_id]
            person_dict[max_class_id] = box[0]+(box[2]/2), box[1]+(box[3]/2), box[0], box[1], box[0]+(box[2]), box[1]+(box[3])

    red_id = []
    red_box = []

    for (id1, pt1), (id2, pt2) in combinations(person_dict.items(),2):
        dist = dist1((pt1[0]-pt2[0]), (pt1[1]-pt2[1]))
        if dist < 75.0:
            if id1 not in red_id:
                red_id.append(id1)
                red_box.append([pt1[2], pt1[3], pt1[4], pt1[5]])
            if id2 not in red_id:
                red_id.append(id2)
                red_box.append([pt2[2], pt2[3], pt2[4], pt2[5]])
            cv2.line(img_to_detect,(int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0,0,255),1,cv2.LINE_AA)

    for box in red_box:
        cv2.rectangle(img_to_detect,(box[0], box[1]), (box[2], box[3]),(0,0,255),1)
    for (id, pt) in person_dict.items():
        if id not in red_id:
            cv2.rectangle(img_to_detect, (pt[2], pt[3]), (pt[4], pt[5]), (0, 255, 0), 1)
    text = "People at risk: %s"% str(len(red_box))

    cv2.putText(img_to_detect, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (1,1,255), 2, cv2.LINE_AA)

    cv2.imshow("Detection Output", img_to_detect)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

video_stream.release()
cv2.destroyAllWindows()
