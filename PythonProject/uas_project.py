import cv2 as cv
from numpy.ma.core import append
from ultralytics import YOLO
import numpy as np
model=YOLO("C:\\Users\\Admin\\Downloads\\best(1).pt")

def iou(box1,box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Compute intersection
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Compute union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def detection_(images):
    image=cv.imread(images)
    results=model(image)
    plants=[]
    fruits=[]
    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2=map(int,box.xyxy[0])
            cls=int(box.cls[0])


            if cls==3:
                plants.append((x1,y1,x2,y2))
            else:
                fruits.append((x1,y1,x2,y2,cls))

    return fruits,plants

#the above code will detect all fruits and plants,now , we need to give each plant its fruit.

def remove_duplicates(fruits_front,fruits_back,iou_threshold=0.5):
    unique_fruits=list(fruits_front)
    for back_fruit in fruits_back:
        x1_b, y1_b, x2_b, y2_b, cls_b = back_fruit
        is_duplicate=False
        for front_fruit in fruits_front:
            x1_f, y1_f, x2_f, y2_f, cls_f = front_fruit
            if cls_b==cls_f and iou((x1_b, y1_b, x2_b, y2_b),(x1_f, y1_f, x2_f, y2_f))>iou_threshold:
                is_duplicate=True
                break
        if not is_duplicate:
            unique_fruits.append(back_fruit)

    return unique_fruits

def assign_(plants,fruits):
    assign_dict_={m:{0:0,1:0,2:0} for m in range(len(plants))}
    #initialise count of each type of fruit in each plant to 0
    for fruit in fruits:
        x1,y1,x2,y2,cls=fruit
        centre_fruit=((x1+x2)/2,(y1+y2)/2)

        best_plant=None
        min_distance=float("inf")

        for m,(px1,py1,px2,py2) in enumerate(plants):
            centre_plant=((px1+px2)/2,(py1+py2)/2)
            distance = np.linalg.norm(np.array(centre_fruit) - np.array(centre_plant))

            if distance<min_distance:
                min_distance=distance
                best_plant=m

        if best_plant is not None:
            assign_dict_[best_plant][cls] += 1


    return assign_dict_


f_1,p_1=detection_("imageuptodown_img_7_4.jpg") #front view
rear=cv.imread("imageuptodown_img_7_7.jpg")
flip=cv.flip(rear,1)
cv.imwrite("mirror_image.jpg",flip)
f_2,p_2=detection_("mirror_image.jpg") #back view
unique_fruit=remove_duplicates(f_1,f_2)
final_assignment=assign_(p_1,unique_fruit)
print("Final fruit assignment to plants:", final_assignment)





















































