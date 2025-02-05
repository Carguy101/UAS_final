import cv2 as cv
from numpy.ma.core import append
from ultralytics import YOLO
import numpy as np
model=YOLO("C:\\Users\\Admin\\Downloads\\best(1).pt")



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


def assign_(plants,fruits):
    #assign_dict_={m:{0:0,1:0,2:0} for m in range(len(plants))}
    transformed_fruits={m: [] for m in range(len(plants))}
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
            px1, py1, px2, py2 = plants[best_plant]
            width_best_plant=abs(px2-px1)
            height_best_plant=abs(py2-py1)
            center_plant = ((px1 + px2) / 2, (py1 + py2) / 2)

            transformed_x = (centre_fruit[0] - center_plant[0])/width_best_plant
            transformed_y = (centre_fruit[1] - center_plant[1])/height_best_plant

            transformed_fruits[best_plant].append([transformed_x, transformed_y, cls])
            #assign_dict_[best_plant][cls]+=1

    return transformed_fruits



f_1,p_1=detection_("imageuptodown_img_4_1.jpg") #front view
rear=cv.imread("imageuptodown_img_4_5.jpg")
flip=cv.flip(rear,1)
cv.imwrite("mirror image1.jpg",flip)
f_2,p_2=detection_("mirror image1.jpg")
d1=assign_(p_1,f_1)
d2=assign_(p_2,f_2)


def find_unique_fruits(d1, d2, threshold=0.1):
    """Finds unique fruits by comparing transformed coordinates in front and back views."""
    unique_fruits = {m: [] for m in d1.keys()}  # Store unique fruits for each plant


    for plant_id in d1:
        front_fruits = d1[plant_id]  # Fruits in front view (transformed)
        back_fruits = d2.get(plant_id, [])  # Fruits in back view (transformed)

        matched_indices = set()  # Keep track of matched indices in back_fruits

        # Step 1: Process front view fruits
        for f1 in front_fruits:
            x1, y1, cls1 = f1
            is_duplicate = False

            for i, f2 in enumerate(back_fruits):
                x2, y2, cls2 = f2

                if cls1 == cls2:  # Same fruit type
                    distance = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))

                    if distance < threshold:  # If fruit positions are close, consider duplicate
                        is_duplicate = True
                        matched_indices.add(i)  # Mark this back fruit as counted
                        break

            # If not duplicate, add it to the unique list
            unique_fruits[plant_id].append(f1)

        # Step 2: Process back view fruits that were NOT matched
        for i, f2 in enumerate(back_fruits):
            if i not in matched_indices:  # Only add unmatched fruits from back view
                unique_fruits[plant_id].append(f2)
    unique_fruits_count = {m: len(unique_fruits[m]) for m in d1.keys()}

    return unique_fruits,unique_fruits_count

# Find unique fruits
d3,d4= find_unique_fruits(d1, d2)

print("Unique Fruits per Plant:", d4)































































































