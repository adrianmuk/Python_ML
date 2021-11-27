import cv2
from random import randrange

# video
video_file = cv2.VideoCapture(0)

# pre-trained car and pedestrian classifiers
car_file = 'cars.xml'
pedestrian_file = 'haarcascade_fullbody.xml'

# create car and pedestrian classifiers
car_tracker = cv2.CascadeClassifier(car_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_file)

while True:
    # Read current frame
    successful_frame_read, frame = video_file.read()
    # convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect cars and pedestrians
    car_coordinates = car_tracker.detectMultiScale(gray_frame)
    ped_coordinates = pedestrian_tracker.detectMultiScale(gray_frame)
    # Draw rectangles around cars
    for (x, y, w, h) in car_coordinates:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Draw rectangles around pedestrians
    for (x, y, w, h) in ped_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow('Self Driving Car', frame)
    key = cv2.waitKey(1)

    # Stop if Q is pressed
    if key == 81 or key == 113:
        break

# Release the video capture object
video_file.release()
"""
# image
img_file = 'images (1).jpeg'

# pre-trained car classifier
classifier_file = 'cars.xml'

# create opencv image
img = cv2.imread(img_file)

# Convert image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# Detect car
car_coordinates = car_tracker.detectMultiScale(gray_img)
print(car_coordinates)
# Draw rectangle around car
for (x, y, w, h) in car_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

# Display the image with the faces spotted
cv2.imshow('Clever Programmer Car Detector', img)

# Don't auto-close
cv2.waitKey()
"""
print("Code completed")
