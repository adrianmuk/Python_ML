import cv2
from random import randrange
# load pre-trained data on face fronts from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# capture video from webcam
webcam = cv2.VideoCapture(0)

# Iterate forever over frames
while True:
    # Read current frame
    successful_frame_read, frame = webcam.read()
    # convert to grayscale
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)
    # Draw rectangles around faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(grayscaled_frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)

    cv2.imshow('Clever Programmer Face Detector', grayscaled_frame)
    key = cv2.waitKey(1)

    # Stop if Q is pressed
    if key == 81 or key == 113:
        break

# Release the video capture object
webcam.release()
print('Code completed')

"""
# choose an image to detect faces in
img = cv2.imread('avg.jpg')

# converting to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw rectangles around faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

print(face_coordinates)


#
cv2.imshow('Clever Programmer Face Detector', img)
cv2.waitKey()

print('Code completed')
"""