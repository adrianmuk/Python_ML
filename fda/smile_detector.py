import cv2

# load pre-trained data on face fronts from opencv
trained_face_file = 'haarcascade_frontalface_default.xml'
trained_smile_file = 'haarcascade_smile.xml'
trained_eye_file = 'haarcascade_eye.xml'
# capture video from webcam
webcam = cv2.VideoCapture(0)

# smile and face classifier
face_detector = cv2.CascadeClassifier(trained_face_file)
smile_detector = cv2. CascadeClassifier(trained_smile_file)
eye_detector = cv2.CascadeClassifier(trained_eye_file)

# Iterate forever over frames
while True:
    # Read current frame
    successful_frame_read, frame = webcam.read()
    if not successful_frame_read:
        break
    # convert to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    face_coordinates = face_detector.detectMultiScale(grayscale_frame)
    print(face_coordinates)
    # Draw rectangles around faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
        # Get sub-frame using numpy N-dimensional array slicing
        the_face = frame[y:y+h, x:x+w]
        # Change to grayscale
        grayscale_face = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)
        eye_coordinates = eye_detector.detectMultiScale(grayscale_face, scaleFactor=1.1, minNeighbors=20)
        smile_coordinates = smile_detector.detectMultiScale(grayscale_face, scaleFactor=1.7, minNeighbors=20)
        # Find and label smile
        if len(smile_coordinates) > 0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale=2, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0, 255, 255))
        for (a, b, c, d) in eye_coordinates:
            cv2.rectangle(the_face, (a, b), (a + c, b + d), (255, 255, 255), 2)

    cv2.imshow('Smile Detector', frame)
    key = cv2.waitKey(1)

    # Stop if Q is pressed
    if key == 81 or key == 113:
        break

# Release the video capture object
webcam.release()
cv2.destroyAllWindows()
print('Code completed')

"""
# choose an image to detect faces in
img = cv2.imread('avg.jpg')

# converting to grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

# Draw rectangles around faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

print(face_coordinates)


#
cv2.imshow('Clever Programmer Face Detector', img)
cv2.waitKey()

print('Code completed')
"""