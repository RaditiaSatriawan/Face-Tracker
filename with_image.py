# import library
import cv2

# pre trained algortithm
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# import image
img = cv2.imread('ningning.jpeg')

# grayscaled the image
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# coordinates
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
# draw rectangle
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 5)
# show result
cv2.imshow('Face Tracker', img)
cv2.waitKey()

print ("Process Completed")