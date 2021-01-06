# import library
import cv2

# pre trained algortithm
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img = cv2.imread('ningning.jpeg')
webcam = cv2.VideoCapture(0)

while True : 
    succesful_frame_read, frame = webcam.read()
    # grayscaled the image
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # coordinates
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    # draw rectangle
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 10)
    # show webcam
    cv2.imshow('Face Tracker', frame)
    cv2.waitKey(1)

print ("Process Completed")