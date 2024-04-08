import cv2
import numpy as np
import os
import pickle
import face_recognition
import cvzone

# 1. PART - Set camera settings and start capturing video
cap = cv2.VideoCapture(0);
cap.set(3, 640);
cap.set(4, 480);

# 2. PART - Read the background image
imgBackground = cv2.imread("Resources/background.png")

# 3. PART - Read the mode images into a list
folderModePath = 'Resources/Modes'
modePath = os.listdir(folderModePath)
imgModeList = []

for path in modePath:
    imgModeList.append(cv2.imread(f'{folderModePath}/{path}'))
# print(len(imgModeList))
    
# 4. PART - Load the encoded file
print('Loading Encoded File...')
file = open('EncodedFile.p', 'rb')
encodedListWithIds = pickle.load(file)
file.close()
encodedImages, studentIds = encodedListWithIds
print('File Loaded')
# print(studentIds)

while True:
    success, img = cap.read()

    # make image smaller using scale
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB) # Convert the image to RGB format, because face_recognition library uses RGB format


    faceCurrentFrame = face_recognition.face_locations(imgS) # Find the face locations in the image
    encodeCurrentFrame = face_recognition.face_encodings(imgS, faceCurrentFrame) # Encode the face locations in the image

    imgBackground[162:162+480, 55:55+640] = img # Overlay the background image with the camera image
    imgBackground[44:44+633,808:808+414] = imgModeList[0] # Overlay the background image with the mode image

    for encodeFace, faceLoc in zip(encodeCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(encodedImages, encodeFace) # Compare the faces
        faceDistance = face_recognition.face_distance(encodedImages, encodeFace) # Calculate the face distance lower is better
        # * TESTING
        # print("matches ", matches)
        # print("faceDistance ", faceDistance)

        matchIndex = np.argmin(faceDistance) # Get the index of the minimum face distance
        # print("matchIndex ", matchIndex)

        if matches[matchIndex]:
            print("Known face detected. Student ID: ", studentIds[matchIndex])
            # show the rectangle around the face
            y1, x2, y2, x1 = faceLoc # Get the face locations
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)

    cv2.imshow("Face Attendance", imgBackground) # Show the background image
    cv2.waitKey(1)