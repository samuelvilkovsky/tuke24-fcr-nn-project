import cv2
import os
import pickle

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
file = open('EncodedFile.p', 'rb')
encodedListWithIds = pickle.load(file)

while True:
    success, img = cap.read()

    imgBackground[162:162+480, 55:55+640] = img # Overlay the background image with the camera image
    imgBackground[44:44+633,808:808+414] = imgModeList[0] # Overlay the background image with the mode image

    cv2.imshow("Face Attendance", imgBackground) # Show the background image
    cv2.waitKey(1)