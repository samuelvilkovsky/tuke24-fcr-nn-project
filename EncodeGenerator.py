import cv2
import face_recognition
import pickle
import os

# Importing images
folderPath = 'Images'
pathList = os.listdir(folderPath)
imgList = []
studentIds = []

for path in pathList:
    # print(path)
    imgList.append(cv2.imread(f'{folderPath}/{path}'))
    studentIds.append(path.split('.')[0])
studentIds.pop(0) # Samuel bug it is only working with this
imgList.pop(0) # Samuel bug it is only working with this
# print(studentIds)

# Encoding images
def encodeImages(images):
    encodedList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the image to RGB format, because face_recognition library uses RGB format
        encode = face_recognition.face_encodings(img)[0] # Encode the image
        encodedList.append(encode)
    
    return encodedList
print('Encoding Started...')
encodedImages = encodeImages(imgList)
# print(encodedImages)
encodedImagesWithIds = [encodedImages, studentIds]
print('Encoding Complete')

file = open('EncodedFile.p', 'wb')
pickle.dump(encodedImagesWithIds, file)
file.close()
print('File Saved')
