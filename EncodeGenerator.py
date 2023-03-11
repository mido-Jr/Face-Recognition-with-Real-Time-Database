import os
import cv2
import pickle
import face_recognition

# load the Student IDS
FolderImagesPath = "Images"
ImagesList = os.listdir(FolderImagesPath)
StudentImages = []
StudentIDS = []

for path in ImagesList:
    # get the image
    img_temp = cv2.imread(os.path.join(FolderImagesPath, path))
    StudentImages.append(img_temp)

    # get the ID
    id_temp = os.path.splitext(path)[0]
    StudentIDS.append(id_temp)


# create Encoding Feature
def create_encoding(imageslist):

    encodinglist = []
    for img in imageslist:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodinglist.append(encode)

    return encodinglist


print("Start Encoding ...")
EncodingListKnown = create_encoding(StudentImages)
print("End Encoding Process ... :) ")

# add IDs for encoding images
EncodingListKnownWithIDS = [EncodingListKnown, StudentIDS]

# create pickle file
file = open("EncodeFile.p", 'wb')
pickle.dump(EncodingListKnownWithIDS, file)
file.close()
print("End pickles Process ...")
print("File Saved ... :) ")

