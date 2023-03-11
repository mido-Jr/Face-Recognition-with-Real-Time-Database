# import Libraries
import os
import cv2
import pickle
import cvzone
import numpy as np
import face_recognition
from firebase import get_student_info, get_student_img

print("Welcome to Students Attendance Systems")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# load graphic images
imgBackGround = cv2.imread("Resources/background.png")

# loading Modes Images
FolderModesPath = "Resources/Modes"
ModesList = os.listdir(FolderModesPath)
ModesImages = []

for path in ModesList:
    img_temp = cv2.imread(os.path.join(FolderModesPath,path))
    ModesImages.append(img_temp)

# loading Encoding File
print("Start load encoding file")
file = open('EncodeFile.p', 'rb')
EncodingListKnownWithIDS = pickle.load(file)
file.close()
EncodingListKnown, StudentIDS = EncodingListKnownWithIDS
print("End load encoding file ... :) ")

# Some Param for Change APP Mode
mode_type = 0
counter = 0


# Run the APP
while True:
    success, img = cap.read()
    img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    # find face location for cur frame
    faceCurFrame = face_recognition.face_locations(img_small)
    faceCurFrameEncode = face_recognition.face_encodings(img_small, faceCurFrame)

    # Display the resulting frame
    imgBackGround[162:162+480, 55:55+640] = img
    imgBackGround[44:44 + 633, 808:808 + 414] = ModesImages[mode_type]

    # loop through the images encoded
    for encodeFace, faceLocation in zip(faceCurFrameEncode, faceCurFrame):
        matches = face_recognition.compare_faces(EncodingListKnown, encodeFace)
        faceDistance = face_recognition.face_distance(EncodingListKnown, encodeFace)

        # find the idx for the min faceDistance
        matchIdx = np.argmin(faceDistance)
        student_id = StudentIDS[matchIdx]

        if matches[matchIdx]:
            #print(f"Detect Student {student_id}")

            # Start coordinate,
            y1, x2, y2, x1 = faceCurFrame[0]
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            bbox = x1+55, y1+162, x2-x1, y2-y1

            # draw the rectangle
            imgBackGround = cvzone.cornerRect(imgBackGround, bbox, rt=0)

            # define the student
            if counter == 0:
                counter = 1
                mode_type = 1

        if counter != 0:

            if counter == 1:
                # get the images and Data from FireBase
                Student_info = get_student_info(student_id)
                student_img = get_student_img(student_id)

            cv2.putText(imgBackGround, str(Student_info['Total_attendance']), (861, 125),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            cv2.putText(imgBackGround, str(Student_info['Major']), (1006, 550),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (70, 70, 70), 1)
            cv2.putText(imgBackGround, str(student_id), (1006, 493),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (70, 70, 70), 1)
            cv2.putText(imgBackGround, str(Student_info['Grade']), (910, 625),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
            cv2.putText(imgBackGround, str(Student_info['Year']), (1025, 625),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
            cv2.putText(imgBackGround, str(Student_info['Starting_Year']), (1125, 625),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)
            (w, h), _ = cv2.getTextSize(Student_info['Name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            offset = (414-w)//2
            cv2.putText(imgBackGround, str(Student_info['Name']), (808+offset, 445),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

            imgBackGround[175:175+216, 909:909+216] = student_img

            #counter += 1

    cv2.imshow("Face Attendance", imgBackGround)
    # break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
cap.release()

# Destroy all the windows
cv2.destroyAllWindows()
