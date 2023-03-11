import os
import cv2
import firebase_admin
import numpy as np
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://face-recognition-attenda-6313f-default-rtdb.firebaseio.com/",
    "storageBucket": "face-recognition-attenda-6313f.appspot.com"
})

ref = db.reference("Students")

# write your own data for each student

data = {
    "498205": {
        "Name": "Mido",
        "Major": "Computer Eng",
        "Starting_Year": 2017,
        "Total_attendance": 7,
        "Grade": "A",
        "Year": 4,
        "Last_attendance_time": "2022-6-18 00:12:53"

    },
    "852741": {
        "Name": "ShoSho",
        "Major": "Computer Eng",
        "Starting_Year": 2021,
        "Total_attendance": 9,
        "Grade": "A+",
        "Year": 3,
        "Last_attendance_time": "2023-2-21 00:11:04"

    },
    "963852": {
        "Name": "Mohamed",
        "Major": "Computer Eng",
        "Starting_Year": 2023,
        "Total_attendance": 2,
        "Grade": "B",
        "Year": 1,
        "Last_attendance_time": "2023-3-10 00:02:58"

    },
    "123456": {
            "Name": "AMR",
            "Major": "Accountant",
            "Starting_Year": 2023,
            "Total_attendance": 32,
            "Grade": "C",
            "Year": 3,
            "Last_attendance_time": "2021-4-8 00:07:59"

    },
    "456123": {
                "Name": "Ozil",
                "Major": "Player",
                "Starting_Year": 2023,
                "Total_attendance": 32,
                "Grade": "C",
                "Year": 3,
                "Last_attendance_time": "2021-4-8 00:07:59"

    },
    "951753": {
                "Name": "Emy",
                "Major": "Quran Teacher",
                "Starting_Year": 2023,
                "Total_attendance": 32,
                "Grade": "C",
                "Year": 3,
                "Last_attendance_time": "2021-4-8 00:07:59"

            }
}


def send_data():
    # send the data to realTime Database
    print("Start Loading to RealTime DataBase ...!!")
    for key, value in data.items():
        ref.child(key).set(value)
    print("Loading to RealTime DataBase Done :)")


def upload_images():
    # Upload Images to Firebase Storage
    print("Start upload_ Images  to Storage FireBase ...!!")
    images_path = "Images"
    images_list = os.listdir(images_path)
    for path in images_list:
        file_name = f'{images_path}/{path}'
        bucket = storage.bucket()
        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_name)
    print("upload Images Done :) ")


# Want to Update the Date ?
# send_data()
# upload_images()


# get Information from firebase
def get_student_info(student_id):
    info = db.reference(f"Students/{student_id}").get()
    return info


# get the Images from Storage FireBase
def get_student_img(student_id):
    bucket = storage.bucket()
    blob = bucket.get_blob(f'Images/{student_id}.jpg')
    array = np.frombuffer(blob.download_as_string(), np.uint8)
    img_student = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
    return img_student

