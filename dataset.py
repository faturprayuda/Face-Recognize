import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(4, 480) # set tinggi video
# Deteksi Wajah
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Setiap orang, memasukkan 1 id wajah
face_id = input('\n Masukkan ID Wajah, kemudian tekan Enter : ')
print("\n [INFO] Mulai menangkap wajah. Lihatlah kamera dan tunggu...")

# Inisialisasi contoh wajah per individu
count = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        # Menyimpan photo wajah pada folder TRAINER
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Tekan 'ESC' untuk keluar dari Video
    if k == 27:
        break
    elif count >= 30: # Mengambil foto wajah sebanyak 30 dan menghentikan Video
         break

# Membersihkan Memory
print("\n [INFO] Keluar Progam")
cam.release()
cv2.destroyAllWindows()


