import cv2
import dlib

predictor = dlib.shape_predictor('./resources/shape_predictor_68_face_landmarks.dat')
faceCascadeClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

videoCapture = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('output/FaceRecognition.avi', fourcc, 10.0, (1280, 1024))
controlPoints = [36, 45, 39, 42, 48, 54, 30]

while videoCapture.isOpened():
    isSuccess, image = videoCapture.read()
    if isSuccess:
        image = cv2.flip(image, 1)
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascadeClassifier.detectMultiScale(imageGray, 1.9, 5)

        for (x, y, w, h) in faces:
            left = x
            top = y
            right = (x + w)
            bottom = (y + h)
            face = dlib.rectangle(left, top, right, bottom)
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

            landmarks = predictor(imageGray, face)
            for n in range(0, landmarks.num_parts):
                if n in controlPoints:
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(image, (x, y), 3, (255, 0, 0), -1)

        cv2.imshow('Done', image)
        videoWriter.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('Не удалось считать изображение с камеры')
        break

videoCapture.release()
videoWriter.release()
cv2.destroyAllWindows()
