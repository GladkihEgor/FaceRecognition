import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./resources/shape_predictor_68_face_landmarks.dat')

videoCapture = cv2.VideoCapture(0)

while True:
    success, image = videoCapture.read()
    if success:
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = detector(imageGray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            landmarks = predictor(imageGray, face)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(image, (x, y), 3, (255, 0, 0), -1)

        cv2.imshow('Done', image)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    else:
        print('Не удалось считать изображение с камеры')
        break

videoCapture.release()
cv2.destroyAllWindows()
