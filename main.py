import cv2

faceCascadeClassifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
videoCapture = cv2.VideoCapture(0)

while True:
    success, image = videoCapture.read()
    if success:
        imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceCascadeClassifier.detectMultiScale(imageGray, 1.1, 19)
        for (x, y, h, w) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Done', image)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    else:
        print('Не удалось считать изображение с камеры')
        break

videoCapture.release()
cv2.destroyAllWindows()
