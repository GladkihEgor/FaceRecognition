import cv2
import dlib

predictor = dlib.shape_predictor('./resources/shape_predictor_68_face_landmarks.dat')
face_cascade_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter('output/FaceRecognition.avi', fourcc, 10.0, (1280, 1024))

FACE_RECOGNITION = 'Face recognition'
ON = 0
OFF = 1
show_rectangle = 'Show rectangle'
show_dots = 'Show_dots'
count = 'Dots count'
control_points = [30, 48, 54, 36, 45, 39, 42, 62]
faces_count = []


def nothing(*args):
    pass


def choose_dots(flag):
    if flag == 0:
        return control_points
    elif flag == 1:
        return range(17, 27)
    elif flag == 2:
        return range(36, 48)
    elif flag == 3:
        return range(27, 36)
    elif flag == 4:
        return range(48, 68)
    elif flag == 5:
        return range(0, 68)


cv2.namedWindow(FACE_RECOGNITION)
cv2.createTrackbar(count, FACE_RECOGNITION, 0, 5, nothing)
cv2.createTrackbar(show_rectangle, FACE_RECOGNITION, ON, OFF, nothing)
cv2.createTrackbar(show_dots, FACE_RECOGNITION, ON, OFF, nothing)

while True:
    is_show_rectangle = cv2.getTrackbarPos(show_rectangle, FACE_RECOGNITION)
    dots_count = cv2.getTrackbarPos(count, FACE_RECOGNITION)
    is_show_dots = cv2.getTrackbarPos(show_dots, FACE_RECOGNITION)
    is_success, image = video_capture.read()
    if is_success:
        image = cv2.flip(image, 1)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = face_cascade_classifier.detectMultiScale(image_gray, 1.9, 5)
        cv2.putText(img=image,
                    text=f'Number of recognized people: {len(faces_count)}',
                    org=(10, 50),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=1,
                    color=(255, 255, 255))
        faces_count.clear()

        for (x, y, w, h) in face:
            left = x
            top = y
            right = (x + w)
            bottom = (y + h)
            face = dlib.rectangle(left, top, right, bottom)
            faces_count.append(face)
            if is_show_rectangle == ON:
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

            landmarks = predictor(image_gray, face)
            for n in choose_dots(dots_count):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                if is_show_dots == ON:
                    cv2.circle(image, (x, y), 3, (255, 0, 0), -1)

        cv2.imshow(FACE_RECOGNITION, image)
        video_writer.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('Не удалось считать изображение с камеры')
        break

video_capture.release()
video_writer.release()
cv2.destroyAllWindows()
