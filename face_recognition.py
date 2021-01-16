import cv2 as cv
import dlib

from methods import FaceRecognitionMethod


class FaceRecognition:
    _control_points = [30, 48, 54, 36, 45, 39, 42, 62]
    _fourcc = cv.VideoWriter_fourcc(*'XVID')
    _video_writer = cv.VideoWriter('output/FaceRecognition.avi', _fourcc, 10.0, (1280, 1024))
    _video_capture = cv.VideoCapture(0)
    _detector = dlib.get_frontal_face_detector()
    _predictor = dlib.shape_predictor('./resources/shape_predictor_68_face_landmarks.dat')
    _face_cascade_classifier = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def __init__(self, view):
        self.view = view

    def run(self, method=FaceRecognitionMethod.HAARCASCADE):
        self.view.setup()
        people_count = []
        while self._video_capture.isOpened():
            (is_show_rectangle, dots_count, is_show_landmarks) = self.view.create_trackbars()
            is_success, image = self._video_capture.read()

            if is_success:
                image = cv.flip(image, 1)
                image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

                if method == FaceRecognitionMethod.HAARCASCADE:
                    face = self._use_haarcascade(image_gray)
                elif method == FaceRecognitionMethod.DETECTOR:
                    face = self._use_dlib_detector(image_gray)
                else:
                    face = dlib.rectangle
                    print('No such method')
                self.view.add_people_count_text(len(people_count), image)
                people_count.clear()

                for points in face:
                    people_count.append(1)
                    if is_show_rectangle == self.view.on:
                        self.view.draw_face_rectangle(image, points)

                    if is_show_landmarks == self.view.on:
                        landmarks = self._predict_landmarks(image_gray, points)
                        count = self._choose_dots(dots_count)
                        self.view.draw_landmarks(landmarks, count, image)

                self.view.show_image(image)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                print('Не удалось считать изображение с камеры')
                break

        self._video_capture.release()
        self._video_writer.release()
        self.view.close()

    def _choose_dots(self, flag):
        if flag == 0:
            return self._control_points
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

    def _use_haarcascade(self, gray_image):
        face = self._face_cascade_classifier.detectMultiScale(gray_image, 1.9, 5)
        for (x, y, w, h) in face:
            left = x
            top = y
            right = (x + w)
            bottom = (y + h)
            face = dlib.rectangle(left, top, right, bottom)
        return face

    def _use_dlib_detector(self, gray_image, ):
        return self._detector(gray_image, 0)

    def _predict_landmarks(self, gray_image, face):
        return self._predictor(gray_image, face)
