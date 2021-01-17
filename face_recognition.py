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
        self._image = None
        self._image_gray = None

    def run(self, method=FaceRecognitionMethod.HAARCASCADE):
        self.view.setup()
        people_count = []
        while self._video_capture.isOpened():
            (is_show_rectangle, dots_count, is_show_landmarks) = self.view.create_trackbars()
            is_success, self._image = self._video_capture.read()

            if is_success:
                self._image = cv.flip(self._image, 1)
                self.view.set_image(self._image)
                self._image_gray = cv.cvtColor(self._image, cv.COLOR_BGR2GRAY)

                if method == FaceRecognitionMethod.HAARCASCADE:
                    frame = self._use_haarcascade()
                elif method == FaceRecognitionMethod.DETECTOR:
                    frame = self._use_dlib_detector()
                else:
                    print('No such method')
                    break
                self.view.add_people_count_text(len(people_count))
                people_count.clear()

                for face in frame:
                    people_count.append(1)
                    if is_show_rectangle == self.view.on:
                        self.view.draw_face_rectangle(face)

                    if is_show_landmarks == self.view.on:
                        landmarks = self._predict_landmarks(face)
                        count = self._choose_dots(dots_count)
                        self.view.draw_landmarks(landmarks, count)

                self.view.show_image()
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

    def _use_haarcascade(self):
        frame = self._face_cascade_classifier.detectMultiScale(self._image_gray, 1.9, 5)
        result = dlib.rectangles()
        for (x, y, w, h) in frame:
            left = x
            top = y
            right = (x + w)
            bottom = (y + h)
            face = dlib.rectangle(left, top, right, bottom)
            result.append(face)
            return result
        return result

    def _use_dlib_detector(self):
        return self._detector(self._image_gray, 0)

    def _predict_landmarks(self, face):
        return self._predictor(self._image_gray, face)
