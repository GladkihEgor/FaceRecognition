import cv2 as cv


class View:
    title = 'Face recognition'
    on = 0
    off = 1
    _show_rectangle = 'Show rectangle'
    _show_dots = 'Show dots'
    _count = 'Dots count'
    _image = None

    def set_image(self, image):
        self._image = image

    def setup(self):
        cv.namedWindow(self.title)
        cv.createTrackbar(self._count, self.title, 0, 5, self._nothing)
        cv.createTrackbar(self._show_rectangle, self.title, self.on, self.off, self._nothing)
        cv.createTrackbar(self._show_dots, self.title, self.on, self.off, self._nothing)

    def create_trackbars(self):
        is_show_rectangle = cv.getTrackbarPos(self._show_rectangle, self.title)
        dots_count = cv.getTrackbarPos(self._count, self.title)
        is_show_dots = cv.getTrackbarPos(self._show_dots, self.title)
        return is_show_rectangle, dots_count, is_show_dots

    def show_image(self):
        cv.imshow(self.title, self._image)

    @staticmethod
    def close():
        cv.destroyAllWindows()

    def draw_face_rectangle(self, face):
        x1 = face.left()
        x2 = face.right()
        y1 = face.top()
        y2 = face.bottom()
        cv.rectangle(self._image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def draw_landmarks(self, landmarks, count):
        for n in count:
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv.circle(self._image, (x, y), 3, (255, 0, 0), -1)

    def add_people_count_text(self, count):
        cv.putText(img=self._image,
                   text=f'Number of recognized people: {count}',
                   org=(10, 50),
                   fontFace=cv.FONT_HERSHEY_DUPLEX,
                   fontScale=1,
                   color=(255, 255, 255))

    def _nothing(self, *args):
        pass
