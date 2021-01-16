import cv2 as cv


class View:
    title = 'Face recognition'
    on = 0
    off = 1
    _show_rectangle = 'Show rectangle'
    _show_dots = 'Show_dots'
    _count = 'Dots count'

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

    def show_image(self, image):
        cv.imshow(self.title, image)

    @staticmethod
    def close():
        cv.destroyAllWindows()

    @staticmethod
    def draw_face_rectangle(image, points):
        x1 = points.left()
        x2 = points.right()
        y1 = points.top()
        y2 = points.bottom()
        cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    @staticmethod
    def draw_landmarks(landmarks, count, image):
        for n in count:
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv.circle(image, (x, y), 3, (255, 0, 0), -1)

    def _nothing(self, *args):
        pass
