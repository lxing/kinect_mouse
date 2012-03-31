import cv
import freenect
import frame_convert
import pymouse

class Kinect(object):
    dim = (640, 480)

    def __init__(self):
        super(Kinect, self).__init__()
        self.raw_depth = cv.CreateImage(self.dim, 8, 1)
        self.raw_video = cv.CreateImage(self.dim, 8, 3)
        self.contours = cv.CreateImage(self.dim, 8, 1)
        self.thresh = 115 # Depth cutoff for detection, 0 to 255

    def next_frame(self):
        self.raw_depth = frame_convert.pretty_depth_cv(freenect.sync_get_depth()[0])
        self.raw_video = frame_convert.video_cv(freenect.sync_get_video()[0])
        cv.Flip(self.raw_depth, None, 1)
        cv.Flip(self.raw_video, None, 1)

    def depth(self):
        return self.raw_depth

    def video(self):
        return self.raw_video

    def find_contours(self):
        cv.Not(self.raw_depth, self.contours)

        cv.Threshold(self.contours, self.contours, self.thresh, 255, cv.CV_THRESH_TOZERO)
        cv.Dilate(self.contours, self.contours, None, 18)
        cv.Erode(self.contours, self.contours, None, 10)

        storage = cv.CreateMemStorage(0)
        return cv.FindContours(self.contours, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE)

class Mouse(object):
    kin_base = (180, 20)
    kin_dim = (320, 240)
    sys_dim = (1366, 768)
    anchor_thresh = 115
    click_thresh = 160

    def __init__(self):
        self.pm = pymouse.PyMouse()
        self.x, self.y = (0, 0)
        self.anchor = None
        self.mousedown = False

    def process(self, point, w):
        if w > self.click_thresh:
            if not self.mousedown:
                self.mousedown = True
                self.pm.click(self.x, self.y)
        else:
            if self.mousedown:
                self.mousedown = False

        if w > self.anchor_thresh:
            if not self.anchor:
                self.anchor = point
            self.anchor_move(point)
        else:
            if self.anchor:
                self.anchor = None
            self.move(point)

    def kinect_bounds(self):
        return self.kin_base[0], self.kin_base[1], self.kin_dim[0], self.kin_dim[1]

    def anchor_move(self, point):
        dx, dy = point[0] - self.anchor[0], point[1] - self.anchor[1]
        self.move((self.anchor[0] + dx / 3.0, self.anchor[1] + dy / 3.0))

    def move(self, point):
        x, y = point
        self.x = (x - self.kin_base[0]) / float(self.kin_dim[0]) * self.sys_dim[0]
        self.y = (y - self.kin_base[1]) / float(self.kin_dim[1]) * self.sys_dim[1]
        self.pm.move(self.x, self.y)


class Smoother(object):
    def __init__(self, maxdata, jitter_thresh, jitter_max):
        self.index = 0
        self.data = [(0,0) for _ in range(maxdata)]
        self.jitter_count = 0
        self.avg = (0,0)

        self.maxdata = maxdata # Number of points to interpolate
        self.jitter_thresh = jitter_thresh # Lower bound for a point to be considered a jitter anomaly
        self.jitter_max = jitter_max # Maximum number of jitter points to discard

    def jitter(self, old, new):
        """
        Ignore jerky movements until seeing jitter_max of them in a row (fast but legit movement)
        Then disable jitter correction until the data stream returns two consecutive non-jerky points
        """
        d = (old[0] - new[0])**2 + (old[1] - new[1])**2
        if d < self.jitter_thresh:
            self.jitter_count = 0
            return False
        else:
            if self.jitter_count == self.jitter_max:
                return False
            else:
                self.jitter_count += 1
                return True


    def register(self, point):
        old = self.data[self.index]
        if self.jitter(old, point):
            return True

        self.data[self.index] = point
        self.index = (self.index + 1) % self.maxdata

        self.avg = (self.avg[0] + (point[0] - old[0]) / float(self.maxdata),
            self.avg[1] + (point[1] - old[1]) / float(self.maxdata))

        return False

    def average(self):
        return int(self.avg[0]), int(self.avg[1])

def compute_bounds(contour):
    contour = list(contour)

    tip = min(contour, key=lambda (x,y): y)
    contour = filter(lambda (x, y): y < tip[1] + 130, contour) # Filter out the arm

    lh = min(contour, key=lambda (x,y): x)
    rh = max(contour, key=lambda (x,y): x)

    return tip, lh, rh

if __name__ == '__main__':
    blue = cv.RGB(17, 110, 255)

    kin = Kinect()
    mouse_sm = Smoother(8, 1e5, 5)
    width_sm = Smoother(3, 1e2, 8)
    mouse = Mouse()

    cv.NamedWindow("Hi")
    small = cv.CreateImage((320, 240), 8, 3)

    while True:
        kin.next_frame()
        video = kin.video()
        contour = kin.find_contours()

        if contour:
            tip, lh, rh = compute_bounds(contour)
            jitter = mouse_sm.register(tip)
            x, y = mouse_sm.average()

            w = rh[0] - lh[0]
            width_sm.register((w, w))
            w, _ = width_sm.average()

            anchor = mouse.anchor
            if anchor:
                cv.Rectangle(video, (anchor[0], anchor[1]), (anchor[0] + 5, anchor[1] + 5), blue)
                cv.Line(video, (anchor[0] + 2, anchor[1] + 2), (x + 2, y + 2), blue)
            cv.Rectangle(video, (x, y), (x + 5, y + 5), blue)
            cv.Rectangle(video, (lh[0], y), (lh[0] + 5, y + 5), blue)
            cv.Rectangle(video, (rh[0], y), (rh[0] + 5, y + 5), blue)
            #cv.DrawContours(video, contour, blue, blue, -1)

            if not jitter:
                mouse.process((x, y), w)

        x, y, w, h = mouse.kinect_bounds()
        cv.Rectangle(video, (x, y), (x + w, y + h), blue)

        cv.Resize(video, small)
        cv.ShowImage("Hi", small)

        if cv.WaitKey(10) == 27:
            break
