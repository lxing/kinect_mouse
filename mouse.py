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
        self.thresh = 120 # Depth cutoff for detection, 0 to 255

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
    kin_base = (80, 0)
    kin_dim = (480, 360)
    sys_dim = (1366, 768)

    def __init__(self):
        self.pm = pymouse.PyMouse()

    def kinect_bounds(self):
        return self.kin_base[0], self.kin_base[1], self.kin_dim[0], self.kin_dim[1]

    def move(self, point):
        x, y = point
        x = (x - self.kin_base[0]) / float(self.kin_dim[0]) * self.sys_dim[0]
        y = (y - self.kin_base[1]) / float(self.kin_dim[1]) * self.sys_dim[1]
        self.pm.move(x,y)


class Smoother(object):
    maxdata = 9
    jitter_max = 5
    jitter_thresh = 1e5

    def __init__(self):
        self.index = 0
        self.data = [(0,0) for _ in range(self.maxdata)]
        self.jitter_count = 0
        self.avg = (0,0)

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
        if self.jitter(old, point): return

        self.data[self.index] = point
        self.index = (self.index + 1) % self.maxdata

        self.avg = (self.avg[0] + (point[0] - old[0]) / float(self.maxdata),
            self.avg[1] + (point[1] - old[1]) / float(self.maxdata))

    def average(self):
        return int(self.avg[0]), int(self.avg[1])

def compute_bounds(contour):
    contour = list(contour)

    tip = min(contour, key=lambda (x,y): y)
    contour = filter(lambda (x, y): y < tip[1] + 150, contour)

    lh = min(contour, key=lambda (x,y): x)
    rh = max(contour, key=lambda (x,y): x)

    return tip, lh, rh

if __name__ == '__main__':
    white = cv.RGB(17, 110, 255)
    kin = Kinect()
    sm = Smoother()
    mouse = Mouse()

    cv.NamedWindow("Contours")

    while True:
        kin.next_frame()
        video = kin.video()
        contour = kin.find_contours()

        if contour:
            tip, lh, rh = compute_bounds(contour)
            sm.register(tip)
            px, py = sm.average()
            cv.Rectangle(video, (px, py), (px + 5, py + 5), white)
            cv.Rectangle(video, (lh[0], tip[1]), (rh[0], tip[1] + 150), white)
            mouse.move((px, py))

        x, y, w, h = mouse.kinect_bounds()
        cv.Rectangle(video, (x, y), (x + w, y + h), white)

        cv.DrawContours(video, contour, white, white, -1)
        cv.ShowImage("Contours", video)

        if cv.WaitKey(10) == 27:
            break
