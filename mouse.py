import cv
import freenect
import frame_convert

from Xlib.display import Display
from Xlib.ext.xtest import fake_input
from Xlib import X

class Kinect(object):
    dim = (640, 480)

    def __init__(self):
        super(Kinect, self).__init__()
        self.raw_depth = cv.CreateImage(self.dim, 8, 1)
        self.raw_video = cv.CreateImage(self.dim, 8, 3)
        self.contours = cv.CreateImage(self.dim, 8, 1)
        self.thresh = 115 # Depth cutoff for detection, 0 to 255
        self.detect_height = 130

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

    def compute_bounds(self, contour):
        tip = self.compute_tip(contour)
        contour = filter(lambda (x, y): y < tip[1] + self.detect_height, contour) # Filter out arms

        lh = min(contour, key=lambda (x,y): x)
        rh = max(contour, key=lambda (x,y): x)

        return tip, lh, rh

    def compute_tip(self, contour):
        return min(contour, key=lambda (x,y): y)

class Window(object):
    def __init__(self):
        self.display = Display()
        self.root = self.display.screen().root

    def active_window(self):
        window_id = self.root.get_full_property(self.display.intern_atom('_NET_ACTIVE_WINDOW'), X.AnyPropertyType).value[0]
        window = self.display.create_resource_object('window', window_id)

    def find_window(self, title):
        window_ids = self.root.get_full_property(self.display.intern_atom('_NET_CLIENT_LIST'), X.AnyPropertyType).value
        for window_id in window_ids:
            window = self.display.create_resource_object('window', window_id)
            if title == window.get_wm_name():
                return window
        return None

    def resize(self, window, dim):
        window.configure(width = dim[0], height = dim[1])
        self.display.sync()

    def move(self, window, pos):
        window.configure(x = pos[0], y = pos[1])
        self.display.sync()

    def shape(self, window):
        geo = window.get_geometry()
        return (geo.width, geo.height)


class Mouse(object):
    kin_base = (180, 20)
    kin_dim = (320, 240)
    anchor_thresh = 115
    click_thresh = 160

    def __init__(self):
        self.anchor = None
        self.mousedown = False
        self.display = Display()

        geo = self.display.screen().root.get_geometry()
        self.sys_dim = (geo.width, geo.height)

    def process_dual(sel, p1, p2):
        pass

    def process(self, point, w):
        # Width exceeds click_thresh => click action
        if w > self.click_thresh:
            if not self.mousedown:
                self.mousedown = True
                self.click(1)
        else:
            if self.mousedown:
                self.mousedown = False

        # Width exceeds anchor_thresh => anchor the cursor for precision
        if w > self.anchor_thresh:
            if not self.anchor:
                self.anchor = point
            self.anchor_move(point)
        else:
            if self.anchor:
                self.anchor = None
            self.move(point)

    def anchor_move(self, point):
        dx, dy = point[0] - self.anchor[0], point[1] - self.anchor[1]
        self.move((self.anchor[0] + dx / 3.0, self.anchor[1] + dy / 3.0))

    def move(self, point):
        x, y = point
        x = (x - self.kin_base[0]) / float(self.kin_dim[0]) * self.sys_dim[0]
        y = (y - self.kin_base[1]) / float(self.kin_dim[1]) * self.sys_dim[1]
        fake_input(self.display, X.MotionNotify, x=x, y=y)
        self.display.sync()

    def click(self, button):
        fake_input(self.display, X.ButtonPress, button)
        fake_input(self.display, X.ButtonRelease, button)
        self.display.sync()


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
            return False # Discard jitter

        self.data[self.index] = point # Load data into the buffer
        self.index = (self.index + 1) % self.maxdata

        # Recompute the average
        self.avg = (self.avg[0] + (point[0] - old[0]) / float(self.maxdata),
            self.avg[1] + (point[1] - old[1]) / float(self.maxdata))

        return True

    def average(self):
        return int(self.avg[0]), int(self.avg[1])

blue = cv.RGB(17, 110, 255)

def draw_point(video, p):
    cv.Rectangle(video, p, (p[0] + 5, p[1] + 5) ,blue)

if __name__ == '__main__':
    kin = Kinect()
    mouse_sm = Smoother(8, 1e5, 5) # Smoothers for input
    mouse_sm2 = Smoother(8, 1e5, 5)
    width_sm = Smoother(3, 1e2, 3) # Smoother for width
    mouse = Mouse()
    wind = Window()

    title = "qazwsxedcrfvtgbyhnuj"
    cv.NamedWindow(title)
    small = cv.CreateImage((320, 240), 8, 3)
    window = None

    while True:
        kin.next_frame()
        video = kin.video()
        contour = kin.find_contours()

        if contour:
            if not contour.h_next(): # Single input
                tip, lh, rh = kin.compute_bounds(list(contour))
                jitter = not mouse_sm.register(tip)
                x, y = mouse_sm.average()

                w = rh[0] - lh[0]
                width_sm.register((w, w))
                w, _ = width_sm.average()

                anchor = mouse.anchor
                if anchor:
                    draw_point(video, anchor)
                    cv.Line(video, (anchor[0] + 2, anchor[1] + 2), (x + 2, y + 2), blue)
                draw_point(video, (x, y))
                draw_point(video, (lh[0], y))
                draw_point(video, (rh[0], y))

                if not jitter:
                    mouse.process((x, y), w)
            else: # Multiple input
                tip = kin.compute_tip(list(contour))
                tip2 = kin.compute_tip(list(contour.h_next()))

                mouse_sm.register(tip)
                mouse_sm2.register(tip2)
                p1 = mouse_sm.average()
                p2 = mouse_sm2.average()

                mouse.process_dual(p1, p2)

            #cv.DrawContours(video, contour, blue, blue, -1)


        x, y, w, h = mouse.kin_base[0], mouse.kin_base[1], mouse.kin_dim[0], mouse.kin_dim[1]
        cv.Rectangle(video, (x, y), (x + w, y + h), blue)

        cv.Resize(video, small)
        cv.ShowImage(title, small)

        if window == None:
            window = wind.find_window(title)
            if window:
                wind.move(window, (mouse.sys_dim[0] - 320, mouse.sys_dim[1] - 240))

        if cv.WaitKey(10) == 27:
            break
