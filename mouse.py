import cv
import freenect
import frame_convert
import math

from Xlib.display import Display
from Xlib.ext.xtest import fake_input
from Xlib import X

class Kinect(object):
    dim = (640, 480) # Dimensions of the video feed

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
        self.title = 'qazwsxedcrfvtgbyhnuj'
        self.video = None

    def active_window(self):
        window_id = self.root.get_full_property(self.display.intern_atom('_NET_ACTIVE_WINDOW'), X.AnyPropertyType).value[0]
        window = self.display.create_resource_object('window', window_id)
        return window

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
        window.set_wm_name('help')
        self.display.sync()

    def destroy(self, window):
        window.destroy()
        self.display.sync()

    def shape(self, window):
        geo = window.get_geometry()
        return (geo.width, geo.height)

    def position(self, window):
        p = window.query_pointer()
        return (p.root_x - p.win_x, p.root_y - p.win_y - 28) # The 28 seems to be the taskbar


class Mouse(object):
    kin_base = (180, 20) # U-right corner of the detection window
    kin_dim = (320, 240) # Dimensions of the detection window

    anchor_thresh = 115 # Width threshold for anchoring
    anchor_speed = 3.0 # Anchored move zoom scale
    click_thresh = 160 # Width threshold for clicking

    scroll_thresh = 35 # Height threshold for scrolling
    scroll_speed = 0.001 # Inverse of rate at which scroll anchor catches up

    grab_thresh = 270 # Distance threshold for grabbing a window
    drag_thresh = 30 # Distance threshold for dragging a window around
    drag_speed = 25.0 # Inverse of rate at which drag anchor catches up
    drag_scale = (0.2, 0.2) # Scale to translate kinect drag actions into onscreen pixels
    destroy_thresh = 500 # Distance threshold for destroying a window

    def __init__(self, wind):
        self.anchor = None
        self.dual_anchor = None
        self.mousedown = False
        self.active_window = None
        self.dual_mode = 'scroll'

        self.x = 0
        self.y = 0
        self.clickx = 0
        self.clicky = 0

        self.wind = wind
        self.sys_dim = wind.shape(wind.root)

    def dist(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def process_dual(self, pl, pr):
        self.anchor = None
        self.mousedown = True

        if self.dual_anchor == None:
            self.destroyable = self.dist(pl, pr) < self.destroy_thresh * 2 / 3 # Make gesture destroy difficult; must drag and expand
            self.active_window = self.wind.active_window()
            self.dual_anchor = pr

            if self.wind.shape(self.active_window)[0] != self.sys_dim[0]: # Skip maximized windows
                if self.dist(pl, pr) < self.grab_thresh: # Grab the window if the points are close enough
                    self.dual_mode = 'move'
                else:
                    self.x, self.y = self.clickx, self.clicky # Center scrolling on the last clicked spot
                    self.reposition()
            else: # Scrolling
                self.x, self.y = self.clickx, self.clicky
                self.reposition()
        else:
            if self.dual_mode == 'move': # Window operations
                d = self.dist(self.dual_anchor, pr)
                if d > self.drag_thresh:
                    dx, dy = pr[0] - self.dual_anchor[0], pr[1] - self.dual_anchor[1]
                    self.dual_anchor = (int(self.dual_anchor[0] + dx / self.drag_speed), int(self.dual_anchor[1] + dy / self.drag_speed))
                    x, y = self.wind.position(self.active_window)
                    self.wind.move(self.active_window, (x + self.drag_scale[0] * dx, y + self.drag_scale[1] * dy))
            elif self.dual_mode == 'scroll': # Scrolling
                diff = pr[1] - self.dual_anchor[1]
                if math.fabs(diff) > self.scroll_thresh:
                    self.dual_anchor = (self.dual_anchor[0], int(self.dual_anchor[1] + 1 / (diff * self.scroll_speed)))
                    if diff < 0:
                        self.click(4)
                    else:
                        self.click(5)

        if self.destroyable and self.dist(pl, pr) > self.destroy_thresh and pr[1] > self.sys_dim[1] / 2:
            self.wind.destroy(self.active_window)
            self.active_window = self.wind.active_window()
            self.dual_mode = 'scroll'
            self.destroyable = False
            cv.WaitKey(500)

    def process(self, point, w):
        self.dual_anchor = None
        self.active_window = None
        self.dual_mode = 'scroll'

        if w > self.click_thresh:
            if not self.mousedown:
                self.mousedown = True
                self.click(1)
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

    def anchor_move(self, point):
        dx, dy = point[0] - self.anchor[0], point[1] - self.anchor[1]
        self.move((self.anchor[0] + dx / self.anchor_speed, self.anchor[1] + dy / self.anchor_speed))

    def move(self, point):
        x, y = point
        self.x = (x - self.kin_base[0]) / float(self.kin_dim[0]) * self.sys_dim[0]
        self.y = (y - self.kin_base[1]) / float(self.kin_dim[1]) * self.sys_dim[1]
        self.reposition()

    def reposition(self):
        fake_input(self.wind.display, X.MotionNotify, x=self.x, y=self.y)
        self.wind.display.sync()

    def click(self, button):
        self.clickx = self.x
        self.clicky = self.y

        fake_input(self.wind.display, X.ButtonPress, button)
        fake_input(self.wind.display, X.ButtonRelease, button)
        self.wind.display.sync()


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

    def flush(self, point):
        self.jitter_count = 0
        for i, _ in enumerate(self.data):
            self.data[i] = point
        self.avg = point

    def average(self):
        return int(self.avg[0]), int(self.avg[1])

blue = cv.RGB(17, 110, 255)

def draw_point(video, p):
    cv.Circle(video, p, 10 ,blue)

if __name__ == '__main__':
    kin = Kinect()
    mouse_sm = Smoother(8, 1e5, 5) # Smoothers for input
    mouse_sm2 = Smoother(8, 1e5, 5)
    width_sm = Smoother(3, 1e2, 3) # Smoother for width
    wind = Window()
    mouse = Mouse(wind)

    cv.NamedWindow(wind.title)
    small = cv.CreateImage((320, 240), 8, 3)
    active = True
    inputs = 1

    while True:
        kin.next_frame()
        video = kin.video()
        contour = kin.find_contours()

        if not active:
            if contour:
                _, _, w, _ = cv.BoundingRect(contour)
                if w > Kinect.dim[0] / 2:
                    cv.WaitKey(500)
                    mouse.mousedown = True
                    active = True
        else:
            if contour:
                if not contour.h_next(): # Single input
                    tip, lh, rh = kin.compute_bounds(list(contour))
                    jitter = not mouse_sm.register(tip)
                    x, y = mouse_sm.average()

                    w = rh[0] - lh[0]
                    width_sm.register((w, w))
                    w, _ = width_sm.average()

                    if not jitter:
                        mouse.process((x, y), w)
                    inputs = 1

                    anchor = mouse.anchor
                    if anchor:
                        draw_point(video, anchor)
                        cv.Line(video, anchor, (x, y), blue)
                    draw_point(video, (x, y))
                    draw_point(video, (lh[0], y))
                    draw_point(video, (rh[0], y))

                    _, _, _, h = cv.BoundingRect(contour)
                    if h > Kinect.dim[1] * 5 / 6 and tip[0] < Kinect.dim[0] / 5:
                        active = False
                        mouse.x, mouse.y = mouse.sys_dim[0] / 2, mouse.sys_dim[1] / 2
                        mouse.reposition()

                else: # Multiple input
                    tips = (kin.compute_tip(list(contour)), kin.compute_tip(list(contour.h_next())))
                    tipl = min(tips, key=lambda (x, y): x)
                    tipr = max(tips, key=lambda (x, y): x)

                    if inputs == 1: # For drag and release scrolling, flush the buffer on each drag
                        mouse_sm.flush(tipl)
                        mouse_sm2.flush(tipr)
                    else:
                        mouse_sm.register(tipl)
                        mouse_sm2.register(tipr)

                    pl = mouse_sm.average()
                    pr = mouse_sm2.average()

                    mouse.process_dual(pl, pr)
                    inputs = 2

                    anchor = mouse.dual_anchor
                    if anchor:
                        draw_point(video, anchor)
                        cv.Line(video, anchor, pr, blue)
                    draw_point(video, tipl)
                    draw_point(video, tipr)


            x, y, w, h = mouse.kin_base[0], mouse.kin_base[1], mouse.kin_dim[0], mouse.kin_dim[1]
            cv.Rectangle(video, (x, y), (x + w, y + h), blue)

            cv.Resize(video, small)
            cv.ShowImage(wind.title, small)

            if not wind.video:
                window = wind.find_window(wind.title)
                if window:
                    wind.move(window, (mouse.sys_dim[0] - 320, 0))
                    wind.video = window

        k = cv.WaitKey(10)
        if k == 32:
            active = not active
        elif k == 27: # Esc
            break
