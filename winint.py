import win32gui
import win32ui
import win32api
import win32con
import numpy
import PIL.Image

from utils import Point, Resolution
from config import CFG

from time import sleep


class GameWindowNotFoundException(Exception):
    def __init__(self):
        pass
        # super(GameWindowNotFoundException, "GameWindowNotFound")


class GameWindow:
    def __init__(self, ms_id_name):
        hwnd = win32gui.FindWindow(None, ms_id_name)
        if not hwnd:
            raise GameWindowNotFoundException()

        (
            window_start_point_x,
            window_start_point_y,
            window_end_point_x,
            window_end_point_y
        ) = win32gui.GetWindowRect(hwnd)
        # screen_resolution = Point(
        #     win32api.GetSystemMetrics(0),
        #     win32api.GetSystemMetrics(1)
        # )

        window_start_point_x = round(window_start_point_x * CFG.resolution_scale_bug_coefficient)
        window_start_point_y = round(window_start_point_y * CFG.resolution_scale_bug_coefficient)
        window_end_point_x = round(window_end_point_x * CFG.resolution_scale_bug_coefficient)
        window_end_point_y = round(window_end_point_y * CFG.resolution_scale_bug_coefficient)

        wDC = win32gui.GetWindowDC(hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(
            dcObj,
            window_end_point_x - window_start_point_x,
            window_end_point_y - window_start_point_y
        )
        cDC.SelectObject(dataBitMap)

        object_DC = dcObj
        compatible_DC = cDC
        buffer_dataBitMap = dataBitMap

        self.hwnd = hwnd
        self.start_point = Point(window_start_point_x, window_start_point_y)
        self.end_point = Point(window_end_point_x, window_end_point_y)
        self.resolution = Resolution(
            window_end_point_x - window_start_point_x,
            window_end_point_y - window_start_point_y
        )
        self.object_DC = object_DC
        self.compatible_DC = compatible_DC
        self.buffer_dataBitMap = buffer_dataBitMap

    def _set_game_window_foreground(self):
        win32gui.ShowWindow(self.hwnd, 1)
        win32gui.SetActiveWindow(self.hwnd)
        win32gui.SetForegroundWindow(self.hwnd)

    def _restore_window(self):
        win32gui.ShowWindow(self.hwnd, 1)

    def screenshot(self):
        self._restore_window()
        self.compatible_DC.BitBlt(
            (0, 0),
            (
                self.end_point.x - self.start_point.x,
                self.end_point.y - self.start_point.y,
            ),
            self.object_DC,
            (0, 0),
            win32con.SRCCOPY
        )

        bmpinfo = self.buffer_dataBitMap.GetInfo()
        bmparray = numpy.asarray(self.buffer_dataBitMap.GetBitmapBits(), dtype=numpy.uint8)
        pil_im = PIL.Image.frombuffer('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmparray, 'raw', 'BGRX', 0, 1)
        cv_im = numpy.array(pil_im)

        return cv_im

    def click(self, point: Point):
        self._set_game_window_foreground()
        point += self.start_point
        point = point.scale(1/CFG.resolution_scale_bug_coefficient)
        win32api.SetCursorPos(point.totuple())
        for _ in range(10):
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, point.x, point.y, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, point.x, point.y, 0, 0)

