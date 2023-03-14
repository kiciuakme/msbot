import cv2
import numpy
from copy import copy
import time

from config import CFG
from utils import UnitSquareID, Point, Resolution, UnitResolution, MineHint, color
from winint import GameWindow

from utils import show_cvimage


class GameLostException(Exception):
    pass


class GameBoardInt:
    def __init__(self, game_window: GameWindow):
        ss = game_window.screenshot()
        ss8 = copy(ss)
        ss8[:, :] = numpy.floor(ss[:, :] / 128) * 255
        sss = numpy.zeros((ss8.shape[:2]))  # black
        sss[numpy.any(ss8 == 255, 2)] = 1  # other

        ss_recog_start_point = Point(
            round((game_window.end_point.x - game_window.start_point.x)/2),
            round((game_window.end_point.y - game_window.start_point.y)/2)
        )

        while not sss[ss_recog_start_point.inverse().totuple()]:
            ss_recog_start_point.x += 1
            ss_recog_start_point.y += 1
        ss_recog_end_point = copy(ss_recog_start_point)

        flag_mods = 1
        while flag_mods:
            flag_mods = 0
            if numpy.any(sss[ss_recog_start_point.y:ss_recog_end_point.y+1, ss_recog_start_point.x]):
                ss_recog_start_point.x -= 1
                flag_mods = 1
            if numpy.any(sss[ss_recog_start_point.y:ss_recog_end_point.y+1, ss_recog_end_point.x]):
                ss_recog_end_point.x += 1
                flag_mods = 1
            if numpy.any(sss[ss_recog_start_point.y, ss_recog_start_point.x:ss_recog_end_point.x+1]):
                ss_recog_start_point.y -= 1
                flag_mods = 1
            if numpy.any(sss[ss_recog_end_point.y, ss_recog_start_point.y:ss_recog_end_point.y+1]):
                ss_recog_end_point.y += 1
                flag_mods = 1

        square_internal_start_x = ss_recog_start_point.x
        square_internal_end_x = ss_recog_end_point.x
        square_internal_start_y = ss_recog_start_point.y
        square_internal_end_y = ss_recog_end_point.y

        flag_mods = 1
        while flag_mods:
            flag_mods = 0
            if not numpy.any(sss[ss_recog_start_point.y:ss_recog_end_point.y + 1, ss_recog_start_point.x]):
                ss_recog_start_point.x -= 1
                flag_mods = 1
            if not numpy.any(sss[ss_recog_start_point.y:ss_recog_end_point.y + 1, ss_recog_end_point.x]):
                ss_recog_end_point.x += 1
                flag_mods = 1
            if not numpy.any(sss[ss_recog_start_point.y, ss_recog_start_point.x:ss_recog_end_point.x + 1]):
                ss_recog_start_point.y -= 1
                flag_mods = 1
            if not numpy.any(sss[ss_recog_end_point.y, ss_recog_start_point.y:ss_recog_end_point.y + 1]):
                ss_recog_end_point.y += 1
                flag_mods = 1

        square_external_start_x = ss_recog_start_point.x
        square_external_end_x = ss_recog_end_point.x
        square_external_start_y = ss_recog_start_point.y
        square_external_end_y = ss_recog_end_point.y

        flag_mods = 1
        while flag_mods:
            flag_mods = 0
            if not numpy.all(sss[ss_recog_start_point.y:ss_recog_end_point.y, ss_recog_start_point.x]):
                ss_recog_start_point.x -= 1
                flag_mods = 1
            if not numpy.all(sss[ss_recog_start_point.y:ss_recog_end_point.y, ss_recog_end_point.x]):
                ss_recog_end_point.x += 1
                flag_mods = 1
            if not numpy.all(sss[ss_recog_start_point.y, ss_recog_start_point.x:ss_recog_end_point.x]):
                ss_recog_start_point.y -= 1
                flag_mods = 1
            if not numpy.all(sss[ss_recog_end_point.y, ss_recog_start_point.y:ss_recog_end_point.y]):
                ss_recog_end_point.y += 1
                flag_mods = 1

        board_external_start_x = ss_recog_start_point.x
        board_external_end_x = ss_recog_end_point.x
        board_external_start_y = ss_recog_start_point.y
        board_external_end_y = ss_recog_end_point.y

        border_thickness_x = square_external_end_x - square_internal_end_x
        border_thickness_y = square_external_end_y - square_internal_end_y
        ss_recog_start_point.x += border_thickness_x
        ss_recog_start_point.y += border_thickness_y
        ss_recog_end_point.x -= border_thickness_x
        ss_recog_end_point.y -= border_thickness_y

        flag_mods = 1
        while flag_mods:
            flag_mods = 0
            if not numpy.any(sss[ss_recog_start_point.y:ss_recog_end_point.y, ss_recog_start_point.x]):
                ss_recog_start_point.x += 1
                flag_mods = 1
            if not numpy.any(sss[ss_recog_start_point.y:ss_recog_end_point.y, ss_recog_end_point.x]):
                ss_recog_end_point.x -= 1
                flag_mods = 1
            if not numpy.any(sss[ss_recog_start_point.y, ss_recog_start_point.x:ss_recog_end_point.x]):
                ss_recog_start_point.y += 1
                flag_mods = 1
            if not numpy.any(sss[ss_recog_end_point.y, ss_recog_start_point.y:ss_recog_end_point.y]):
                ss_recog_end_point.y -= 1
                flag_mods = 1

        board_internal_start_x = ss_recog_start_point.x
        board_internal_end_x = ss_recog_end_point.x
        board_internal_start_y = ss_recog_start_point.y
        board_internal_end_y = ss_recog_end_point.y

        square_pxres = Resolution(
            square_internal_end_x - square_internal_start_x,
            square_internal_end_y - square_internal_start_y
        )

        x_square_start_points = []
        y_square_start_points = []

        while not numpy.any(sss[(board_internal_end_y - square_pxres.y):board_internal_end_y, ss_recog_start_point.x]):
            ss_recog_start_point.x += 1
        while ss_recog_start_point.x < ss_recog_end_point.x:
            x_square_start_points.append(ss_recog_start_point.x)
            while numpy.any(sss[(board_internal_end_y - square_pxres.y):board_internal_end_y, ss_recog_start_point.x]):
                ss_recog_start_point.x += 1
            while not numpy.any(sss[(board_internal_end_y - square_pxres.y):board_internal_end_y, ss_recog_start_point.x]):
                ss_recog_start_point.x += 1

        while not numpy.any(sss[ss_recog_start_point.y, (board_internal_end_x - square_pxres.x):board_internal_end_x]):
            ss_recog_start_point.y += 1
        while ss_recog_start_point.y < ss_recog_end_point.y:
            y_square_start_points.append(ss_recog_start_point.y)
            while numpy.any(sss[ss_recog_start_point.y, (board_internal_end_x - square_pxres.x):board_internal_end_x]):
                ss_recog_start_point.y += 1
            while not numpy.any(sss[ss_recog_start_point.y, (board_internal_end_x - square_pxres.x):board_internal_end_x]):
                ss_recog_start_point.y += 1

        self.game_window = game_window
        self.uncovered_board_sample = copy(ss)
        self.board_sample = ss

        self.board_us_res = UnitResolution(len(x_square_start_points), len(y_square_start_points))
        self.square_pxres = square_pxres
        self.square_start_point = numpy.asarray(
            [[Point(x, y) for y in y_square_start_points] for x in x_square_start_points]
        )

        self.processed_digit_sample = numpy.zeros((10, 20, 20, 3))
        ipd_samples = cv2.imread(CFG.filtered_board_recog_samples)
        processed_digit_samples = cv2.cvtColor(ipd_samples, cv2.COLOR_BGR2RGB)
        for digit_id in range(self.processed_digit_sample.shape[0]):
            self.processed_digit_sample[digit_id] = processed_digit_samples[:, (digit_id * 20):((digit_id+1) * 20), :]

        # #DG STT
        # sswl = copy(ss) * 0
        # sswl[:, :] = color.white/2
        # for cx in range(len(x_square_start_points)):
        #     for cy in range(len(y_square_start_points)):
        #         x = x_square_start_points[cx]
        #         y = y_square_start_points[cy]
        #
        #         ssc = ss[y:(y+self.square_pxres.y), x:(x+self.square_pxres.x)]
        #         ssc = ssc - numpy.ones(ssc.shape) * numpy.min(ssc[:, :], axis=2)[:, :, numpy.newaxis]
        #
        #         sscd = numpy.zeros(ssc.shape)
        #         for i in range(ssc.shape[1]-1):
        #             sscd[:, i] += numpy.abs(ssc[:, i] - ssc[:, i+1])
        #         for j in range(ssc.shape[0]-1):
        #             sscd[j, :] += numpy.abs(ssc[j, :] - ssc[j+1, :])
        #
        #         sscc = sscd[
        #                self.square_pxres.scale(.2).x:self.square_pxres.scale(.8).x,
        #                self.square_pxres.scale(.2).y:self.square_pxres.scale(.8).y,
        #                ]
        #         sscc = cv2.resize(sscc, (20, 20))
        #
        #         r = numpy.sum(sscc[:, :, 0]) / (sscc.shape[0] * sscc.shape[1])
        #         g = numpy.sum(sscc[:, :, 1]) / (sscc.shape[0] * sscc.shape[1])
        #         b = numpy.sum(sscc[:, :, 2]) / (sscc.shape[0] * sscc.shape[1])
        #
        #         print(cx, cy, round(r), round(g), round(b))
        #
        #         sscc[sscc <= 32] = 0
        #         sscc[sscc > 32] = 255
        #
        #         r = numpy.sum(sscc[:, :, 0]) / (sscc.shape[0] * sscc.shape[1])
        #         g = numpy.sum(sscc[:, :, 1]) / (sscc.shape[0] * sscc.shape[1])
        #         b = numpy.sum(sscc[:, :, 2]) / (sscc.shape[0] * sscc.shape[1])
        #
        #         for iy in range(sscc.shape[0]):
        #             fill_recog_x_start = 0
        #             fill_recog_x_end = sscc.shape[1] - 1
        #             while not (numpy.any(sscc[iy, fill_recog_x_start]) or fill_recog_x_start == fill_recog_x_end):
        #                 fill_recog_x_start += 1
        #             while not (numpy.any(sscc[iy, fill_recog_x_end]) or fill_recog_x_start == fill_recog_x_end):
        #                 fill_recog_x_end -= 1
        #
        #             fill_color = color.red if r > g and r > b else color.green if g > b else color.blue
        #             sscc[iy, fill_recog_x_start:fill_recog_x_end] = fill_color
        #
        #         r = numpy.sum(sscc[:, :, 0]) / (sscc.shape[0] * sscc.shape[1])
        #         g = numpy.sum(sscc[:, :, 1]) / (sscc.shape[0] * sscc.shape[1])
        #         b = numpy.sum(sscc[:, :, 2]) / (sscc.shape[0] * sscc.shape[1])
        #
        #         # print(cx, cy, round(r), round(g), round(b))
        #
        #         # cv2.rectangle(sscd, self.square_pxres.scale(.1).totuple(), self.square_pxres.scale(.9).totuple(), (0, 255, 0))
        #
        #         sswl[y:(y + 20), x:(x + 20)] = sscc
        #
        # # cv2.imwrite("__all_samples_20px.bmp", cv2.cvtColor(sswl, cv2.COLOR_RGB2BGR))
        # show_cvimage(sswl)
        # #DG END

    def update_board(self):
        self.board_sample = self.game_window.screenshot()

    def is_square_uncovered(self, square_id: UnitSquareID) -> bool:
        square_start_point = self.square_start_point[square_id.totuple()]
        square_end_point = square_start_point + self.square_pxres
        square_ss = self.board_sample[
                    square_start_point.y:square_end_point.y,
                    square_start_point.x:square_end_point.x
                    ]

        uncovered_square_ss = self.uncovered_board_sample[
                              square_start_point.y:square_end_point.y,
                              square_start_point.x:square_end_point.x
                              ]

        src_coef = CFG.square_recog_crop_coefficient
        diff = numpy.abs(square_ss[
                     self.square_pxres.scale(src_coef).x:self.square_pxres.scale(1 - src_coef).x,
                     self.square_pxres.scale(src_coef).y:self.square_pxres.scale(1 - src_coef).y,
                     ] - uncovered_square_ss[
                          self.square_pxres.scale(src_coef).x:self.square_pxres.scale(1 - src_coef).x,
                          self.square_pxres.scale(src_coef).y:self.square_pxres.scale(1 - src_coef).y,
                          ])

        return numpy.all(diff == 0)

    def recognize_unc_square_mine_hint(self, square_id: UnitSquareID) -> MineHint:
        square_start_point = self.square_start_point[square_id.totuple()]
        square_end_point = square_start_point + self.square_pxres
        square_ss = self.board_sample[
            square_start_point.y:square_end_point.y,
            square_start_point.x:square_end_point.x
        ]

        ssup = square_ss  # square_screenshot_under_processing
        ssup = ssup - numpy.ones(ssup.shape) * numpy.min(ssup[:, :], axis=2)[:, :, numpy.newaxis]

        sscc = numpy.zeros(ssup.shape)
        for i in range(ssup.shape[1] - 1):
            sscc[:, i] += numpy.abs(ssup[:, i] - ssup[:, i + 1])
        for j in range(ssup.shape[0] - 1):
            sscc[j, :] += numpy.abs(ssup[j, :] - ssup[j + 1, :])

        # print(ssup)  # DG
        # show_cvimage(square_ss.astype(numpy.uint8))  # DG

        src_coef = CFG.square_recog_crop_coefficient
        sscc = sscc[
               self.square_pxres.scale(src_coef).x:self.square_pxres.scale(1 - src_coef).x,
               self.square_pxres.scale(src_coef).y:self.square_pxres.scale(1 - src_coef).y,
               ]
        sscc = cv2.resize(sscc.astype(numpy.uint8), (20, 20))

        sscc[sscc <= 32] = 0
        sscc[sscc > 32] = 255

        r = numpy.sum(sscc[:, :, 0])  # / (sscc.shape[0] * sscc.shape[1])
        g = numpy.sum(sscc[:, :, 1])  # / (sscc.shape[0] * sscc.shape[1])
        b = numpy.sum(sscc[:, :, 2])  # / (sscc.shape[0] * sscc.shape[1])

        fill_color = None
        for iy in range(sscc.shape[0]):
            fill_recog_x_start = 0
            fill_recog_x_end = sscc.shape[1] - 1
            while not (numpy.any(sscc[iy, fill_recog_x_start]) or fill_recog_x_start == fill_recog_x_end):
                fill_recog_x_start += 1
            while not (numpy.any(sscc[iy, fill_recog_x_end]) or fill_recog_x_start == fill_recog_x_end):
                fill_recog_x_end -= 1

            fill_color = color.red if r > g and r > b else color.green if g > b else color.blue
            sscc[iy, fill_recog_x_start:fill_recog_x_end] = fill_color

        # forbidden_hints = numpy.ones(10) * float("inf")
        # if not r == g and g == b and b == 0:
        #     forbidden_hints[numpy.arange(1, 9)] = 0
        # else:
        #     forbidden_hints[0] = 0

        square_status = MineHint(
            numpy.argmin(numpy.sum(numpy.abs(self.processed_digit_sample - sscc), axis=(1, 2, 3))))
        if square_status == MineHint.EXPLOSION:
            raise GameLostException
        return square_status

    def uncover_square(self, square_id: UnitSquareID):
        self.game_window.click(self.square_start_point[square_id.totuple()] + self.square_pxres.scale(.5))

