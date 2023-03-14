from __future__ import annotations

from copy import copy, deepcopy
import PIL.Image
import numpy
from enum import Enum
from typing import Any, Tuple, Optional, Iterable, Generator, Set, Type


class color:
    white = numpy.asarray([255]*3)
    black = numpy.asarray([0]*3)
    red = numpy.asarray([255, 0, 0])
    green = numpy.asarray([0, 255, 0])
    blue = numpy.asarray([0, 0, 255])


Px = int
no_mines = int


class Cartesian2d:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def totuple(self) -> Tuple[int, int]:
        return self.x, self.y

    def scale(self, coefficient: float) -> Cartesian2d:
        return Cartesian2d(round(self.x * coefficient), round(self.y * coefficient))

    def inverse(self) -> Cartesian2d:
        return Cartesian2d(self.y, self.x)

    def mirror(self) -> Cartesian2d:
        return Cartesian2d(-self.x, -self.y)

    def __add__(self, other: Cartesian2d) -> Cartesian2d:
        return Cartesian2d(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Cartesian2d) -> Cartesian2d:
        return Cartesian2d(self.x - other.x, self.y - other.y)

    def __mul__(self, other: int) -> Cartesian2d:
        return Cartesian2d(other * self.x, other * self.y)

    def __eq__(self, other: Cartesian2d) -> bool:
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return self.totuple().__hash__()

    def __str__(self):
        return "Carthesian({}, {})".format(self.x, self.y)


class Point(Cartesian2d):
    def __init__(self, x: Px, y: Px):
        super().__init__(x, y)
class Resolution(Cartesian2d):
    def __init__(self, x: Px, y: Px):
        super().__init__(x, y)
class UnitSquareID(Cartesian2d):
    def __init__(self, x: int, y: int):
        super().__init__(x, y)
class RelativeUnitSquareID(UnitSquareID):
    def __init__(self, x: int, y: int):
        super().__init__(x, y)
class UnitResolution(Cartesian2d):
    def __init__(self, x: int, y: int):
        super().__init__(x, y)


class MineHint(Enum):
    UNDEFINED = -1
    M0 = 0
    M1 = 1
    M2 = 2
    M3 = 3
    M4 = 4
    M5 = 5
    M6 = 6
    M7 = 7
    M8 = 8
    EXPLOSION = 9

    def get_no_mines(self):
        mine_hints = {
            MineHint.M0: 0,
            MineHint.M1: 1,
            MineHint.M2: 2,
            MineHint.M3: 3,
            MineHint.M4: 4,
            MineHint.M5: 5,
            MineHint.M6: 6,
            MineHint.M7: 7,
            MineHint.M8: 8,
        }
        return mine_hints[self]


class SquareStatus(Enum):
    MARKED = -2
    UNSOLVED = -1
    HINT_UNSOLVED = 0
    HINT_SOLVED = 1


# GuiSquareBinding = {
#     SquareStatus.MARKED: 'F',
#     SquareStatus.UNCOVERED: 'X',
#     SquareStatus.M0: '.',
#     SquareStatus.M1: '1',
#     SquareStatus.M2: '2',
#     SquareStatus.M3: '3',
#     SquareStatus.M4: '4',
#     SquareStatus.M5: '5',
#     SquareStatus.M6: '6',
#     SquareStatus.M7: '7',
#     SquareStatus.M8: '8',
#     SquareStatus.UNKNOWN: '?'
# }


def show_cvimage(cvimg: numpy.ndarray):
    im = PIL.Image.fromarray(cvimg)
    im.show("dupa")


class SurroundMask:
    def __init__(self, center: Cartesian2d, mask: Iterable[Iterable[int]]):
        self.center = center
        self.mask = numpy.asarray(mask).astype(bool)

    def get_mutual(self, other: SurroundMask, relative_center_position: Cartesian2d = Cartesian2d(0, 0)) -> SurroundMask:
        op = deepcopy(self)
        op.mask = numpy.zeros(op.mask.shape, dtype=bool)

        other_position_start = self.center - other.center + relative_center_position
        other_position_end = self.center + other.center + relative_center_position + Cartesian2d(1, 1)

        sx = min(self.center.x * 2 + 1, max(0, other_position_start.x))
        ex = min(self.center.x * 2 + 1, max(0, other_position_end.x))
        sy = min(self.center.x * 2 + 1, max(0, other_position_start.y))
        ey = min(self.center.x * 2 + 1, max(0, other_position_end.y))

        osx = min(other.center.x * 2 + 1, max(0, sx - other_position_start.x))
        oex = min(other.center.x * 2 + 1, max(0, ex - other_position_start.x))
        osy = min(other.center.x * 2 + 1, max(0, sy - other_position_start.y))
        oey = min(other.center.x * 2 + 1, max(0,  ey - other_position_start.y))

        op.mask[sy:ey, sx:ex] = numpy.logical_and(
            self.mask[sy:ey, sx:ex],
            other.mask[osy:oey, osx:oex]
        )

        return op

    def get_exclude(self, other: SurroundMask, relative_center_position: Cartesian2d = Cartesian2d(0, 0)) -> SurroundMask:
        op = deepcopy(self)
        debug = self.get_mutual(other, relative_center_position).mask #DG
        op.mask[self.get_mutual(other, relative_center_position).mask] = False
        return op

    def get_sum(self, other: SurroundMask, relative_center_position: Cartesian2d = Cartesian2d(0, 0)) -> SurroundMask:
        op = deepcopy(self)
        other_centered = SurroundMask(self.center, SurroundMask(self.center, numpy.ones(self.mask.shape)).get_mutual(other, relative_center_position).mask)
        op.mask = numpy.logical_or(self.mask, other_centered.mask)
        return op

    def get_shift(self, relative_center_position: Cartesian2d):
        op = copy(self)
        op.mask = numpy.ones(op.mask.shape, dtype=bool)
        return op.get_mutual(self, relative_center_position)

    def get_mask_pos_change(self, relative_center_position):
        return self.get_shift(relative_center_position.mirror())

    def get_negation(self):
        op = deepcopy(self)
        op.mask = numpy.logical_not(self.mask)
        return op

    def get_coordinate_negation(self, coordinate: Cartesian2d) -> SurroundMask:
        op = deepcopy(self)
        op[coordinate] = not op[coordinate]
        return op

    def get_center_negation(self) -> SurroundMask:
        return self.get_coordinate_negation(Cartesian2d(0, 0))

    def any(self) -> bool:
        return numpy.any(self.mask)

    def is_contained_in(self, other: SurroundMask) -> bool:
        return not self.get_exclude(other).any()

    def count_trues(self) -> int:
        return int(numpy.sum(self.mask.astype(int), None))

    def relative_coordinates_generator(self) -> Generator[Cartesian2d, None, None]:
        ny_base = numpy.arange(self.mask.shape[0]) - self.center.y
        nx_base = numpy.arange(self.mask.shape[1]) - self.center.x
        unit_cords = [Cartesian2d(nx, ny) for ny in ny_base for nx in nx_base]
        for unit_cord in unit_cords:
            if self[unit_cord]:
                yield unit_cord

    def __getitem__(self, item: Cartesian2d) -> bool:
        return self.mask[(item + self.center).y, (item + self.center).x]

    def __setitem__(self, key: Cartesian2d, value):
        self.mask[(key + self.center).y, (key + self.center).x] = value


class AdjacentMask(SurroundMask):
    setbit_grouping = [[] for _ in range(10)]

    def __init__(self, mask: Optional[Iterable[Iterable[int]], bool]):
        if type(mask) == bool:
            mask = numpy.ones((3, 3)) * mask
        super().__init__(Cartesian2d(1, 1), mask)

    def get_mirror(self) -> AdjacentMask:
        op = AdjacentMask(False)
        for mask_element_id in self.relative_coordinates_generator():
            op[mask_element_id.mirror()] = self[mask_element_id]
        return op

    def submask_generator(self, no_trues: Optional[int]) -> Generator[AdjacentMask, None, None]:
        unit_masks_no = 0
        unit_masks_weights = numpy.zeros((3, 3), dtype=int)
        for unit_cord in self.relative_coordinates_generator():
            unit_mask = AdjacentMask(False)
            unit_mask[unit_cord] = True
            if unit_mask.is_contained_in(self):
                unit_masks_weights += unit_mask.mask * (2 ** unit_masks_no)
                unit_masks_no += 1

        if type(no_trues) is int:
            for i in self.setbit_grouping[no_trues]:
                if not (i < 2 ** unit_masks_no):
                    break
                yield AdjacentMask(numpy.bitwise_and(i, unit_masks_weights).astype(bool))

        elif no_trues is None:
            for i in range(2 ** unit_masks_no):
                yield AdjacentMask(numpy.bitwise_and(i, unit_masks_weights).astype(bool))

    def interval_submask_generator(self, no_trues_start, no_trues_end, include_end=True) -> Generator[AdjacentMask, None, None]:
        for no_trues in range(no_trues_start, no_trues_end + int(include_end)):
            for yield_mask in self.submask_generator(no_trues):
                yield yield_mask


for sbgi in range(2**9):
    no_bits = int(numpy.sum(numpy.bitwise_and(sbgi, 2 ** numpy.arange(9)).astype(bool)))
    AdjacentMask.setbit_grouping[no_bits].append(sbgi)


class LocalMask(SurroundMask):
    def __init__(self, mask: Optional[Iterable[Iterable[int]], bool]):
        if type(mask) == bool:
            mask = numpy.ones((5, 5)) * mask
        super().__init__(Cartesian2d(2, 2), mask)


def mask2cluster_data_id(mask: AdjacentMask):
    return int(numpy.sum(mask.mask * 2 ** numpy.arange(9).reshape((3, 3))))

