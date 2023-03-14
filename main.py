from __future__ import annotations
import time
import abc
import os

from winint import GameWindow, GameWindowNotFoundException
from brdint import GameBoardInt, GameLostException
from utils import *
from config import CFG
from typing import List, Optional, Iterable, Set


class MaskSolution:
    def __init__(self):
        self._data: int = 0

    @staticmethod
    def usol2key(unitary_solution: no_mines) -> int:
        return 1 << unitary_solution

    def append(self, unitary_solution: no_mines) -> None:
        self._data = self._data | self.usol2key(unitary_solution)

    def remove(self, unitary_solution: no_mines) -> None:
        self._data = self._data & (0x1ff - self.usol2key(unitary_solution))

    def set_as_interval(self, start_usol: no_mines, end_usol: no_mines) -> MaskSolution:
        for usol in range(start_usol, end_usol + 1):
            self.append(usol)
        return self

    def set_as_unitary(self, unitary_solution: no_mines) -> MaskSolution:
        self.append(unitary_solution)
        return self

    def is_including(self, unitary_solution) -> bool:
        return bool(self._data & self.usol2key(unitary_solution))

    def __eq__(self, other: MaskSolution):
        return self._data == other._data

    def __str__(self):
        op = "MSol: "
        for i in range(10):
            if self.is_including(i):
                op += str(i) + ", "
        return op


class IndependentClusterData:
    def __init__(self):
        self._data = numpy.asarray([MaskSolution() for _ in range(2 ** 9)], dtype=MaskSolution)

    def __getitem__(self, mask: AdjacentMask) -> MaskSolution:
        return self._data[mask2cluster_data_id(mask)]

    def __setitem__(self, mask: AdjacentMask, solution: MaskSolution) -> None:
        self._data[mask2cluster_data_id(mask)] = solution


class UniqueClusterContainer:
    _positive_border_ext = UnitResolution(2, 2)

    def __init__(self, size: UnitResolution):
        self._extented_data = numpy.zeros((size + self._positive_border_ext).totuple() + (2 ** 9,), dtype=MaskSolution)

        extended_board_sq_ids = numpy.asarray([UnitSquareID(sx, sy)
                                               for sx in range((size + self._positive_border_ext).x)
                                               for sy in range((size + self._positive_border_ext).y)],
                                              dtype=UnitSquareID)

        idle_solution_templates = [MaskSolution().set_as_interval(0, max_no_mines) for max_no_mines in range(10)]
        ext_data_templates = numpy.zeros((2 ** 9, 2 ** 9), dtype=MaskSolution)

        temp_rel_ids = [rel_id for rel_id in LocalMask(True).get_mutual(
            LocalMask(True), Cartesian2d(-1, -1)).relative_coordinates_generator()] + [RelativeUnitSquareID(-3, -3)]
        for shift_relative_id in temp_rel_ids:
            coverage_mask = AdjacentMask(True).get_mutual(AdjacentMask(True), shift_relative_id)
            coverage_mask_exclusion = AdjacentMask(True).get_exclude(coverage_mask)

            for strict_fill_mask in coverage_mask.submask_generator(None):
                idle_max_no_mines = strict_fill_mask.count_trues()
                idle_solution = idle_solution_templates[idle_max_no_mines]
                for supplement_mask in coverage_mask_exclusion.submask_generator(None):
                    fill_mask = strict_fill_mask.get_sum(supplement_mask)
                    ext_data_templates[mask2cluster_data_id(coverage_mask)][mask2cluster_data_id(fill_mask)] \
                        = idle_solution

        for extended_square_id in extended_board_sq_ids:
            resp_fcm_sid_x = max(1, min((size.x - 2), extended_square_id.x))
            resp_fcm_sid_y = max(1, min((size.y - 2), extended_square_id.y))
            responding_full_coverage_mask_square_id = UnitSquareID(resp_fcm_sid_x, resp_fcm_sid_y)
            resp_fcm_square_relative_id = responding_full_coverage_mask_square_id - extended_square_id

            coverage_mask = AdjacentMask(True).get_mutual(AdjacentMask(True), resp_fcm_square_relative_id)
            self._extented_data[extended_square_id.totuple()] = \
                copy(ext_data_templates[mask2cluster_data_id(coverage_mask)])

    @staticmethod
    def _find_uniques(square_id: UnitSquareID, mask: AdjacentMask) -> Tuple[UnitSquareID, AdjacentMask]:
        unique_relative_id = RelativeUnitSquareID(0, 0)
        unique_relative_id.x += int(mask.get_mask_pos_change(UnitSquareID(1, 0)).count_trues() == mask.count_trues())
        unique_relative_id.x += int(mask.get_mask_pos_change(UnitSquareID(2, 0)).count_trues() == mask.count_trues())
        unique_relative_id.y += int(mask.get_mask_pos_change(UnitSquareID(0, 1)).count_trues() == mask.count_trues())
        unique_relative_id.y += int(mask.get_mask_pos_change(UnitSquareID(0, 2)).count_trues() == mask.count_trues())

        unique_mask = mask.get_mask_pos_change(unique_relative_id)
        unique_id = square_id + unique_relative_id
        return unique_id, unique_mask

    def set_unique(self, square_id: UnitSquareID, mask: AdjacentMask, solution: MaskSolution) -> None:
        unique_id, unique_mask = self._find_uniques(square_id, mask)
        self._extented_data[unique_id.totuple()][mask2cluster_data_id(unique_mask)] = solution

    def get_unique(self, square_id: UnitSquareID, mask: AdjacentMask) -> MaskSolution:
        unique_id, unique_mask = self._find_uniques(square_id, mask)
        return self._extented_data[unique_id.totuple()][mask2cluster_data_id(unique_mask)]


class SquareCluster:
    def __init__(self, square_id: UnitSquareID, unique_cluster_container: UniqueClusterContainer):
        self.square_id = square_id
        self.unicc = unique_cluster_container
        self.square_status = SquareStatus.UNSOLVED
        self.center_hint = MineHint.UNDEFINED

    def __getitem__(self, mask: AdjacentMask) -> MaskSolution:
        return self.unicc.get_unique(self.square_id, mask)

    def __setitem__(self, mask: AdjacentMask, solution: MaskSolution) -> None:
        self.unicc.set_unique(self.square_id, mask, solution)

    def import_cluster_data(self, cluster_data: IndependentClusterData, changes_mask: AdjacentMask):
        for mask in changes_mask.submask_generator(None):
            self[mask] = cluster_data[mask]

    def update_status(self, square_status: SquareStatus, center_hint: MineHint):
        self.square_status = square_status
        self.center_hint = center_hint

    def is_solved(self) -> bool:
        unsolved_solution = MaskSolution().set_as_interval(0, 1)
        for unitary_mask in AdjacentMask(True).submask_generator(1):
            if self[unitary_mask] == unsolved_solution:
                return False
        else:
            return True


class SquareClusterContainer:
    def __init__(self, game_board: GameBoardInt):
        self._game_board = game_board
        self._board_us_res = self._game_board.board_us_res
        self._unicc = UniqueClusterContainer(self._board_us_res)

        self._board = numpy.zeros(self._board_us_res.totuple(), dtype=SquareCluster)

        board_sq_ids = numpy.asarray([UnitSquareID(sx, sy)
                                      for sx in range(self._board_us_res.x)
                                      for sy in range(self._board_us_res.y)], dtype=UnitSquareID)
        for square_id in board_sq_ids:
            self[square_id] = SquareCluster(square_id, self._unicc)

    def __getitem__(self, square_id: UnitSquareID) -> SquareCluster:
        return self._board[square_id.totuple()]

    def __setitem__(self, square_id: UnitSquareID, square_cluster: SquareCluster) -> None:
        self._board[square_id.totuple()] = square_cluster

    def is_inside(self, square_id: UnitSquareID) -> bool:
        return 0 <= square_id.x < self._board_us_res.x and 0 <= square_id.y < self._board_us_res.y


class GameSolver:
    def __init__(self, game_board: GameBoardInt, cluster_container: SquareClusterContainer):
        self.game_board = game_board
        self.clusters = cluster_container

        self.solved_squares_to_recognize = set()
        self.hint_squares_to_update = set()

    def hint_square_cluster_update(self, square_id: UnitSquareID) -> None:
        square_cluster = self.clusters[square_id]
        new_cluster_data = IndependentClusterData()

        new_hstu_mask_raw = LocalMask(False)

        unsolved_mask_base = AdjacentMask(False)
        uncovered_mask_base = AdjacentMask(False)
        marked_mask_base = AdjacentMask(False)

        # EXCLUDE PREVIOUSLY SOLVED UNITARIES FROM UPDATE:
        for unitary_mask in AdjacentMask(True).get_center_negation().submask_generator(1):
            if square_cluster[unitary_mask] == MaskSolution().set_as_interval(0, 1):
                unsolved_mask_base = unsolved_mask_base.get_sum(unitary_mask)
            elif square_cluster[unitary_mask] == MaskSolution().set_as_unitary(1):
                marked_mask_base = marked_mask_base.get_sum(unitary_mask)
            elif square_cluster[unitary_mask] == MaskSolution().set_as_unitary(0):
                uncovered_mask_base = uncovered_mask_base.get_sum(unitary_mask)

        no_mines_solved = marked_mask_base.count_trues()
        no_mines_unsolved = square_cluster.center_hint.get_no_mines() - no_mines_solved
        no_squares_unsolved = unsolved_mask_base.count_trues()

        # EXCLUDE FORBIDDEN SOLUTIONS BY CONDITIONS
        new_cluster_data[unsolved_mask_base] = MaskSolution().set_as_unitary(no_mines_unsolved)

        for test_mine_mask in unsolved_mask_base.submask_generator(no_mines_unsolved):
            for condition_mask in unsolved_mask_base.interval_submask_generator(2, no_squares_unsolved):
                partial_test_mine_mask = test_mine_mask.get_mutual(condition_mask)
                partial_test_mine_mask_no_mines = partial_test_mine_mask.count_trues()
                partial_test_mine_mask_condition = square_cluster[condition_mask]
                if not partial_test_mine_mask_condition.is_including(partial_test_mine_mask_no_mines):
                    break
            else:
                for solution_mask in unsolved_mask_base.interval_submask_generator(1, no_squares_unsolved - 1):
                    solution_no_mines = test_mine_mask.get_mutual(solution_mask).count_trues()
                    new_cluster_data[solution_mask].append(solution_no_mines)

        # HANDLE NEWLY SOLVED UNITARIES SOLUTIONS
        unsolved_mask_base_new_unitaries_excluded = deepcopy(unsolved_mask_base)

        for unsolved_unitary_relative_id in unsolved_mask_base.relative_coordinates_generator():
            unsolved_unitary_mask = AdjacentMask(False).get_coordinate_negation(unsolved_unitary_relative_id)
            new_unitary_solution = new_cluster_data[unsolved_unitary_mask]
            if new_unitary_solution != MaskSolution().set_as_interval(0, 1):
                square_cluster[unsolved_unitary_mask] = new_unitary_solution
                self.solved_squares_to_recognize.add(square_id + unsolved_unitary_relative_id)

                if new_unitary_solution == MaskSolution().set_as_unitary(0):  # SRP VIOLATION! for time perf impovement
                    self.game_board.uncover_square(square_id + unsolved_unitary_relative_id)

                unsolved_mask_base_new_unitaries_excluded[unsolved_unitary_relative_id] = False
                new_hstu_mask_raw = new_hstu_mask_raw.get_sum(AdjacentMask(True), unsolved_unitary_relative_id)

        if not unsolved_mask_base_new_unitaries_excluded.any():
            square_cluster.update_status(SquareStatus.HINT_SOLVED, square_cluster.center_hint)

        # HANDLE NEWLY SOLVED NON-UNITARIES SOLUTIONS
        umbnue_no_squares = unsolved_mask_base_new_unitaries_excluded.count_trues()
        for unsolved_mask in unsolved_mask_base_new_unitaries_excluded.interval_submask_generator(2, umbnue_no_squares):
            if square_cluster[unsolved_mask] != new_cluster_data[unsolved_mask]:
                square_cluster[unsolved_mask] = new_cluster_data[unsolved_mask]

                new_hstu_mask_raw_partial = LocalMask(True)
                for unsolved_mask_unitary_relative_id in unsolved_mask.relative_coordinates_generator():
                    new_hstu_mask_raw_partial = \
                        new_hstu_mask_raw_partial.get_mutual(AdjacentMask(True), unsolved_mask_unitary_relative_id)
                new_hstu_mask_raw = new_hstu_mask_raw.get_sum(new_hstu_mask_raw_partial)

        # HANDLE NEW HINT SQUARES TO UPDATE
        new_hstu_mask_raw[RelativeUnitSquareID(0, 0)] = False
        for new_hstu_raw_relative_id in new_hstu_mask_raw.relative_coordinates_generator():
            new_htsu_raw_id = square_id + new_hstu_raw_relative_id
            if self.clusters.is_inside(new_htsu_raw_id):
                if self.clusters[new_htsu_raw_id].square_status == SquareStatus.HINT_UNSOLVED:
                    self.hint_squares_to_update.add(new_htsu_raw_id)

    def board_recognize_new_squares(self):
        while len(self.solved_squares_to_recognize):
            solved_square_id = self.solved_squares_to_recognize.pop()
            solved_square_cluster = self.clusters[solved_square_id]
            solved_square_solution = solved_square_cluster[AdjacentMask(False).get_center_negation()]

            if solved_square_solution == MaskSolution().set_as_unitary(1):
                solved_square_cluster.update_status(SquareStatus.MARKED, MineHint.UNDEFINED)
            elif solved_square_solution == MaskSolution().set_as_unitary(0):
                while self.game_board.is_square_uncovered(solved_square_id):
                    self.game_board.uncover_square(solved_square_id)
                    self.game_board.update_board()
                    print("Square hint recog falied:", solved_square_id, "trying again")

                mine_hint = self.game_board.recognize_unc_square_mine_hint(solved_square_id)
                if mine_hint != MineHint.M0:
                    solved_square_cluster.update_status(SquareStatus.HINT_UNSOLVED, mine_hint)
                    self.hint_squares_to_update.add(solved_square_id)
                else:
                    solved_square_cluster.update_status(SquareStatus.HINT_SOLVED, mine_hint)
                    # solved_square_cluster[AdjacentMask(False).get_center_negation()] = MaskSolution().set_as_unitary(0)
                    self.board_fast_recog_blank_area(solved_square_id)

    def board_fast_recog_blank_area(self, start_square_id: UnitSquareID):
        expansion_front = {start_square_id}
        while len(expansion_front) > 0:
            expansion_square_id = expansion_front.pop()
            for rel_exp_sq_id in AdjacentMask(True).get_center_negation().relative_coordinates_generator():
                next_exp_sq_id = expansion_square_id + rel_exp_sq_id
                if not self.clusters.is_inside(next_exp_sq_id):
                    continue
                next_exp_sq_cluster = self.clusters[next_exp_sq_id]
                if next_exp_sq_cluster.square_status != SquareStatus.UNSOLVED:
                    continue
                next_exp_sq_cluster[AdjacentMask(False).get_center_negation()] = MaskSolution().set_as_unitary(0)

                while self.game_board.is_square_uncovered(next_exp_sq_id):
                    self.game_board.uncover_square(next_exp_sq_id)
                    self.game_board.update_board()
                    print("Square hint recog falied:", next_exp_sq_id, "trying again")

                next_exp_sq_hint = self.game_board.recognize_unc_square_mine_hint(next_exp_sq_id)
                if next_exp_sq_hint == MineHint.M0:
                    next_exp_sq_cluster.update_status(SquareStatus.HINT_SOLVED, next_exp_sq_hint)
                    expansion_front.add(next_exp_sq_id)
                else:
                    next_exp_sq_cluster.update_status(SquareStatus.HINT_UNSOLVED, next_exp_sq_hint)
                    self.hint_squares_to_update.add(next_exp_sq_id)


class InitConsole:
    class Initializations(Enum):
        Modules = 0
        GameStatus = 1
        ESClusterContainer_templates = 2
        ESClusterContainer_squares = 3

    output_period = .05

    def __init__(self):
        self.last_output_time = 0

        self.initialization_progress = {}

    def update(self, key_module: InitConsole.Initializations, new_progress: float):
        self.initialization_progress[key_module] = new_progress
        self.output()

    def output(self):
        if time.time() - self.last_output_time > self.output_period:
            self.last_output_time = time.time()

            os.system("cls")
            print("  Initialization progress:")
            for init_type, progress in self.initialization_progress.items():
                print("{}: {}%".format(init_type, int(progress * 100)))


init_console = InitConsole()


def main():
    init_console.update(InitConsole.Initializations.Modules, 1)
    game_window = GameWindow(CFG.ms_id_name)
    game_board = GameBoardInt(game_window)

    cluster_container = SquareClusterContainer(game_board)
    game_solver = GameSolver(game_board, cluster_container)

    start_square = UnitSquareID(int(game_board.board_us_res.x/2), int(game_board.board_us_res.y/2))
    game_board.uncover_square(start_square)
    game_solver.clusters[start_square][AdjacentMask(False).get_center_negation()] = MaskSolution().set_as_unitary(0)
    game_solver.clusters[start_square].update_status(SquareStatus.HINT_SOLVED, MineHint.M0)
    game_solver.board_fast_recog_blank_area(start_square)

    for _ in range(96):
        print("#", _)
        game_solver.board_recognize_new_squares()
        if len(game_solver.hint_squares_to_update) == 0:
            break
        while len(game_solver.hint_squares_to_update) > 0:
            hstu = game_solver.hint_squares_to_update.pop()
            game_solver.hint_square_cluster_update(hstu)

        # for y in range(game_board.board_us_res.y):
        #     line_op = ""
        #     for x in range(game_board.board_us_res.x):
        #         if cluster_container[UnitSquareID(x, y)].square_status in {SquareStatus.HINT_UNSOLVED, SquareStatus.HINT_SOLVED}:
        #             if cluster_container[UnitSquareID(x, y)].center_hint == MineHint.M0:
        #                 line_op += '.'
        #             else:
        #                 line_op += str(cluster_container[UnitSquareID(x, y)].center_hint.get_no_mines())
        #         elif cluster_container[UnitSquareID(x, y)].square_status == SquareStatus.MARKED:
        #             line_op += '*'
        #         else:
        #             line_op += '#'
        #     print(line_op)

main()
