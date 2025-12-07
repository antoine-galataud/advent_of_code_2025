from datetime import datetime

import numpy as np

with open("day_4_input.txt") as file:
    grid = file.read().strip()
    grid_rows = grid.split("\n")

grid_array = np.array([list(map(str, row)) for row in grid_rows])
n_rows, n_cols = grid_array.shape
print(n_rows, n_cols)
# print(grid_array[:2, :])


def part_1():
    total_picked = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if grid_array[i, j] == "@":
                adjacent_positions = grid_array[
                    max(0, i - 1) : min(n_rows, i + 2), max(0, j - 1) : min(n_cols, j + 2)
                ]
                count_adjacent_rolls = np.sum(adjacent_positions == "@") - 1
                if count_adjacent_rolls < 4:
                    total_picked += 1
    print(total_picked)


def part_2():
    total_picked = 0
    round_picked = 0
    print("total nb rolls before:", np.sum(grid_array == "@"))
    while round_picked != 0 or total_picked == 0:
        round_picked = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if grid_array[i, j] == "@":
                    adjacent_positions = grid_array[
                        max(0, i - 1) : min(n_rows, i + 2), max(0, j - 1) : min(n_cols, j + 2)
                    ]
                    # count adjacent rolls ('x' are removable but still there), excluding self
                    count_adjacent_rolls = (
                        np.sum(adjacent_positions == "@") + np.sum(adjacent_positions == "x") - 1
                    )
                    if count_adjacent_rolls < 4:
                        round_picked += 1
                        # mark as removable
                        grid_array[i, j] = "x"
        # remove all marked as 'x' at end of round
        grid_array[grid_array == "x"] = "."
        total_picked += round_picked

    print("total picked", total_picked)
    print("total nb rolls after:", np.sum(grid_array == "@"))


start_time = datetime.now()
# part_1()
part_2()
end_time = datetime.now()
print(f"Duration: {end_time - start_time}")
