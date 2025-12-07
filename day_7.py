from datetime import datetime

import numpy as np

with open("day_7_input.txt") as file:
    grid = file.read().strip()
    grid_rows = grid.split("\n")

grid_array = np.array([list(map(str, row)) for row in grid_rows])
n_rows, n_cols = grid_array.shape
print(f"Grid shape: {(n_rows, n_cols)}")


def save_to_file(grd):
    with open("/tmp/day_7_output.txt", "w") as o:
        for i in range(n_rows):
            o.write("".join(grd[i, :].tolist()))
            o.write("\n")


def mark_beams_paths():
    """Descent the grid from top row to down and mark the beam trajectory."""
    for i in range(0, n_rows - 1):
        for j in range(0, n_cols):
            # on first row
            if grid_array[i, j] == "S":
                grid_array[i + 1, j] = "|"

            if grid_array[i, j] == "|":
                # cell on next row is a splitter
                if grid_array[i + 1, j] == "^":
                    grid_array[i + 1, j - 1] = "|"
                    grid_array[i + 1, j + 1] = "|"
                # continue beam
                if grid_array[i + 1, j] == ".":
                    grid_array[i + 1, j] = "|"


def part_1():
    # count how many times the beam is split
    # it corresponds to the number of times a splitter has a beam input and beams outputs
    count = 0
    for i in range(0, n_rows - 1):
        for j in range(0, n_cols):
            if grid_array[i, j] == "^" and grid_array[i - 1, j] == "|":
                count += 1

    print(count)


def part_2():
    # counting all possible paths in the quantum tachyon manifold.
    # starting from the top, we count, at each level, the number of possible paths to reach each '|' cell
    # then propagate downstream.
    paths = np.zeros(grid_array.shape, dtype=int)
    for i in range(0, n_rows):
        for j in range(0, n_cols):
            if grid_array[i, j] == "|":
                # start of beam
                if i == 1:
                    paths[i, j] = 1
                # this is the continuation of a beam, simply propagate
                if grid_array[i - 1, j] == "|":
                    paths[i, j] = paths[i - 1, j]
                # this comes from a split, we accumulate
                if grid_array[i, j - 1] == "^":
                    paths[i, j] += paths[i - 1, j - 1]
                if j < n_cols - 1 and grid_array[i, j + 1] == "^":
                    paths[i, j] += paths[i - 1, j + 1]
    # we sum all path counts at the bottom
    paths_count = np.sum(paths[paths.shape[0] - 1, :])
    print(paths_count)


mark_beams_paths()
# for visualization
save_to_file(grid_array)

# part_1()
start_time = datetime.now()
part_2()
end_time = datetime.now()
print(f"Duration: {end_time - start_time}")
