import numpy as np

with open("day_9_input.txt") as file:
    grid = file.read().strip()
    grid_rows = grid.split("\n")

coordinates = np.array([list(map(int, row.split(","))) for row in grid_rows])
print("input data shape:", coordinates.shape)
n_coords = coordinates.shape[0]
# x, y coordinates of each point on the grid
assert coordinates.shape[1] == 2


def part_1():
    # compute surface between each pairs of points, adding 1 to account for line width
    diff = coordinates[np.newaxis, :] - coordinates[:, np.newaxis]  # (n_coords, n_coords, 2)
    surface = (np.abs(diff[:, :, 0]) + 1) * (np.abs(diff[:, :, 1]) + 1)  # (n_coords, n_coords)
    max_surface = np.max(surface)
    print("Part 1: max surface is", max_surface)


def part_2():
    # /!\ not a general solution, only works for the given input data.
    # visualizing the data, the shape is a circular hull with a big horizontal cut in the middle.
    # we can split the points in two sets: those above the cut and those below the cut.
    # then we can compute the max surface for each set and take the max of both.
    x_coords = coordinates[:, 0]
    x_diff = np.abs(x_coords[np.arange(n_coords)] - x_coords[np.arange(n_coords) - 1])
    cut_index_upper = np.argmax(x_diff)

    # /!\ turns out the solution is found in the upper part of the hull for the given input.
    # but this should have been done for both parts.
    upper_coords = coordinates[np.where(coordinates[:, 1] >= coordinates[cut_index_upper, 1])]
    # cut the upper coords into 2 parts (left and right of the half circle)
    upper_coords_1 = upper_coords[upper_coords.shape[0] // 2 :]
    upper_coords_2 = upper_coords[: upper_coords.shape[0] // 2 + 1]

    # we iterate over all possible x coordinates between min and max x of the cut line
    # and try to form a rectangle with the cut line as one side
    min_x = np.min(upper_coords[:, 0])
    max_x = coordinates[cut_index_upper][0]
    max_surface = 0
    max_surface_points = None
    for x in range(min_x, max_x):
        point_a = [x, coordinates[cut_index_upper][1]]
        point_b = upper_coords_1[np.where(upper_coords_1[:, 0] == x)]
        if point_b.size == 0:
            continue
        point_b = point_b[0].tolist()
        point_c = [coordinates[cut_index_upper][0], coordinates[cut_index_upper][1]]
        point_d = [coordinates[cut_index_upper][0], point_b[1]]

        # point_d should be left of the closest point in upper_coords_2 with same y as point_b
        min_dist = np.inf
        closest_pc = None
        for pc in upper_coords_2:
            dist = np.linalg.norm(np.array(point_d) - np.array(pc))
            if dist < min_dist:
                min_dist = dist
                closest_pc = pc

        if closest_pc is None or closest_pc[0] <= point_d[0]:
            continue

        # now compute surface
        width = abs(point_a[0] - point_c[0]) + 1
        height = abs(point_b[1] - point_c[1]) + 1
        surface = width * height

        if surface > max_surface:
            max_surface = surface
            max_surface_points = (point_a, point_b, point_c, point_d)

    print("Part 2: max surface is", max_surface, "between points", max_surface_points)


part_2()
