from datetime import datetime

import numpy as np

with open("day_8_input.txt") as file:
    grid = file.read().strip()
    grid_rows = grid.split("\n")

coordinates = np.array([list(map(int, row.split(","))) for row in grid_rows])
n_coords = coordinates.shape[0]
print("input data shape:", coordinates.shape)
# x, y, z coordinates of each junction box
assert coordinates.shape[1] == 3


def part_1_and_part_2(num_iters_part_1):
    # compute euclidean distance for each pairs
    # we use broadcasting here so the diff matrix contains, for each coordinate, its distance to all others.
    # e.g. diff[0][1] = distance between coord 0 and coord 1
    diff = coordinates[np.newaxis, :, :] - coordinates[:, np.newaxis, :]  # (n_coords, n_coords, 3)
    dist = np.sqrt(np.sum(diff**2, axis=2))  # (n_coords, n_coords)

    # we only need lower triangular part of the distance matrix
    # (distances are repeated symmetrically on each side of the diagonal)
    mask = np.triu(np.ones(dist.shape))
    mask[mask == 1] = np.inf
    dist = dist + mask

    # done only when all boxes form a single circuit
    done = False
    i = 0
    circuits = []
    while not done:
        # fetch actual coords using argmin indices
        cc_a_idx, cc_b_idx = np.unravel_index(np.argmin(dist), dist.shape)
        cc_a = tuple(coordinates[cc_a_idx])
        cc_b = tuple(coordinates[cc_b_idx])

        # iterate through already connected boxes and try to assign the current one
        # to an existing circuit or a new one
        added_c_idx = None
        for c_idx, coord_pairs in enumerate(circuits):
            # add coords to the set (if they are already there, it's a noop)
            if cc_a in coord_pairs or cc_b in coord_pairs:
                if added_c_idx is None:
                    coord_pairs.add(cc_a)
                    coord_pairs.add(cc_b)
                    added_c_idx = c_idx
                # already added once, means one of the coords belongs to another set that we need to merge
                else:
                    for cc in coord_pairs:
                        circuits[added_c_idx].add(cc)
                    circuits.remove(circuits[c_idx])
                    break

        # not found in an existing circuit, so this is a new one
        if added_c_idx is None:
            circuits.append({cc_a, cc_b})

        # mark coords as processed
        dist[cc_a_idx, cc_b_idx] = np.inf

        # compute circuits lengths
        # ideally, this shouldn't be done on each iteration for performance reasons.
        # here we know that the max iterations is achieved when all boxes are in the same circuit, which should
        # be x * n_coords iterations where x is a small value.
        circuits_idx_len = {i: len(chain) for i, chain in enumerate(circuits)}
        circuits_idx_len = {
            k: v for k, v in sorted(circuits_idx_len.items(), key=lambda item: -item[1])
        }
        lengths = list(circuits_idx_len.values())

        if i == num_iters_part_1 - 1:
            print(
                f"Part 1: product of len of 3 longest circuits after {num_iters_part_1} iterations: {np.prod(lengths[:3])}"
            )
        if lengths[0] == n_coords:
            print(
                f"Part 2: all boxes are in the same circuit of length {lengths[0]} on iteration {i}"
            )
            print(f"Part 2: final coordinates: {cc_a}, {cc_b}")
            print(f"Part 2: x_a * x_b = {cc_a[0] * cc_b[0]}")
            done = True

        i += 1


start_time = datetime.now()
part_1_and_part_2(1000)
end_time = datetime.now()
print(f"Duration: {end_time - start_time}")
