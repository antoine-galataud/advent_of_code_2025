import numpy as np

with open("day_10_input.txt") as file:
    machines_instructions = file.read().strip()
    machines_instructions = machines_instructions.split("\n")


def part_1(debug: bool = False):
    sum_min_combinations = 0

    for mi in range(len(machines_instructions)):
        instruction = machines_instructions[mi]

        expected_lights_pattern = instruction.split(" ")[0]
        expected_lights_pattern = [0 if p == "." else 1 for p in expected_lights_pattern[1:-1]]

        toggles = instruction.split(" ")[1:]
        toggles = "".join(toggles).split("{")[0].strip()[1:-1].split(")(")
        toggles_arr = np.zeros((len(toggles), len(expected_lights_pattern)), dtype=int)
        for toggle in range(len(toggles)):
            for toggle_index in list(map(int, toggles[toggle].split(","))):
                toggles_arr[toggle, toggle_index] = 1
        n_toggles = toggles_arr.shape[0]

        if debug:
            print(f"Machine {mi + 1}: expected pattern {expected_lights_pattern}")
            print(f"Toggles: {toggles} (n={n_toggles})")
            print(f"Toggles: {toggles_arr}")

        # given that lights are off at start, we need to find the combination of toggles that will
        # lead to the expected pattern with minimal number of toggles used.
        # here we brute force all combinations of toggles. This is doable since lights patterns are small (<=13 lights)
        # as the total number of combinations is 2^n_toggles so at most 8192 combinations for n_toggles=13.
        min_toggle_count = np.inf
        best_toggles_combination = []

        for toggle_combination in range(1, 2**n_toggles):
            # we generate toggle bits from the integer representation
            # so for instance if n_toggles = 3 and toggle_combination = 5 (binary 101)
            # then toggle_bits = [1,0,1] meaning we apply toggles 0 and 2
            toggle_bits = [(toggle_combination >> i) & 1 for i in range(n_toggles)]
            toggled_pattern = np.zeros(len(expected_lights_pattern), dtype=int)
            toggles_combination = []

            for t in range(n_toggles):
                if toggle_bits[t] == 1:
                    toggled_pattern = (toggled_pattern + toggles_arr[t]) % 2
                    toggles_combination.append(toggles[t])

            if np.array_equal(toggled_pattern, expected_lights_pattern):
                toggle_count = sum(toggle_bits)
                if toggle_count < min_toggle_count:
                    min_toggle_count = toggle_count
                    best_toggles_combination = toggles_combination

        if debug:
            print(
                f"Machine {mi}: best toggle combination {best_toggles_combination} with {min_toggle_count} toggles\n"
            )

        sum_min_combinations += min_toggle_count

    print(f"Part 1: sum of minimal toggle combinations is: {sum_min_combinations}")


def part_2(debug: bool = False):
    """
    DISCLAIMER: this is too slow to solve part 2, despite my efforts to optimize a DFS (sparing you the analysis here).
    It seems correct though, as it solves test cases correctly.

    Fell back to constrained optim libraries to solve the problem (see part2_with_solver).
    """
    sum_min_combinations = 0

    for mi in range(len(machines_instructions)):

        print(f"Processing machine {mi + 1}")

        instruction = machines_instructions[mi]
        toggles_and_joltages = "".join(instruction.split(" ")[1:]).split("{")
        toggles = toggles_and_joltages[0].strip()[1:-1].split(")(")
        expected_joltages = np.array(toggles_and_joltages[1].strip()[:-1].split(",")).astype(int)

        toggles_arr = np.zeros((len(toggles), len(expected_joltages)), dtype=int)
        for toggle in range(len(toggles)):
            for toggle_index in list(map(int, toggles[toggle].split(","))):
                toggles_arr[toggle, toggle_index] = 1

        n_toggles = toggles_arr.shape[0]
        n_joltages = expected_joltages.shape[0]

        if debug:
            print(f"Machine {mi + 1}: expected joltages {expected_joltages} (n={n_joltages})")
            print(f"Toggles: {toggles} (n={n_toggles})")
            print(f"Toggles: {toggles_arr}")

        # the maximum number of presses is upper-bounded by the sum of counters (each press increments one counter)
        max_presses = int(sum(expected_joltages))

        # search space is larger than for part 1, so we can't brute force anymore.
        # we're going to use a depth-first search (DFS) approach to explore all possible combinations of toggle presses.
        # we also use several pruning techniques to return fast when a path is not interesting.
        # disclaimer: this code is largely inspired by source found online.
        def dfs(i, current_vec, nb_presses):
            nonlocal max_presses

            # this path doesn't yield a better result (pruning for optimization).
            if nb_presses >= max_presses:
                return

            # success
            if np.array_equal(current_vec, expected_joltages):
                max_presses = min(max_presses, nb_presses)
                return

            # we tested all buttons
            if i >= n_toggles:
                return

            # compute remaining needed increments
            remaining = expected_joltages - current_vec
            # if impossible (negative anywhere), prune
            if np.any(remaining < 0):
                return

            # maximum times we can press this toggle without exceeding target
            toggle_mask = toggles_arr[i]
            # we can't press it anymore
            if np.all(toggle_mask == 0):
                return

            # Try pressing this toggle k times, with k between 0 and max_times while moving the next toggle.
            # max_times can be either the sum of total remaining presses, or the min of presses the current toggle
            # can still perform. For instance if remaining is [0, 2, 3] and toggle is [1], we can only press this
            # toggle 2 more times.
            total_remaining = int(remaining.sum())
            toggle_remaining = int(
                (remaining[toggle_mask > 0] // toggle_mask[toggle_mask > 0]).min()
            )
            max_times = min(total_remaining, toggle_remaining)
            for k in range(max_times + 1):
                dfs(i + 1, current_vec + k * toggle_mask, nb_presses + k)

        dfs(0, np.zeros(len(expected_joltages)), 0)

        print(f"Optimal number of presses for machine {mi + 1}: {max_presses}")
        sum_min_combinations += max_presses

    print(f"Part 2: sum of minimal toggle combinations is: {sum_min_combinations}")


def part_2_with_solver(solver="milp"):
    assert solver in ["milp", "z3"]

    with open("day_10_input.txt") as f:
        lines = [line.rstrip("\n") for line in f.read().split("\n") if line.strip()]

    grand_total = 0

    for idx, line in enumerate(lines, start=1):
        toggles_and_joltages = "".join(line.split(" ")[1:]).split("{")
        toggles = [
            list(map(int, t.split(","))) for t in toggles_and_joltages[0].strip()[1:-1].split(")(")
        ]
        expected_joltages = list(map(int, toggles_and_joltages[1].strip()[:-1].split(",")))

        if solver == "z3":
            machine_total, presses = solve_with_z3(toggles, expected_joltages)
        else:
            machine_total, presses = solve_with_milp(toggles, expected_joltages)

        grand_total += machine_total

    print(grand_total)


def solve_with_z3(buttons, targets):
    """Solve using the z3-solver library.

    Not a hand-crafted solution (mostly copy-pasted), but works fast.
    """
    from z3 import IntVector, Optimize, Sum, sat

    num_buttons = len(buttons)
    num_counters = len(targets)

    # x[b] = how many times to press button b (non-negative integer)
    x = IntVector("x", num_buttons)
    opt = Optimize()

    # Non-negative constraints
    for b in range(num_buttons):
        opt.add(x[b] >= 0)

    # Counter constraints
    for ci in range(num_counters):
        terms = []
        for b, btn in enumerate(buttons):
            if ci in btn:
                terms.append(x[b])

        if not terms:
            # No button affects this counter -> it must stay at 0
            opt.add(targets[ci] == 0)
        else:
            opt.add(Sum(terms) == targets[ci])

    # Objective: minimize total number of presses
    total_presses = Sum(x)
    opt.minimize(total_presses)

    if opt.check() != sat:
        raise RuntimeError("No solution found for a machine (unsat).")

    model = opt.model()
    presses = [model[x[b]].as_long() for b in range(num_buttons)]
    return sum(presses), presses


def solve_with_milp(toggles, expected_joltages):
    """Solve with scipy as mixed-integer linear programming optimization problem in the form
    min(cx) subject to Ax = b, x >= 0.

    where A is the matrix of toggles and b the machine expected joltages.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    n_counters = len(expected_joltages)
    n_toggles = len(toggles)

    A = np.zeros((n_counters, n_toggles), dtype=int)
    b = np.array(expected_joltages, dtype=int)

    # populate A such as Aij = 1 if button j increases counter i by 1
    # so xj will represent how many presses of button j are needed to reach bj
    for i in range(n_counters):
        for j, btn in enumerate(toggles):
            if i in btn:
                A[i, j] = 1

    # objective: minimize c x
    c = np.ones(n_toggles, dtype=float)

    # x_j integer:
    # integrality[k] = 1 for integer, 0 for continuous
    integrality = np.ones(n_toggles, dtype=int)

    # bounds: x >=0
    bounds = Bounds(lb=np.zeros(n_toggles), ub=np.full(n_toggles, np.inf))

    # linear constraints: A x = b  => lower = upper = b
    constraints = LinearConstraint(A, lb=b, ub=b)

    res = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
    )

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    # due to integrality, res.x should already be integers, but guard with rounding
    x_int = np.rint(res.x).astype(int)
    total_presses = int(x_int.sum())

    return total_presses, x_int


# part_2(False)
part_2_with_solver("milp")
