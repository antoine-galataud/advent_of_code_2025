from functools import cache

with open("day_11_input.txt") as file:
    all_inst = file.read().strip()
    all_inst = all_inst.split("\n")

in_out = {}
for line in all_inst:
    i_o = line.split(": ")
    i = i_o[0]
    if i in in_out:
        raise ValueError(f"Duplicate input found: {i}")
    o = i_o[1].strip().split(" ")
    in_out[i] = o


def part_1():
    # This is a directed acyclic graph (DAG).
    # To find every path from "you" to "out", we can use a depth-first search (DFS) approach (thank you day 10).
    # Update: after solving part 2, I realized that caching the count of paths from each node to 'out' is way more efficient.

    @cache
    def count(current_node) -> int:
        """Count paths from current_node to 'out' passing through all nodes in must_do set.

        caching is key and is possible because it's a DAG, so no cycles: once we computed
        the count for a given node and its subpaths, we won't need to recompute it.
        """
        if current_node == "out":
            return 1
        else:
            return sum(count(child) for child in in_out[current_node])

    all_counts = count("you")
    print(f"Part 1: total paths from 'you' to 'out': {all_counts}")


def part_2():
    """
    Quote: "They now know that the problematic data path passes through both dac (a digital-to-analog converter)
    and fft (a device which performs a fast Fourier transform)".
    Lovely!

    This time we must be sure all paths go through both 'dac' and 'fft', in any order.
    """

    @cache
    def count(current_node, must_do) -> int:
        """Count paths from current_node to 'out' passing through all nodes in must_do set.

        caching is key and is possible because it's a DAG, so no cycles: once we computed
        the count for a given node and its subpaths, we won't need to recompute it.
        """
        must_do = must_do - {current_node}
        if current_node == "out":
            return 1 if len(must_do) == 0 else 0
        else:
            return sum(count(child, must_do) for child in in_out[current_node])

    all_counts = count(current_node="svr", must_do=frozenset({"dac", "fft"}))
    print(
        f"Part 2: total paths from 'svr' to 'out' passing through both 'dac' and 'fft': {all_counts}"
    )


part_1()
part_2()
