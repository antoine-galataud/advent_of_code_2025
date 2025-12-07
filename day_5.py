from datetime import datetime

with open("day_5_input.txt") as file:
    all_lines = file.read().strip()
    lines = all_lines.split("\n")


def part_1():
    fresh_ingredient_ranges = set()
    all_ingredient_ids = set()
    for line in lines:
        if "-" in line:
            start, end = map(int, line.split("-"))
            fresh_ingredient_ranges.add((start, end))
        elif line.isdigit():
            all_ingredient_ids.add(int(line))

    fresh_ingredients = set()
    for ing_id in all_ingredient_ids:
        for start, end in fresh_ingredient_ranges:
            if start <= ing_id <= end:
                fresh_ingredients.add(ing_id)
                break

    print(len(fresh_ingredients))


def part_2():
    fresh_ingredient_ranges = set()
    for line in lines:
        if "-" in line:
            start, end = map(int, line.split("-"))
            fresh_ingredient_ranges.add((start, end))

    # we sort the ranges so that we can merge overlapping ones sequentially,
    # only considering the current range and the last merged one
    fresh_ingredient_ranges = sorted(fresh_ingredient_ranges)

    # merge overlapping ranges
    merged_ranges = []
    for start, end in fresh_ingredient_ranges:
        # if start of next range is before or equal end of last merged range, then merge
        if merged_ranges and start <= merged_ranges[-1][1]:
            merged_ranges[-1] = (
                # start of last merged range
                merged_ranges[-1][0],
                # max between end of last merged range and end of current range
                max(merged_ranges[-1][1], end),
            )
        else:
            merged_ranges.append((start, end))

    total_nb_ingredients = 0
    for start, end in merged_ranges:
        total_nb_ingredients += end - start + 1

    print(total_nb_ingredients)


start_time = datetime.now()
part_2()
end_time = datetime.now()
print(f"Duration: {end_time - start_time}")
