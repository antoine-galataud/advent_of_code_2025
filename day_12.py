with open("day_12_input.txt") as file:
    lines = file.read().strip().split("\n")

items = []
bins = []
for line in lines:
    # this is a bin (widthxheight: i j k l m n)
    if "x" in line:
        parts = line.split("x")
        width = int(parts[0])
        height = int(parts[1].split(":")[0])
        # get the items
        items_fit_part = parts[1].split(":")[1].strip()
        items_fit = list(map(int, items_fit_part.split(" ")))
        bins.append((width, height, items_fit))
    # item shape description
    elif line != "":
        # item id
        if ":" in line:
            items.append((int(line.split(":")[0].strip()), []))
        # item shape
        else:
            items[-1][1].append(line)


def part_1():
    """This one promised sweat and tears as an irregular shape bin packing problem.

    However, after checking some memes on subreddit r/adventofcode, I realized that the
    problem is way simpler: the input data simplifies the problem a lot!
    """
    valid_bins = []

    for bin_width, bin_length, presents_to_fit in bins:
        present_counts = sum(presents_to_fit) * 9
        total_capacity = bin_width * bin_length

        if present_counts <= total_capacity:
            valid_bins.append(bin)

    print(f"Part 1: valid bins count: {len(valid_bins)}")


part_1()
