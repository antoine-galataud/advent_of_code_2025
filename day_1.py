from operator import add, sub


def part_2():
    curr_pos = 50
    zero_count = 0

    with open("/tmp/day_1_input.txt") as file:
        for line in file:
            prev_pos = curr_pos
            direction, move_count = add if line[0] == "R" else sub, int(line[1:])
            additional_circles = move_count // 100
            zero_count += additional_circles
            move_count_remaining = move_count % 100

            # apply moves
            curr_pos = direction(curr_pos, move_count_remaining)
            if curr_pos >= 100 or curr_pos <= 0:
                curr_pos %= 100
                if curr_pos != prev_pos and prev_pos != 0:
                    zero_count += 1

    return zero_count


part_2()
