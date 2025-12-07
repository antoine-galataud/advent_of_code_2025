from datetime import datetime

with open("day_2_input.txt") as file:
    data = file.read().strip()
    ranges = data.split(",")


def part_1():
    # sorry, didn't keep it
    pass


def part_2():
    invalid_ids = []
    for r in ranges:
        start, end = map(int, r.split("-"))
        # print(f"Range from {start} to {end}")
        for num in range(start, end + 1):
            # an ID is invalid if it is made only of some sequence of digits repeated at least twice
            str_num = str(num)
            length = len(str_num)
            for sub_len in range(1, length // 2 + 1):
                if length % sub_len == 0:
                    sub_str = str_num[:sub_len]
                    if sub_str * (length // sub_len) == str_num:
                        invalid_ids.append(str_num)
                        break

    print("nb invalid ids", sum(map(int, invalid_ids)))


start_t = datetime.now()
part_2()
end_t = datetime.now()
print(f"Duration: {end_t - start_t}")
