from datetime import datetime

import numpy as np

with open("day_3_input.txt") as file:
    batteries_banks = file.read().strip()
    batteries_banks = batteries_banks.split("\n")
# print(batteries_banks)


def part_1():
    all_joltages = []
    for bank in batteries_banks:
        bank = list(map(int, bank))
        first_digit_index = np.argmax(bank[:-1])
        second_digit_index = np.argmax(bank[first_digit_index + 1 :]) + first_digit_index + 1
        bank_joltage = bank[first_digit_index] * 10 + bank[second_digit_index]
        all_joltages.append(bank_joltage)
    print(sum(all_joltages))


def part_2():
    all_joltages = []
    for bank in batteries_banks:
        bank = list(map(int, bank))
        joltage_digits = []
        cur_pos = 0
        while len(joltage_digits) != 12:
            window = bank[cur_pos : len(bank) - (12 - len(joltage_digits)) + 1]
            max_digit_index = np.argmax(window)
            joltage_digits.append(bank[max_digit_index + cur_pos])
            cur_pos = cur_pos + max_digit_index + 1
        bank_joltage = int("".join(map(str, joltage_digits)))
        all_joltages.append(bank_joltage)
    print(sum(all_joltages))


start_time = datetime.now()
part_2()
end_time = datetime.now()
print(f"Duration: {end_time - start_time}")
