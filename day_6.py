from datetime import datetime

import numpy as np


def part_1():
    with open("day_6_input.txt") as file:
        all_lines = file.read().strip()
        lines = all_lines.split("\n")

    arrs = []
    for i, line in enumerate(lines[:-1]):
        arr = list(map(int, filter(lambda x: x != "", line.strip().split(" "))))
        arrs.append(arr)

    operands = np.array(arrs)
    print(operands.shape)
    print(operands.dtype)

    operations = list(filter(lambda x: x != "", lines[-1].strip().split(" ")))
    assert operands.shape[1] == len(operations)

    results = []
    for i, operands in enumerate(operands.T):
        operation = operations[i]
        result = 0
        if operation == "*":
            result = np.prod(operands)
        elif operation == "+":
            result = np.sum(operands)
        results.append(result)

    print(sum(results))


def part_2():
    # loading. We need to account for spaces
    with open("day_6_input.txt") as file:
        all_lines = file.read()
        lines = all_lines.split("\n")

    # operands can be split by columns containing only spaces
    all_chars = []
    for i, line in enumerate(lines[:-1]):
        all_chars.append([c for c in line])

    all_chars = np.array(all_chars)
    start_operand_idx = 0
    all_operands = []
    for c_idx in range(all_chars.shape[1]):
        if np.all(all_chars[:, c_idx] == " "):
            all_operands.append(all_chars[:, start_operand_idx:c_idx])
            start_operand_idx = c_idx + 1
    # add the last one
    all_operands.append(all_chars[:, start_operand_idx : c_idx + 1])

    # load operations (* or +)
    operations = list(filter(lambda x: x != "", lines[-1].strip().split(" ")))
    assert len(operations) == len(all_operands)

    # compute result for each operation
    results = []
    for o_idx, operands in enumerate(all_operands):
        col_operands = []
        operands = np.array(operands)
        for j in range(len(operands[0]) - 1, -1, -1):
            operand = int(
                "".join([operands[i, j] for i in range(len(operands)) if operands[i, j] != " "])
            )
            col_operands.append(operand)

        # apply operation
        operation = operations[o_idx]
        results.append(np.prod(col_operands) if operation == "*" else np.sum(col_operands))

    print(sum(results))


start_time = datetime.now()
part_2()
end_time = datetime.now()
print(f"Duration: {end_time - start_time}")
