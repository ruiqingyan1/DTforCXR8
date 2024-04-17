def count_lines_in_file(file_path):
    count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            count += 1
    return count


# 使用函数
file_path = 'train_list.txt'  # 替换为你的文件路径
number_of_lines = count_lines_in_file(file_path)
print(f"The train_file has {number_of_lines} lines.")

# 使用函数
file_path = 'test_list.txt'
number_of_lines = count_lines_in_file(file_path)
print(f"The test_file has {number_of_lines} lines.")


# 使用函数
file_path = 'train_list_224.txt'  # 替换为你的文件路径
number_of_lines = count_lines_in_file(file_path)
print(f"The train_file_224 has {number_of_lines} lines.")

# 使用函数
file_path = 'test_list_224.txt'
number_of_lines = count_lines_in_file(file_path)
print(f"The test_file_224 has {number_of_lines} lines.")