import os

def get_files_info(folder):
    files_info = {}
    for root, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            relative_path = os.path.relpath(file_path, folder)
            files_info[relative_path] = file_size
    return files_info

def compare_folders(folder1, folder2):
    files_info1 = get_files_info(folder1)
    files_info2 = get_files_info(folder2)
    
    only_in_folder1 = set(files_info1.keys()) - set(files_info2.keys())
    only_in_folder2 = set(files_info2.keys()) - set(files_info1.keys())
    
    different_sizes = {
        file: (files_info1[file], files_info2[file]) 
        for file in set(files_info1.keys()) & set(files_info2.keys()) 
        if files_info1[file] != files_info2[file]
    }
    
    if not only_in_folder1 and not only_in_folder2 and not different_sizes:
        return "The folders have the same files in terms of number, names, and sizes."
    else:
        result = []
        if only_in_folder1:
            result.append("Files only in folder1:")
            result.extend(only_in_folder1)
        if only_in_folder2:
            result.append("Files only in folder2:")
            result.extend(only_in_folder2)
        if different_sizes:
            result.append("Files with different sizes:")
            for file, sizes in different_sizes.items():
                result.append(f"{file}: {sizes[0]} (folder1) vs {sizes[1]} (folder2)")
        return "\n".join(result)

# Usage
folder2 = '/home/ubuntu/ml-1cc/data/mlperf/bert_bk/download/results4'
folder1 = '/home/ubuntu/ml-1cc/data/mlperf/bert/download/results4'

result = compare_folders(folder1, folder2)
print(result)
