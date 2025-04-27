import os

def print_directory_tree(startpath, prefix='', max_depth=None, current_depth=0):
    if max_depth is not None and current_depth >= max_depth:
        return
    files = os.listdir(startpath)
    files.sort()
    for index, name in enumerate(files):
        path = os.path.join(startpath, name)
        connector = "└── " if index == len(files) - 1 else "├── "
        print(prefix + connector + name)
        if os.path.isdir(path):
            extension = "    " if index == len(files) - 1 else "│   "
            print_directory_tree(path, prefix + extension, max_depth, current_depth + 1)

if __name__ == "__main__":
    workspace_dir = "."  # you can change this
    max_depth = 3        # set None for no limit, or any integer (e.g., 2, 3, 5)

    print(workspace_dir)
    print_directory_tree(workspace_dir, max_depth=max_depth)
