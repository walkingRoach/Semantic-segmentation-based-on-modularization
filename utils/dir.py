import os


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(path):
        raise print("{} can't mkdir".format(path))


def del_file_in_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        if not os.path.exists(path):
            raise print("{} can't mkdir".format(path))
    else:
        files_list = [f for f in os.listdir(path)]
        if len(files_list) > 0:
            for file in files_list:
                print(file)
                os.remove(os.path.join(path, file))


def recursive_glob(root, suffix):
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(root)
            for filename in filenames if filename.endswith(suffix)]
