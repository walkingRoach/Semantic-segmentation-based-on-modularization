import os
import random
import shutil
import re


class BaseAug:
    def __init__(self, file_path, save_root, file_name):
        self.file_path = file_path
        self.save_root = save_root
        self.files = self.get_files(self.file_path)
        train_file = os.path.join(self.save_root, file_name)
        # test_file = os.path.join(self.save_root, 'final_test.txt')
        print('create train_file {}'.format(train_file))
        self.train_f = open(train_file, 'w')
        # self.test_f = open(test_file, 'w')

    def get_files(self, file_path):
        train_file = os.path.join(file_path, "final_test.txt")
        if not os.path.exists(train_file):
            raise print('can not find file')
        files = self._read_txt(train_file)
        return files

    def _read_txt(self, file_path):
        img_list = []
        label_list = []
        with open(file_path, 'r') as fs:
            while True:
                liens = fs.readline()
                if not liens:
                    break
                item = liens.strip().split()
                if not item:
                    continue
                img_list.append(item[0])
                label_list.append(item[1])

        fs.close()
        return list(zip(img_list, label_list))

    def augment(self):
        for i in range(len(self.files)):
            self.augment_by_index(i)

    def augment_by_index(self, index):
        raise NotImplementedError

    def create_dir(self, path):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    def aug_files(self):
        raise NotImplementedError

    def update_file(self, img_path, label_path, is_train=False):
        line = img_path + ' ' + label_path + '\n'
        if is_train:
            self.train_f.write(line)
        else:
            self.test_f.write(line)

    def split_dataset(self):
        # dataset_files = self.get_files(file_path)
        random.shuffle(self.files)

        split_index = int(len(self.files) * 0.8)
        val_indexes = self.files[split_index:]
        train_indexes = self.files[:split_index]
        self.move_file(train_indexes, True)
        self.move_file(val_indexes, False)

        self.train_f.close()
        # self.test_f.close()

    def move_file(self, dataset_list, is_train=False):
        # 排序
        dataset_list = sorted(dataset_list, key=lambda x: (int(str(x[0]).split('/')[-3].replace('-', '').replace('3', '0')),
                              int(str(x[0]).split('/')[-2].replace('_', ''))))
        print('dataset len is {}'.format(len(dataset_list)))

        for i in range(len(dataset_list)):
            img_path, label_path = dataset_list[i]
            if not os.path.exists(os.path.dirname(img_path)):
                raise print('error')
            self.update_file(img_path, label_path, is_train)

            source_path = os.path.dirname(str(img_path))
            if is_train and 'test' in source_path:
                save_path = str(source_path).replace('test', 'train')
                # self.create_dir()
                print("from {} to {}".format(source_path, save_path))
                shutil.move(source_path, save_path)

                save_path = str(label_path).replace('test', 'train')
                self.create_dir(save_path)
                shutil.move(label_path, save_path)
            elif not is_train and 'train' in source_path:
                save_path = str(source_path).replace('train', 'test')
                print("from {} to {}".format(source_path, save_path))
                shutil.move(source_path, save_path)

                save_path = str(label_path).replace('train', 'test')
                self.create_dir(save_path)
                shutil.move(label_path, save_path)
            else:
                continue
