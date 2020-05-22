from datasets.dataset_aug.tusimple import SeqLaneAug
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='augment datasets')
    parser.add_argument('-s', '--src', default='train.txt', type=str,
                        help='the config used to make sure augment dataset')
    parser.add_argument('-d', '--des', type=str,
                        help='Path to img path to save')
    parser.add_argument('-f', '--file', type=str,
                        help='file name for train txt')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    seqLaneAug = SeqLaneAug(args.src, args.des, args.file)
    seqLaneAug.create_train(step=[1, 2, 3, 4, 5])
    # seqLaneAug.split_dataset()
    # for i in range(len(seqLaneAug.files)):
    #     seqLaneAug.augment(i)


if __name__ == '__main__':
    main()
