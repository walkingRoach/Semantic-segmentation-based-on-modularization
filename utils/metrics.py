import numpy as np
import torch
import cv2


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)  # 每个元素相乘
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        # print(self.avg, type(self.avg))
        return np.round(self.avg, 5)  # 返回小数点后５位


def batch_pix_accuracy(output, target):
    _, predict = torch.max(output, 1)

    predict = predict.int() + 1
    target = target.int() + 1

    pixel_labeled = (target > 0).sum()
    pixel_correct = ((predict == target)*(target > 0)).sum()
    # print(pixel_correct, pixel_labeled)
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct.cpu().numpy(), pixel_labeled.cpu().numpy()


def batch_intersection_union(output, target, num_class):
    _, predict = torch.max(output, 1)
    predict = predict + 1
    target = target + 1

    predict = predict * (target > 0).long()
    intersection = predict * (predict == target).long()

    area_inter = torch.histc(intersection.float(), bins=num_class, max=num_class, min=1)
    area_pred = torch.histc(predict.float(), bins=num_class, max=num_class, min=1)
    area_lab = torch.histc(target.float(), bins=num_class, max=num_class, min=1)
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter.cpu().numpy(), area_union.cpu().numpy()


# get accuracy
def eval_metrics(output, target, num_classes, is_train=True):
    correct, labeled = batch_pix_accuracy(output.data, target)
    inter, union = batch_intersection_union(output.data, target, num_classes)
    if not is_train:
        precision, recall, f1, error = get_f1(output.data, target)
        return [np.round(correct, 5), np.round(labeled, 5), np.round(inter, 5), np.round(union, 5),
                np.round(precision, 5), np.round(recall, 5), np.round(f1, 5), error]
    else:
        return [np.round(correct, 5), np.round(labeled, 5), np.round(inter, 5), np.round(union, 5)]


def pixel_accuracy(output, target):
    output = np.asarray(output)
    target = np.asarray(target)
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((output == target) * (target > 0))
    return pixel_correct, pixel_labeled


def inter_over_union(output, target, num_class):
    output = np.asarray(output) + 1
    target = np.asarray(target) + 1
    output = output * (target > 0)

    intersection = output * (output == target)
    area_inter, _ = np.histogram(intersection, bins=num_class, range=(1, num_class))
    area_pred, _ = np.histogram(output, bins=num_class, range=(1, num_class))
    area_lab, _ = np.histogram(target, bins=num_class, range=(1, num_class))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union


def get_f1(output, target):
    _, predict = torch.max(output, 1)
    img = torch.squeeze(predict).cpu().numpy()*255
    lab = torch.squeeze(target).cpu().numpy()*255
    img = img.astype(np.uint8)
    lab = lab.astype(np.uint8)
    kernel = np.uint8(np.ones((3, 3)))

    label_precision = cv2.dilate(lab, kernel)
    pred_recall = cv2.dilate(img, kernel)
    img = img.astype(np.int32)
    lab = lab.astype(np.int32)
    label_precision = label_precision.astype(np.int32)
    pred_recall = pred_recall.astype(np.int32)
    a = len(np.nonzero((img * label_precision)[1]))
    b = len(np.nonzero(img)[1])
    error = 0
    precision = 0
    recall = 0
    f1 = 0
    if b == 0:
        error += 1
    else:
        precision = float(a/b)

    c = len(np.nonzero(pred_recall*lab)[1])
    d = len(np.nonzero(lab)[1])

    if d == 0:
        error += 1
    else:
        recall = float(c / d)

    f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1, error

