import cv2
import json
import numpy as np
import os

root = '/home/ouquanlin/datasets/tusimple/train_set/'
file = open('/home/ouquanlin/datasets/tusimple/train_set/label_data_0601.json', 'r')
train_txt = "/home/ouquanlin/datasets/tusimple/train_set/test.txt"
fp = open(train_txt, 'a')

image_num = 0
for line in file.readlines():
    data = json.loads(line)
    img_path = root + data['raw_file']
    save_path = str(img_path).split('clips')[-1][:-7]
    image = cv2.imread(img_path)
    binaryimage = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
    # instanceimage = binaryimage.copy()
    arr_width = data['lanes']
    arr_height = data['h_samples']
    width_num = len(arr_width)
    height_num = len(arr_height)
    for i in range(height_num):
        # lane_hist = 0
        for j in range(width_num):
            if arr_width[j][i - 1] > 0 and arr_width[j][i] > 0:
                binaryimage[int(arr_height[i]), int(arr_width[j][i])] = 1
                # instanceimage[int(arr_height[i]), int(arr_width[j][i])] = lane_hist
                if i > 0:
                    cv2.line(binaryimage, (int(arr_width[j][i - 1]), int(arr_height[i - 1])),
                             (int(arr_width[j][i]), int(arr_height[i])), 1, 10)
            #         cv2.line(instanceimage, (int(arr_width[j][i - 1]), int(arr_height[i - 1])),
            #                  (int(arr_width[j][i]), int(arr_height[i])), lane_hist, 10)
            # lane_hist += 1
    string1 = "truth" + save_path + ".png"
    fp.write(img_path.split(root)[-1] + ' ' + string1)
    fp.write('\n')
    print(string1)
    if not os.path.exists(os.path.dirname(string1)):
        os.makedirs(os.path.dirname(string1))
    # string2 = "gt_image_instance/" + str(image_num) + ".png"
    # string3 = "image/" + str(image_num) + ".png"
    cv2.imwrite(string1, binaryimage)
    # cv2.imwrite(string2, instanceimage)
    # cv2.imwrite(string3, image)
    image_num = image_num + 1
file.close()
fp.close()
