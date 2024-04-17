import os
import cv2
import h5py
import numpy

Random_Crop = 32
Patch_size = 32
label_size = 20
conv_side = 6
scale = 2


def prepare_data(lr_path, hr_path):
    lr_names = os.listdir(lr_path)
    lr_names = sorted(lr_names)
    nums = lr_names.__len__()

    data = numpy.zeros((nums, Patch_size, Patch_size), dtype=numpy.double)
    label = numpy.zeros((nums, label_size, label_size), dtype=numpy.double)


    for i in range(nums):
        lr_name = os.path.join(lr_path, lr_names[i])
        hr_name = os.path.join(hr_path, lr_names[i])

        if not os.path.isfile(lr_name):
            print(f"Error: file not found: {lr_name}")
            continue

        lr_img = cv2.imread(lr_name, cv2.IMREAD_COLOR)
        if lr_img is None or lr_img.size == 0:
            print(f"Error: image is empty: {lr_name}")
            continue

        hr_img = cv2.imread(hr_name, cv2.IMREAD_COLOR)
        if hr_img is None or hr_img.size == 0:
            print(f"Error: image is empty: {hr_name}")
            continue

        lr_img = cv2.cvtColor(cv2.resize(lr_img, (Patch_size, Patch_size)), cv2.COLOR_BGR2YCrCb)
        hr_img = cv2.cvtColor(cv2.resize(hr_img, (label_size, label_size)), cv2.COLOR_BGR2YCrCb)

        lr_img = lr_img[:, :, 0]
        hr_img = hr_img[:, :, 0]


        lr_img = lr_img.astype(float) / 255.
        hr_img = hr_img.astype(float) / 255.

        data[i, :, :] = lr_img
        label[i, :, :] = hr_img

    return data, label
