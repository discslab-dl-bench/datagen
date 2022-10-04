from starter_code.utils import load_case
import numpy as np
import cv2
import json


def HorizontalFlip(input, type):
    return cv2.flip(input, 1)


def VerticalFlip(input, type):
    return cv2.flip(input, 0)


def GaussianBlurring(input, type):
    if type == "img":
        return cv2.GaussianBlur(input, ksize=(5, 5), sigmaX=1, sigmaY=1)
    return input


def Sharpening(input, type):
    if type == "imge":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(input, -1, kernel)
    return input

methods = {
    "HorizontalFlip":  HorizontalFlip,
    "VerticalFlip": VerticalFlip,
    "GaussianBlurring": GaussianBlurring,
    "Sharpening": Sharpening
}

if __name__ == "__main__":

    # result = dict()
    # for i in range(210):
    #     case = str(i)
    #     while len(case) < 3:
    #         case = "0" + case
    #     image, label = load_case(case)
    #     image_spacings = image.header["pixdim"][1:4].tolist()
    #     # print(image_spacings)
    #     result[case] = image_spacings
    #     json.dump( result, open( "image_spacings.json", 'w' ) )

    Method = "VerticalFlip"

    for i in range(210):
        case = str(i)
        while len(case) < 3:
            case = "0" + case

        print("case00"+ case)
        # volume, segmentation = load_case(case)
        # imgs = volume.get_fdata()
        # masks = segmentation.get_fdata()
        imgs = np.load(f"/raid/data/imseg/preproc-data/case_00{case}_x.npy")
        masks = np.load(f"/raid/data/imseg/preproc-data/case_00{case}_y.npy")
        imgs_new = np.array([methods[Method](img, "img") for img in imgs])
        masks_new = np.array([methods[Method](mask, "mask") for mask in masks])
        np.save(f"/raid/data/unet/augmentation/data_{Method}/case_00{case}_x.npy", imgs_new)
        np.save(f"/raid/data/unet/augmentation/data_{Method}/case_00{case}_y.npy", masks_new)
