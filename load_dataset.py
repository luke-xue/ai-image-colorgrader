import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import cv2


class MapDataset(Dataset):
    def __init__(self, pre_dir, post_dir):
        self.pre_dir = pre_dir
        self.post_dir = post_dir
        self.pre_list_files = os.listdir(self.pre_dir)
        self.post_list_files = os.listdir(self.post_dir)

    def __len__(self):
        return len(self.pre_list_files)

    def __getitem__(self, index):
        img_file = self.pre_list_files[index]
        pre_img_path = os.path.join(self.pre_dir, img_file)
        img_file_without_extension, _ = os.path.splitext(img_file)
        post_img_file = img_file_without_extension + ".jpg" 
        post_img_path = os.path.join(self.post_dir, post_img_file)
        img_pre = cv2.imread(pre_img_path)
        if img_pre is None:
            print(pre_img_path)
            print("img_pre is None")
        res_pre = cv2.resize(img_pre, dsize=(600, 600), interpolation=cv2.INTER_CUBIC)
        img_post = cv2.imread(post_img_path)
        if img_post is None:
            print("img_post is None")
        res_post = cv2.resize(img_post, dsize=(600, 600), interpolation=cv2.INTER_CUBIC)
        input_image = np.array(res_pre)
        target_image = np.array(res_post)

        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image


if __name__ == "__main__":
    dataset = MapDataset("data/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()
