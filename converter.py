import labelme2coco
import torch
# import torchvision

labelme_folder = r"C:\Users\Liika\Desktop\ion\labelme"

export_dir = r"C:\Users\Liika\Desktop\ion\coco"

train_split_rate = 0.85

labelme2coco.convert(labelme_folder, export_dir)



