import numpy as np
from augmentations import get_validation_augmentation, get_preprocessing, get_training_augmentation
from dataset_pr import x_valid_dir, y_valid_dir, x_train_dir, y_train_dir
from evaluation import test_dataset
from train import model
from main import Dataset, Dataloder, visualize, denormalize

n = 5
ids = np.random.choice(np.arange(len(test_dataset)), size=n)

for i in ids:
    image, gt_mask = test_dataset[i]
    image = np.expand_dims(image, axis=0)
    pr_mask = model.predict(image).round()

    visualize(
        image=denormalize(image.squeeze()),
        gt_mask=gt_mask[..., 0].squeeze(),
        pr_mask=pr_mask[..., 0].squeeze(),
    )
