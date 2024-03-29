import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = './data_my_dataset'
# os.startfile(r'.\data_my_dataset\train\0img.png')
# load repo with data if it is not exists
if not os.path.exists(DATA_DIR):
    print('Loading data...')
    os.system('git clone https://github.com/alexgkendall/SegNet-Tutorial ./data')
    print('Done!')

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    # print(images)

    image = images.get('image')
    mask = images.get('mask')

    # print(images)

    # for i, (name, image) in enumerate(images.items()):
    #     plt.subplot(1, n, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.title(' '.join(name.split('_')).title())
    #     plt.imshow(image)

    # print(mask.shape)
    plt.imshow(mask)
    plt.show()


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


# classes for data loading and preprocessing
class Dataset:
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ['detail 1', 'detail 2', 'detail 3', 'detail 4',
               'detail 5']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.mids = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, mask_id) for mask_id in self.mids]

        # convert str names to class values on masks

        # self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.class_values = [14.0, 113.0, 52.0, 89.0, 128.0]
        # print('VALUES', self.class_values)

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    @staticmethod
    def resize(func):
        w = 320
        h = 320

        def res(image, mask):
            sample = func(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            image = cv2.resize(image, (w, h))
            mask = cv2.resize(mask, (w, h))
            return {"image": image, "mask": mask}

        return res

    def __getitem__(self, i):
        w = 1920
        h = 1080
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        image = cv2.resize(image, (w, h))
        image = image[:, :, :4]
        mask = cv2.resize(mask, (w, h))
        # print('MASK', mask)
        # plt.imshow(mask)
        # plt.show()
        # # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # add background if mask is not binary
        # if mask.shape[-1] != 1:
        #     background = 1 - mask.sum(axis=-1, keepdims=True)
        #     mask = np.concatenate((mask, background), axis=-1)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.ids)


class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return batch

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


# Lets look at data we have
dataset = Dataset(x_train_dir, y_train_dir, classes=['detail 1', 'detail 2', 'detail 3', 'detail 4',
               'detail 5'])

image, mask = dataset[40]  # get some sample

# visualize(
#     image=image,
#     mask=mask,
#     # mask=mask,
#     # sky_mask=mask[..., 1].squeeze(),
#     # background_mask=mask[..., 2].squeeze(),
# )
