import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DATA_DIR = './data_my_dataset'

if not os.path.exists(DATA_DIR):
    print('no datadir')

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')

x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')
