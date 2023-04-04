from augmentations import get_validation_augmentation, get_preprocessing, get_training_augmentation
from dataset_pr import x_valid_dir, y_valid_dir, x_train_dir, y_train_dir, x_test_dir, y_test_dir
from train import CLASSES, preprocess_input, model, metrics
from main import Dataset, Dataloder


test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    classes=CLASSES,
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)

test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False)

# load best weights
model.load_weights('best_model.h5')

scores = model.evaluate_generator(test_dataloader)

print("Loss: {:.5}".format(scores[0]))
for metric, value in zip(metrics, scores[1:]):
    print("mean {}: {:.5}".format(metric.__name__, value))

    