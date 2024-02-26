import pandas as pd
import shutil
from pathlib import Path
import albumentations as A
from PIL import Image
import cv2

FOLDERS_TO_LABELS = {"food": "food", "non_food": "non_food"}


def get_files_and_labels(source_path, only_aug_flag=False):
    images = []
    labels = []
    for image_path in source_path.rglob("*/*.jpg"):
        filename = image_path.absolute()
        folder = image_path.parent.name
        folder_parent = image_path.parent.parent.name
        folder_parent_parent = image_path.parent.parent.parent.name

        if folder in FOLDERS_TO_LABELS:
            if folder_parent_parent == "val" or folder_parent == "val":
                images.append(filename)
                label = FOLDERS_TO_LABELS[folder]
                labels.append(label)
            elif folder_parent_parent == "train" or folder_parent == "train":
                if not only_aug_flag:
                    images.append(filename)
                    label = FOLDERS_TO_LABELS[folder]
                    labels.append(label)
                else:
                    if folder_parent == "aug":
                        images.append(filename)
                        label = FOLDERS_TO_LABELS[folder]
                        labels.append(label)

    return images, labels

def save_as_csv(filenames, labels, destination, shuffle_flag):
    data_dictionary = {"filename": filenames, "label": labels}
    data_frame = pd.DataFrame(data_dictionary)

    if shuffle_flag:
        data_frame = data_frame.sample(frac=1).reset_index(drop=True)

    data_frame.to_csv(destination)

def aug(image_path, output_image_path):
    transform = A.Compose([
        A.HorizontalFlip(p=1),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4, p=0.4),
        A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=2, p=0.3),
    ])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transformed = transform(image=image)
    transformed_image = transformed["image"]

    im = Image.fromarray(transformed_image)
    im.save(output_image_path)

def prepare_aug(train_path, train_files, train_labels, number_of_aug_of_one_class):
    # --------------- creating folders /aug/food and aug/non_food ------------
    aug_path = train_path / "aug"

    if aug_path.exists():
        shutil.rmtree(aug_path)

    food_aug_path = aug_path / "food"
    non_food_aug_path = aug_path / "non_food"

    food_aug_path.mkdir(parents=True, exist_ok=True)
    non_food_aug_path.mkdir(parents=True, exist_ok=True)
    # --------------- creating folders /aug/food and aug/non_food ------------

    food_counter = 0
    non_food_counter = 0
    for i in range(len(train_files)):
        image_label = train_labels[i]

        if image_label == "food" and food_counter < number_of_aug_of_one_class:
            food_counter += 1
            image_name = str(train_files[i])

            substr = "/food/"
            insertstr = "/aug"
            idx = image_name.index(substr)
            aug_image_name = image_name[:idx] + insertstr + image_name[idx:] + "_aug.jpg"
            
            aug(image_name, aug_image_name)
        elif image_label == "non_food" and non_food_counter < number_of_aug_of_one_class:
            non_food_counter += 1
            image_name = str(train_files[i])

            substr = "/non_food/"
            insertstr = "/aug"
            idx = image_name.index(substr)
            aug_image_name = image_name[:idx] + insertstr + image_name[idx:] + "_aug.jpg"
            
            aug(image_name, aug_image_name)

        if food_counter >= number_of_aug_of_one_class and non_food_counter >= number_of_aug_of_one_class:
            break


def main(repo_path):
    data_path = repo_path / "data"
    prepared_path = data_path / "prepared"

    train_path = data_path / "raw/train"
    test_path = data_path / "raw/val"

    train_files, train_labels = get_files_and_labels(train_path)
    test_files, test_labels = get_files_and_labels(test_path)

    # # ------------------------------------- aug -------------------------------------
    # prepare_aug(train_path, train_files, train_labels, 500)
    # train_files, train_labels = get_files_and_labels(train_path)
    # # ------------------------------------- aug -------------------------------------

    save_as_csv(train_files, train_labels, prepared_path / "train.csv", True)
    save_as_csv(test_files, test_labels, prepared_path / "test.csv", False)


if __name__ == "__main__":
    repo_path = Path(__file__).parent.parent
    main(repo_path)
