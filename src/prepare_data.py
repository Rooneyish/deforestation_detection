from sklearn.model_selection import train_test_split
import os
import shutil

source_dir = "data/EuroSAT_RGB"
target_dir = "data/processed_data"

target_classes = ['AnnualCrop', 'Forest', 
                'HerbaceousVegetation', 'Highway', 
                'Industrial', 'Pasture', 
                'PermanentCrop', 'Residential', 
                'River', 'SeaLake']


def prepare_dataset():
    shutil.rmtree(target_dir, ignore_errors=True)

    for split in ['train', 'val', 'test']:
        for classes in target_classes:
            os.makedirs(os.path.join(target_dir, split, classes), exist_ok=True)

    for class_name in target_classes:
        src_path = os.path.join(source_dir, class_name)
        images = os.listdir(src_path)
        train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

        def copy_images(image_list, split):
            for img in image_list:
                shutil.copy(os.path.join(src_path, img), os.path.join(target_dir, split, class_name))

        copy_images(train_imgs, 'train')
        copy_images(val_imgs, 'val')
        copy_images(test_imgs, 'test')

if __name__ == "__main__":
    prepare_dataset()