import pandas as pd
import numpy as np

from LoadData import one_hot


def read_image_data_text(path: str, image_width: int, image_height: int, delimiter_whitespace: bool = True,
                         image_size: int = 95):
    all_images = pd.read_csv(path, delim_whitespace=delimiter_whitespace, header=None, dtype=int)

    columns_count = int(all_images.shape[1] / image_width)
    rows_count = int(all_images.shape[0] / image_height)
    train_images = []
    train_classes = []
    test_images = []
    test_classes = []
    pixel_sum = np.zeros((image_height, image_width))
    for i in range(rows_count):
        for j in range(columns_count):
            pixel_sum = pixel_sum +\
                        np.array(all_images.loc[i*image_height: (i+1)*image_height-1, j*image_width: (j+1)*image_width-1])

    pixel_sum = (pixel_sum - np.amin(pixel_sum)) / (np.amax(pixel_sum) - np.amin(pixel_sum))
    columns_to_remove = []
    j = -1
    k = 0
    for i in range(image_width - image_size):
        j = j+1
        if abs(pixel_sum[0, j] - np.mean(pixel_sum[:, j])) < 0.01:
            columns_to_remove.append(j)
            continue
        k = k - 1
        if abs(pixel_sum[0, k] - np.mean(pixel_sum[:, k])) < 0.01:
            columns_to_remove.append(k)
            continue

    rows_to_remove = []
    j = -1
    k = 0
    for i in range(image_height - image_size):
        j = j+1
        if abs(pixel_sum[j, 0] - np.mean(pixel_sum[j, :])) < 0.01:
            rows_to_remove.append(j)
            continue
        k = k - 1
        if abs(pixel_sum[k, 0] - np.mean(pixel_sum[k, :])) < 0.01:
            rows_to_remove.append(k)
            continue




    for i in range(rows_count):
        for j in range(columns_count-2):
            img = all_images.loc[i*image_height: (i+1)*image_height-1, j*image_width: (j+1)*image_width-1].to_numpy()
            img1 = np.array(np.delete(img, rows_to_remove, 0))
            img2 = np.delete(img1, columns_to_remove, 1)
            train_images.append(img2)
            train_classes.append(i+1)
    for i in range(rows_count):
        for j in range(columns_count-2, columns_count):
            img = all_images.loc[i*image_height: (i+1)*image_height-1, j*image_width: (j+1)*image_width-1].to_numpy()
            img1 = np.array(np.delete(img, rows_to_remove, 0))
            img2 = np.array(np.delete(img1, columns_to_remove, 1))
            test_images.append(img2)
            test_classes.append(i+1)

    train_df = pd.DataFrame(data={'image': train_images, 'class': train_classes})
    test_df = pd.DataFrame(data={'image': test_images, 'class': test_classes})

    train_df = train_df.sample(frac=1, ignore_index=True)
    test_df = test_df.sample(frac=1, ignore_index=True)

    train_input = train_df['image']
    train_output = one_hot(train_df['class'])
    test_input = test_df['image']
    test_output = one_hot(test_df['class'])

    return train_input, train_output, test_input, test_output
#
# def reduce_all_images_size(images: pd.DataFrame):
#     for i in range(images.shape[0]):
#         for j in range(images.shape[1]):
#             reduce_image_size(images[i, j])
#
# def reduce_image_size(image: pd.DataFrame):
#
