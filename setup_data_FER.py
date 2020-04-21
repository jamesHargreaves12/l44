# Modified from code in FERPlus Challenge

import os
import csv
import numpy as np
from itertools import islice
from PIL import Image

folder_names = {'Training': 'FER2013Train',
                'PublicTest': 'FER2013Valid',
                'PrivateTest': 'FER2013Test'}


def str_to_image(image_blob):
    image_string = image_blob.split(' ')
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48, 48)
    return Image.fromarray(image_data)


def main(ouput_folder, fer_path):
    for key, value in folder_names.items():
        folder_path = os.path.join(ouput_folder, value)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    index = 0
    with open(fer_path, 'r') as csvfile:
        fer_rows = csv.reader(csvfile, delimiter=',')
        for i,row in enumerate(islice(fer_rows, 1, None)):
            file_name = "{}.png".format(i)
            if len(file_name) > 0:
                image = str_to_image(row[1])
                image_path = os.path.join(ouput_folder, folder_names[row[2]], file_name)
                image.save(image_path, compress_level=0)
            index += 1

    print("Done...")


if __name__ == "__main__":
    main("data/FER", "FER2013/fer2013/fer2013.csv")