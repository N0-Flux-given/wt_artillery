import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import json

# Update these by changing to your own folder locations

# This contains the location of the folder that has the map images
MAPS_FOLDER = "D:/Stuff/data_science/tf_course/pytorch/wt_data/maps/"
# Location of the folder containing tank symbols
SYMBOLS_FOLDER = "D:/Stuff/data_science/tf_course/pytorch/wt_data/tank_symbols/"
# The destination output folder. Blended images will be saved here along with the .xml files
DESTINATION_FOLDER = "D:/Stuff/data_science/tf_course/pytorch/wt_data/processed/"

# How many symbols of a particular tank type do you want to put on the map
SYMBOLS_OF_TYPE_ON_MAP = 15
SHOW_IMAGE = True

IMG_SIZE = 834


def get_folder_filename_path(full_path):
    """Gets folder filename path

    Parameters
    ----------
    full_path : str
        FUll path of the root

    Returns
    -------
    tuple (folder, file_name, full_path)
        Returns the folder, file name and the full path
    """
    components = full_path.split("/")
    folder = components[-2]
    file_name = components[-1]
    full_path = ""
    for component in components:
        full_path += component + "\\"
    full_path = full_path[:-1]

    return folder, file_name, full_path


def blend_alpha(img_b, img_f):
    """Blends alpha channel (transparent portion) of the tank symbols with the background image of the map.

    Parameters
    ----------
    img_b : np.ndarray
        The background image array, with RGB channels
    img_f : np.ndarray
        The tank symbol image with RGBA channels.

    Returns
    -------
    np.ndarray
        Final image with img_f overlaid on the background img_b.
    """
    fg_alpha = img_f[:, :, 3] / 255.0

    for color in range(0, 3):
        img_b[:, :, color] = fg_alpha * img_f[:, :, color] + img_b[:, :, color] * (
            1 - fg_alpha
        )

    return img_b


def place_symbol_on_map(map_img, tank_symbol, tank_type):
    """Places a tank symbol on the map and generates the corresponding <object> tag for it in the XML

    Parameters
    ----------
    map_img : np.ndarray
        Image of the map with RGB channels.
    tank_symbol : np.ndarray
        Image of the symbol of the tank. RGB or RGBA channels.
    tank_type : str
        Class name of the tank. Eg: td -> tank destroyer, ht -> heavy tank

    Returns
    -------
    tuple (map_img, obj_str)
        Tuple containing the map image array and the XML object string with the bounding box info of the
        symbol placed on the map.
    """
    available_x_values = map_img.shape[1] - tank_symbol.shape[1]
    available_y_values = map_img.shape[0] - tank_symbol.shape[0]

    x_random = np.random.randint(available_x_values)
    y_random = np.random.randint(available_y_values)

    x_min = x_random
    y_min = y_random
    x_max = x_random + tank_symbol.shape[1]
    y_max = y_random + tank_symbol.shape[0]

    bound_box = {
        "name": tank_type[:-4],
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
    }

    bg_patch = map_img[
        y_random : y_random + tank_symbol.shape[0],
        x_random : x_random + tank_symbol.shape[1],
        :,
    ]

    # modify map_img by placing the symbol on it
    try:
        # Try to blend alpha chennel if it exists.
        map_img[
            y_random : y_random + tank_symbol.shape[0],
            x_random : x_random + tank_symbol.shape[1],
            :,
        ] = blend_alpha(bg_patch, tank_symbol)
    except ValueError:
        # If the symbol image does not have an alpha channel, then just overlay it directly.
        map_img[
            y_random : y_random + tank_symbol.shape[0],
            x_random : x_random + tank_symbol.shape[1],
            :,
        ] = tank_symbol[:, :, :3]

    # cv2.rectangle(map_img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
    return map_img, bound_box


def show_image(img, label):
    cv2.imshow(label, img)
    cv2.waitKey(0)


def main():
    tank_symbol_images = []

    # Read symbols and store them in a list
    for symbol_file in os.listdir(SYMBOLS_FOLDER):
        symbol_img = cv2.imread(SYMBOLS_FOLDER + symbol_file, cv2.IMREAD_UNCHANGED)
        tank_symbol_images.append((symbol_img, symbol_file))
        show_image(symbol_img, "tank_symbol")

    # Read map images from the maps folder
    for map_file in os.listdir(MAPS_FOLDER):
        map_img = cv2.imread(MAPS_FOLDER + map_file, cv2.IMREAD_UNCHANGED)
        map_img = cv2.resize(map_img, (IMG_SIZE, IMG_SIZE))

        box_list = []

        # REad each symbol and randomlg place it SYMBOLS_OF_TYPE_ON_MAP times on the map
        for tank_symbol_img, tank_type_name in tank_symbol_images:
            for _ in range(SYMBOLS_OF_TYPE_ON_MAP):
                map_img, bound_box = place_symbol_on_map(
                    map_img=map_img,
                    tank_symbol=tank_symbol_img,
                    tank_type=tank_type_name,
                )
                box_list.append(bound_box)

        show_image(map_img, "map wil all symbols")

        # Write the image and xml file to the destination folder
        cv2.imwrite(DESTINATION_FOLDER + map_file, map_img)
        location_json = json.dumps({'data' : box_list})

        with open(f"{DESTINATION_FOLDER}{map_file[:-4]}_loc.json", "w") as json_file:
            json.dump(location_json, json_file)


main()
