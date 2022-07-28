import os
import cv2
import math
from matplotlib import pyplot as plt
from PIL import Image

# TODO: need an import for dm library for dm4 (in convert to jpg function)


def img_param(paths, num_cols, save_crop=False):
    # TODO: think about adding an additional parameter in case one wants to change the
    #  name of the cropped image (PRF).
    # TODO: think about adding an additional parameter in case one wants to change the
    #  saving path of the cropped image (PRF).
    # TODO: Can add parameter in case one doesn't wants to use the paths dictionary
    #  (if this is done, do it for all the other functions as well) (PRF).
    # TODO: Think about implementing the rectangular chips (PRF).
    """ Returns several important image parameters.

        The parameters that are obtained with this functions are: width, height, chip_size,
        num_crops_x, num_crops_y, total_num_crops, pixels_ignored_in_y, pixels_ignored_in_y,
        x-coords, y-coords.

        Parameters:
            paths: (dict)
                Dictionary containing the path to the original image and the path to save
                the cropped image. Keys must necessarily be 'path_to_img' and 'path_to_cropped'
                It is only necessary to provide the path to the cropped image if save_crop=True.

            num_cols: (int)
                This is the number of columns to split the image into.

            save_crop: (Bool)
                Whether you want to save the cropped image
        Returns:
            parameters: (dict)
                A dictionary containing the value of each of the parameters. To call each of
                the parameters just use the names mentioned in the description."""

    # Path for getting the image to extract its parameters
    path_to_img = os.path.join(paths['get_this_img_params'], paths['img_name'])

    # Calculating the image parameters
    # TODO: using image to open
    width, height = Image.open(path_to_img).size
    chip_size = math.floor(width / num_cols)
    num_chips_x = num_cols  # math.floor(width / chip_size)
    num_chips_y = math.floor(height / chip_size)
    pixels_ignored_in_x = width % chip_size
    pixels_ignored_in_y = height % chip_size
    x_coords = list(range(0, width, chip_size))
    y_coords = list(range(0, height, chip_size))
    grid_points = []
    # in the following loops, row_idx and col_idx should be swapped
    for row_idx, x_coord in enumerate(range(0, width - pixels_ignored_in_x, chip_size)):
        for col_idx, y_coord in enumerate(range(0, height - pixels_ignored_in_y, chip_size)):
            grid_points.append((x_coord, y_coord, row_idx, col_idx))

    # Storing the calculated parameters
    parameters = {'width': width,                               # width of the image in pixels
                  'height': height,                             # height of the image in pixels
                  'chip_size': chip_size,                        # chips are squares, size in pixels of one side
                  'num_chips_x': num_chips_x,                   # number of chips along the x direction
                  'num_chips_y': num_chips_y,                   # number of chips along the y direction
                  'total_num_crops': num_chips_x * num_chips_y,  # total number of chips
                  'pixels_ignored_in_x': pixels_ignored_in_x,  # pixels ignored along the x direction
                  'pixels_ignored_in_y': pixels_ignored_in_y,   # pixels ignored along the y direction
                  'x_coords': x_coords,                         # coordinates of the grid points along the x direction
                  'y_coords': y_coords,                         # coordinates of the grid points along the y direction
                  'grid_points': grid_points                    # coordinates of x and y and row and column indices
                  }

    # Saving the cropped image
    if save_crop:
        # Path to store cropped image file
        path_to_cropped = os.path.join(paths['has_cropped_imgs'], paths['img_name'])

        image = Image.open(path_to_img)
        cropped_image = image.crop((0, 0, parameters['width'] - parameters['pixels_ignored_in_x'],
                                    parameters['height'] - parameters['pixels_ignored_in_y']))
        cropped_image.save(path_to_cropped)

    return parameters


def chips_genesis(paths, parameters):

    """ Creates and saves the chips for a given image.

        This function creates the chips for a given image given the
        parameters dictionary returned by the img_param function.

        Parameters:
            paths: (dict)
                Dictionary containing the name of the image, the path to the image
                folder and the path to save the chips. The dict must necessarily contain
                the keys 'path_for_chip_gen', 'path_to_chips' and 'img_name'

            parameters: (dict)
                This is a dictionary that comes from the img_param function and it
                contains the necessary parameters to create the chips.

        Returns:
            No returns."""

    # Path to the image file to be chipped
    path_to_img = os.path.join(paths['chip_this_img'], paths['img_name'])
    # Path to store the chips
    path_to_chips = paths['has_img_chips']

    # Creating and saving the chips
    # image = Image.open(path_to_img)
    image = cv2.imread(path_to_img)
    pixel_count = parameters['chip_size']
    for x, y, row_idx, column_idx in parameters['grid_points']:
        # box = (x, y, x + parameters['chip_size'],
        # y + parameters['chip_size']) #(left, upper, down, right)
        name = f"R{row_idx}C{column_idx}.jpg"  # Name for the chip
        path_for_chip = os.path.join(path_to_chips, name)  # Path for the chip to be stored
        # image.crop(box).save(path_for_chip)
        new_img = image[y: y + pixel_count, x: x + pixel_count, 0]
        plt.imsave(path_for_chip, new_img, vmin=0, vmax=255)
        # the following will save as greyscale. plt applies a viridis colormap by default
        # cv2.imwrite(path_for_chip, new_img)
    return


def convert_to_jpg(paths, save_as=''):

    # TODO: Verify how to optimize and simplify this function even more (PRF).
    # TODO: Fix the .dm4 file extension conversion code (PRF).
    # TODO: Extend the possibilities of file (PRF).

    """Converts images to a .jpg format Currently converts .tiff images only.

        Parameters:
            paths: (dict)
                Dictionary containing the path to the orginal image, the path to where
                the converted images will be saved and the name of the original image.
                Necessary keys: 'has_original_imgs', 'has_converted_imgs' and
                'img_name'.

            save_as: (str)
            New name for the converted image. If not given the name will default to
            the original name of the image. The string must contain the file extension
            .jpg at the end.

        Returns:
            No returns. """

    # Assigning the path to the image file
    img_name = paths['img_name']
    path_to_img = os.path.join(paths['has_original_imgs'], paths['img_name'])

    # Assigning the path for the converted image file
    if save_as:
        path_for_converted = os.path.join(paths['has_converted_imgs'], save_as)
    else:
        path_for_converted = os.path.join(paths['has_converted_imgs'], paths['img_name'])

    if img_name[-4:] == "tiff":
        image = Image.open(path_to_img)     # Opening the original image.
        if image.mode not in ("L", "RGB"):  # rescale 16 bit tiffs to 8 bits
            image.mode = "I"
            image = image.point(lambda i: i * (1.0 / 256))

        new_image = image.convert("RGB")
        new_image.save(path_for_converted)

    # elif img_name[-3:] == "dm4":
        # data = dm.dmReader(path_to_img)['data']
        # new_array = (data - np.min(data)) / (np.max(data) - np.min(data))
        # im = Image.fromarray((255 * new_array).astype('uint8'))
        # im.save(path_for_converted)

    elif img_name[-3:] == 'jpg':
        Image.open(path_to_img).save(path_for_converted)

    else:
        raise Exception('Image file extension is not supported. Currently supported file extensions are .tiff')
    return


def preprocess(paths, prep_type='clahe', save_as='', **kwargs):

    # TODO: Figure out what are the clipLimit, tileGridSize and color_flag parameters (PRF).
    # TODO: Make that the prep_type function is a list in which the user can assign several prep
    #  methods at the same time (PRF).
    # TODO: Do a better implementation of the save_as parameter for the dictionary keys.

    """ For image preprocessing.

        This function performs different types of preprocesisng over an
        image and then saves it to a given folder. Available methods: clahe.

        Parameters:
            paths: (dict)
                Dictionary containing the paths to the converted image and to where
                the clahe processed image will be saved. Keys must necessarily be 'path_to_converted'
                and 'path_to_clahe'.

            prep_type: (str)
                Name of the preprocessing method to use. Available: clahe.

            save_as: (str)
            New name for the preprocessed image. If not given the name will default to
            the original name of the image. The string must contain the file extension.

            **kwargs:
                This argument contains any parameters that belong to the prep_type method selected.

        Returns:
            No returns.
      """

    # Path to get the image to be preprocessed
    path_to_img = os.path.join(paths['preprocess_this_img'], paths['img_name'])

    # Assigning the path for the preprocessed image file
    # if save_as:
    #     path_for_prep = os.path.join(paths['has_prep_imgs'], save_as)
    # else:
    #     path_for_prep = os.path.join(paths['has_prep_imgs'], paths['img_name'])

    if prep_type == 'clahe':
        # Defaults for cv2.createCLAHE were: clipLimit=1.0, tileGridSize=(8,8)

        # Path to save the preprocessed clahe image
        if save_as:
            path_to_clahe = os.path.join(paths['has_clahe_imgs'], save_as)
        else:
            path_to_clahe = os.path.join(paths['has_clahe_imgs'], paths['img_name'])

        # Preprocessing and saving the image
        color_flag = 0
        img = cv2.imread(path_to_img, color_flag)
        clahe = cv2.createCLAHE(**kwargs)
        cl_img = clahe.apply(img)
        cv2.imwrite(path_to_clahe, cl_img)
    return
