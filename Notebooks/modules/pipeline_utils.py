import os
import shutil
import time
import numpy as np
import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.models as models
from skimage import color
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torch.nn import Softmax
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize

# TODO Fix this messy import (PRF).
if __name__ == '__main__':
    # from paths import paths_genesis, read_structure
    from unmodified_models.protonet import PrototypicalNet
    # from unmodified_models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
else:
    from modules.paths import paths_genesis
    from modules.unmodified_models.protonet import PrototypicalNet
    # from modules.unmodified_models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


def build_skeleton(structure, main_path='', data_path='', support_path='', **kwargs):
    """ A function created to save the user some time when deciding
        which structure to use.

        It has pre-maid folder structures for the user to select. In addition,
        it creates the paths dictionary and assigns the desired paths to each
        key.

        Parameters:
            structure: (str)
                The desired structure to use. These are already pre-programmed
                inside this function and the user only has to select which one
                he wants to use. Available values are S1 and S2.

            main_path: (str)
                The path where the structure is going to be created. Default
                value is an empty str but one str should be given.

            data_path: (str)
                The path that will be assigned for the 'has_original_imgs' key
                of the paths dict. This might or might not be a necessary value
                depending on the selected structure. Default value is empty str.

            support_path: (str)
                The path that will be assigned for the 'has_labels_folders' key
                of the paths dict. This might or might not be a necessary value
                depending on the selected structure. Default value is empty str.

            **kwargs:
                Contains any additional parameters that a given structure might
                be programmed to use.

        Returns:
            paths: (dict)
                Contains the necessary keys and their paths
                for the used pipeline.

            folders: (dict)
                Contains the paths for the created folder structure.

            structure: (dict)
                Contains the used folder structure.
            """

    if not main_path:
        raise ValueError('main_path parameter cannot be empty. Please provide a path to place your structure.')

    if structure == 'S1':
        # S1 is a preliminary structure, but it has space for improvement

        # Designing the folder structure
        skeleton = {'username': {'images': [{'all_chips': ['all_chips']},
                                            'converted',
                                            'cropped',
                                            {'preprocessed': ['clahe']}],
                                 'results': None,
                                 'support': None}}

        # Creating the folders
        folders = paths_genesis(skeleton, main_path=main_path)

        # Assigning the desired paths for the desired keys
        paths = {'has_converted_imgs': folders['username_images_converted'],
                 'preprocess_this_img': folders['username_images_converted'],
                 'has_clahe_imgs': folders['username_images_preprocessed_clahe'],
                 'get_this_img_params': folders['username_images_preprocessed_clahe'],
                 'has_cropped_imgs': folders['username_images_cropped'],
                 'chip_this_img': folders['username_images_cropped'],
                 'has_img_chips': folders['username_images_all_chips_all_chips'],
                 'has_results': folders['username_results']}

        # Assigning path for original images
        if data_path:
            # Using a given one
            paths['has_original_imgs'] = data_path
        else:
            # Creating a folder within the S1 structure
            paths['has_original_imgs'] = paths_genesis({'originals': None}, main_path=folders['username_images'])[
                'originals']

        # Assigning path for the support set
        if support_path:
            # Using a given one
            paths['has_labels_folders'] = support_path
        else:
            # Creating a folder within the S1 structure
            paths['has_labels_folders'] = paths_genesis({'support': None}, main_path=folders['username_images'])[
                'support']

    elif structure == 'S2':
        # To use this structure a support_path and a data_path must be given

        # Catching errors
        if 'folder_name' not in kwargs.keys():
            raise Exception('Need to provide a folder_name value.')
        if not data_path:
            raise ValueError('data_path parameter cannot be empty.')
        # if not support_path:
        #     raise ValueError('support_path parameter cannot be empty.')

        # Designing the folder structure
        name = kwargs['folder_name']
        skeleton = {name: {'all_chips': ['all_chips'],
                           'imgs_and_results': None,
                           'support': None}}

        # Creating the folders
        folders = paths_genesis(skeleton, main_path=main_path)

        # Assigning the desired paths for the desired keys
        dummy_name = name + '_imgs_and_results'

        paths = {'has_converted_imgs': folders[dummy_name], 'preprocess_this_img': folders[dummy_name],
                 'has_clahe_imgs': folders[dummy_name], 'get_this_img_params': folders[dummy_name],
                 'has_cropped_imgs': folders[dummy_name], 'chip_this_img': folders[dummy_name],
                 'has_img_chips': folders[name + '_all_chips_all_chips'], 'has_results': folders[dummy_name],
                 'has_original_imgs': data_path, 'has_support_set_folders': folders[name + '_support']}
        # paths['has_labels_folders'] = folders[]
        # Assigning path for the support set
        if support_path:
            # Using a given one
            paths['has_labels_folders'] = support_path

    else:
        raise ValueError('Wrong value given for the structure parameter.')

    return paths, folders, skeleton


def support_genesis(paths, support_dict=None, folder_name=''):

    """ Creates a support set.

        This function can create new support set folders at
        a given location given the desired chips per folder
        containing in a form of a dictionary.

        Parameters:
            paths: (dict)
                Dictionary containing the paths to the main support folder
                (the folder which contain the support folders) and the path
                containing the desired chips. Necessary keys:
                has_support_set_folders' and 'has_img_chips'.

            support_dict: (dict)
                A dictionary representing the labels and their chips. They
                keys must be the name of the label and the values a list
                with the name (including file extension) of the chips that
                want to be place inside the label.

            folder_name: (str)
                The name that is going to be given to the new support set
                folder containing all the labels.
        """

    # Path to the folder containing all the support set folders
    if support_dict is None:
        support_dict = {}
    path_to_support = paths['has_support_set_folders']
    # Path to the folder containing the chips
    path_to_chips = paths['has_img_chips']

    # Catching errors
    if not isinstance(support_dict, dict):
        raise TypeError('support_dict must be dict type.')
    if not support_dict:
        raise Exception('The support_dict provided is empty. Please provide a valid one.')

    # Creating the new support sets folders and filling them with the desired chips.
    for key in support_dict:
        label_folder_path = os.path.join(path_to_support, key)
        os.mkdir(label_folder_path)

        # Filling a support set folder with the desired chips.
        for chip_name in support_dict[key]:
            if chip_name[-3:] != 'jpg':
                chip_name = chip_name + '.jpg'
            chip_path = os.path.join(path_to_chips, chip_name)
            destination_path = os.path.join(label_folder_path, chip_name)
            shutil.copyfile(chip_path, destination_path)


def color_image(paths, results, parameters, ret=False, use_cropped='False', save_as=''):
    """ Creates a colored label-map version of the image.

        This function takes in the results of the prediction and
        then colors the image sections according to how the chips
        where classified.

        Parameters:
            paths: (dict)
                A dictionary containing the necessary path for the functions
                to work. It must contain the keys 'chip_this_img', 'img_name'
                'has_results' and any other necessary key if using the
                copy_format parameter.

            results: (pandas dataframe)
                The pandas dataframe that contains the prediction for
                each of the chips. It is

            parameters: (dict)
                A dictionary containing the image parameters. It comes from
                the img_param() function.

            ret: (bool)
                Default is False. If True the function returns the
                numpy array that contains the label of each of the pixels
                of the image.

            use_cropped: (bool)
                Default is False. If True the function adjusts the color labels
                to fit the cropped image.

            save_as: (str)
                the file name you would like to save color image with

        Returns:
            if ret=True:
                few_labels: (numpy array)
                    A numpy array which contains the label for each of the
                    pixels in the image.
    """

    # TODO: Add an option to change the name of the colored image (PRF)

    # Path for image file
    path_to_img = os.path.join(paths['chip_this_img'], paths['img_name'])
    # Path to store the colored (labeled) image
    if save_as:
        path_to_colored_image = os.path.join(paths['has_results'], save_as)
    else:
        path_to_colored_image = os.path.join(paths['has_results'], 'colored_' + paths['img_name'])

    # TODO: make the mapping work without any order (PRF).
    # Dictionary identifying the predicted labels with integers. Also they are in order.
    mapping = {label: idx for idx, label in enumerate(sorted(set(results.prediction)))}

    # Creating a zeros numpy array to contain the label for each of the pixels.

    if use_cropped:
        width = parameters['width'] - parameters['pixels_ignored_in_x']
        height = parameters['height'] - parameters['pixels_ignored_in_y']
    else:
        width = parameters['width']
        height = parameters['height']

    color_labels = np.zeros((height, width))

    # Assigning colored label sections to the image
    chip_size = parameters['chip_size']
    for x_coord, y_coord, row_idx, col_idx in parameters['grid_points']:
        results_idx = f'R{row_idx}C{col_idx}'
        color_labels[y_coord: y_coord + chip_size, x_coord: x_coord + chip_size] = mapping[
            results.loc[results_idx, "prediction"]]

    # TODO: Add a legend to the image (PRF).
    img = cv2.imread(path_to_img)
    plt.imsave(path_to_colored_image,
               color.label2rgb(color_labels, img, colors=["blue", "red", "green", "yellow", "purple", "pink"],
                               kind='overlay'))
    if ret:
        return color_labels


def predictions(paths, encoder='resnet101', ret=False, seed=42, save_results_as=''):
    # TODO: make a txt file of all the details :encoder name, model name, support images, compute time
    # TODO: Document the parameters seed and save_results_as

    """Uses the few-shot model to predict the class of the chips.

        Parameters:
            paths: (dict)
                Dictionary containing the paths to the support set folder, the chips
                folder and the results folder. Keys must necessarily be 'path_to_support',
                path_to_chips' and 'path_to_result'.


            encoder: (str)
                Selection of encoder. Available encoders are: resnet18,
                resnet34, resnet50, resnet101 and resnet152.

            ret: (bool)
                Default is False. If True the function returns a pandas dataframe
                that contains the prediction for each of the chips and the probabilities
                to belong to each class.

            seed: (int)
                a random number used to set the seed - has no effect on accuracy

            save_results_as: (str)
                the name you would like to use to save the results csv - including file extension
        Returns:
            if ret=True:
                results: (pandas dataframe)
                    A pandas dataframe containing the prediction for each of the chips
                    and the probability they have of belonging to each class.

            """

    # Path to the folder containing the labels folders
    if 'has_labels_folders' in paths.keys():
        path_to_support = paths['has_labels_folders']
    else:
        path_to_support = paths['has_labels_folders']

    # try:
        # path_to_support = paths['has_labels_folders']
    # except:
        # path_to_support = paths['has_support_set_folders']

    # Path to the folder containing the folder containing the chips. (this sentence is not an error.)
    path_to_chips = os.path.dirname(paths['has_img_chips'])
    # Path to where the csv is going to be stored
    path_to_results = paths['has_results']

    # Selecting the encoder
    if encoder == 'resnet18':
        enkoder = models.resnet18(pretrained=True)
    elif encoder == 'resnet34':
        enkoder = models.resnet34(pretrained=True)
    elif encoder == 'resnet50':
        enkoder = models.resnet50(pretrained=True)
    elif encoder == 'resnet101':
        enkoder = models.resnet101(pretrained=True)
    elif encoder == 'resnet152':
        enkoder = models.resnet152(pretrained=True)
    else:
        raise ValueError(
            'Encoder given not available. Available encoders are: resnet18, resnet34, resnet50, resnet101, resnet152.')

    model = PrototypicalNet(encoder=enkoder, device='cpu')

    # Predicting the labels for chips and timing the prediction.
    start_time = time.time()
    results = predict(model, path_to_support, path_to_chips, max_query_size=200, seed=seed)
    total_time = time.time() - start_time
    print(total_time)

    # Saving the results in a csv file
    if save_results_as:
        results.to_csv(os.path.join(path_to_results, save_results_as))
    else:
        results.to_csv(os.path.join(path_to_results, 'results.csv'))

    if ret:
        return results


# Was inside the old .mod_data.py file.
class ShotWayDataset(Dataset):
    def __init__(self, datapath, transform=None, target_transform=None, **kwargs):
        """
        An abstraction of PyTorch's Dataset class designed for fewshot learning.

        Parameters:
            datapath:
            transform:
            target_transform:
        """

        super(ShotWayDataset, self).__init__()

        # There seems to be a particular precaution one has to take with the folder structures for the
        # ImageFolder() function to work.
        self.data = ImageFolder(root=datapath, transform=ToTensor())

        self.labels = self.data.targets
        self.transform = transform
        self.target_transform = target_transform
        self.kwargs = kwargs

    def __getitem__(self, idx):
        item = self.data[idx][0]
        label = self.data[idx][1]
        filename = self.data.imgs[idx][0]
        if self.transform:
            item = self.transform(item, **self.kwargs)
        return item, label, filename

    def __len__(self):
        return len(self.data)


# Was inside the old predict.py file.
def predict(model, path_to_support, path_to_chips, max_query_size=20, seed=42):
    # TODO: Optimize this function (PRF).

    """
        Parameters:
            model:
                a (pretrained) model to be used for inference
            path_to_support: str
                Path to a support set directory
            path_to_chips: str
                Path to the query set directory
            max_query_size: int
                Maximum number of query points to predict at a time
            seed: (int)
                a random number - has no effect on accuracy
        Returns:
            results: Probabilities """

    torch.manual_seed(seed)
    norm = Softmax(dim=1)
    output_probabilities = []
    fnames = []
    # load the data

    support_set = ShotWayDataset(path_to_support, transform=resize, size=(255, 255))
    stacked_support_set = {}

    # for i, x in enumerate(support_set):
    #     print("support_set len(x):", len(x))
    #     print("support_set x[0]:", len(x[0]))
    #     print("support_set x[1]:", x[1])

    # print("support_set.labels:", support_set.labels)
    for s_k in set(support_set.labels):
        # get class inds
        stacked_support_set[s_k] = torch.stack([x[0] for i, x in enumerate(support_set) if x[1] == s_k])
    # print("stacked_support_set:", stacked_support_set)

    query_set = ShotWayDataset(path_to_chips, transform=resize, size=(255, 255))
    query_loader = DataLoader(query_set, batch_size=max_query_size)
    # construct a batches to pass to the model
    # support_set is a dictionary of stacked tensors, query_set is still a Dataset object
    # batch = {'support_examples': support_set, 'query_examples': query_set}
    # model needs to switch to eval mode
    model.eval()
    with torch.no_grad():
        for i, q in enumerate(query_loader):
            # q[0] is a stack of image data q[1] can be ignored
            print("Computing batch %s" % str(i))
            output = model(stacked_support_set, q[0])
            probabilities = norm(output)
            output_probabilities.append(probabilities)
            fnames.append(q[2])

    # Turning the results into a dataframe
    results = pd.DataFrame(tensor_i.tolist() for sublist in output_probabilities for tensor_i in sublist)

    # Assigning label names to the columns
    results.columns = support_set.data.classes

    # Formatting filepaths for each chip
    flat_list = [os.path.basename(item).split('.')[0] for sublist in fnames for item in sublist]

    # Assigning the index row, column position in the convention of R{row_idx}C{col_idx} to the dataframe.
    results.index = flat_list
    # results.index = [os.path.basename(s[0]).split('.')[0] for s in query_loader.dataset.data.samples]

    # Creating the prediction column by comparing the max value between the labels.
    results['prediction'] = results.idxmax(axis=1)
    # TODO:make sure to flip any predictions of the support set that aren't consistent with the support set label
    return results
