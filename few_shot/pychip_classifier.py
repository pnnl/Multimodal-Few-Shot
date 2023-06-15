import sys
import os
from PIL import Image
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from skimage import color
import time
import pandas as pd
import torch
from torch.nn import Softmax, Identity
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
from torchvision.transforms.functional import resize
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from .protonet import PrototypicalNet, MultimodalPrototypicalNet
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .custom_colors import create_overlay
import torchvision.models as models
import pretrained_microscopy_models as pmm
from custom_colors import save_stats_plot
from torch import nn


""" It seems cv2 is faster than PIL (https://towardsdatascience.com/image-processing-opencv-vs-pil-a26e9923cdf3)
and cv2 can work with live videos which will be useful in the future. This is the reason for the selection of 
 the cv2 module over the PIL module. But, pytorch works with PIL hence that's why there is a need for conversion 
 from cv2 to PIL around the code. however, cv2 apparently is difficult to install on the machine, so we might want to 
 use PIL instead."""


class PychipClassifier:
    """This class is responsible for containing all the methods that are used
    within the pipeline of the few-shot predictions as well as several important
    attributes of the image and in a future, image metadata."""

    def __init__(self, data_path, img_name, **kwargs):
        path_to_img = os.path.join(data_path, img_name)
        # Image metadata
        self.directory = data_path
        self.img_name = img_name
        if "image" in kwargs:
            print("an image was passed in directly")
            pil_image = kwargs["image"]
            self.img_og = np.asarray(pil_image)
            # the torch models we use require the input to have 3 channels. if there is only 1, we make more:
            if len(self.img_og.shape) == 2:
                for_torch = np.zeros((self.img_og.shape[0], self.img_og.shape[1], 3))
                for_torch[:, :, 0] = self.img_og  # same value in each channel
                for_torch[:, :, 1] = self.img_og
                for_torch[:, :, 2] = self.img_og
                self.img = for_torch
            else:
                self.img = np.asarray(pil_image)
        if "eds" in kwargs:
            if kwargs["eds"] is not None:
                # bring in spectra and save as self.eds
                assert isinstance(kwargs["eds"], np.ndarray)
                self.eds = kwargs["eds"]
            else:
                self.eds = None
        # else:
        # TODO: figure out img vs img_og. currently img is being used for most things, but it is not the
        #   greyscale image
        # Image read as grayscale
        self.img_og = cv2.imread(path_to_img, 0)
        # This is the attribute that will change as methods are being applied,
        # so the img_cv2 is left untouched
        # self.img is in color
        self.img = cv2.imread(path_to_img)

    # TODO: Make it work with the 3 channels img cv2 (PRF)
    # TODO: Perhaps try implement the albumentations preprocessing module since it seems it has a lot to add (PRF)
    def preprocess(self, prep_type, savepath="", img_name="", **kwargs):
        """This method is the responsible on for preprocessing the image.

        The method takes in a string that indicates the type of preprocessing
        to the image. If the savepath and img_name parameters are given then
        the method will save the preprocessed image in the given savepath with
        the given img_name. The kwargs are for any method that needs them.

        Parameters:
            prep_type: (str) (required)
                Determines which type of preprocessing method is going to be
                applied to the image.

            savepath: (str) (optional)
                The directory path to where the preprocessed image will
                be saved.

            img_name: (str) (required with savepath)
                This parameter is the name for the preprocessed image
                to be saved. If savepath is given then a name for the
                image must be provided as well.

            **kwargs: (any) (as required)
                This parameter is passed to any preprocessing method
                that needs it.

        Returns:
            self: (object)
                It is returned to be able to implement the method
                cascading."""

        if prep_type == "CLAHE":
            if "clipLimit" in kwargs:
                clip_limit = kwargs["clipLimit"]
            else:
                print(
                    "clipLimit not provided for CLAHE preprocess - using default value: 1.0"
                )
                clip_limit = 1.0
            if "tileGridSize" in kwargs:
                tile_grid_size = kwargs["tileGridSize"]
            else:
                print(
                    "tileGridSize not provided for CLAHE preprocess - using default value: (8,8)"
                )
                tile_grid_size = (8, 8)
            clahe = cv2.createCLAHE(clip_limit, tile_grid_size)
            # gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            # using og because it is greyscale (read in with color_flag=0)

            # for color images (will need to be read in differently) (untested)
            if not isinstance(self.img.flat[0], np.uint8):
                print("cv2 requires different datatype, converting to uint8")
                self.img = self.img.astype(np.uint8)
            hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv_img)
            v = clahe.apply(v)
            hsv = cv2.merge([h, s, v])
            self.img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # self.img = clahe.apply(self.img)
            print(self.img.shape)

            # saving the preprocessed image if a non empty string passed to savepath
            if savepath == "":
                return self
            else:
                print("saving preprocessed image")
                save_path = os.path.join(savepath, img_name)
                img = np.array(self.img)
                cv2.imwrite(save_path, img)

        return self

    def parameters(self, num_cols, crop=True):
        """Extracts some important parameters of the image.

        This method extracts several important parameters of the image and
        stores the in the following attributes: height, width, chip_size,
        num_chips_x, num_chips_y, pixels_ignored_in_x, pixels_ignored_in_y,
        x_coords, y_coords and grid_points.

        Parameters:
            num_cols: (int) (required)
                This is the number of columns into which the image is going
                to be segmented. Another way to think about it is the number
                of chips along the x direction.

            crop: (bool) (required) (default value is "True")
                Used to determine if the image is going to be cropped or not.
                Added in case someone wants to extract the image parameters
                but, doesn't want to crop the image when doing it.

        Returns:
            self: (object)
                It is returned to be able to implement the method
                cascading."""

        # Extraction of parameters and storing them within the attributes.
        # shape returns a tuple of (rows, columns, channels*) *if color, if cv2 or numpy
        img_shape = self.img_og.shape
        self.height = img_shape[0]
        self.width = img_shape[1]
        self.chip_size = math.floor(self.width / num_cols)
        self.num_chips_x = num_cols  # math.floor(width / chip_size)
        self.num_chips_y = math.floor(self.height / self.chip_size)
        self.pixels_ignored_in_x = self.width % self.chip_size
        self.pixels_ignored_in_y = self.height % self.chip_size
        self.x_coords = list(range(0, self.width, self.chip_size))
        self.y_coords = list(range(0, self.height, self.chip_size))
        self.grid_points = []

        # Filling the grid_points. This is used when creating the chips.
        for col_idx, x_coord in enumerate(
            range(0, self.width - self.pixels_ignored_in_x, self.chip_size)
        ):
            for row_idx, y_coord in enumerate(
                range(0, self.height - self.pixels_ignored_in_y, self.chip_size)
            ):
                self.grid_points.append((x_coord, y_coord, row_idx, col_idx))

        # Cropping the image of desired.
        if crop:
            # this may only work if grey
            self.img = self.img[
                0 : self.height - self.pixels_ignored_in_y,
                0 : self.width - self.pixels_ignored_in_x,
            ]

        return self

    @staticmethod
    def cv2_to_PIL(cv2_img):
        """Auxilliary method to convert the cv2 images into PIL
        images.

        Parameters:
            cv2_img: (a cv2 image) (required)
                This parameter is the cv2 image that wants to be converted
                into a PIL image.
        Return:
            PIL_img: (a PIL image)
                The cv2 image converted into a PIL image."""

        PIL_img = transforms.ToTensor()(Image.fromarray(255 * cv2_img.astype(np.uint8)))

        # to just return the input cv2_img for debugging:
        # cv_img = transforms.ToTensor()(cv2_img)
        return PIL_img

    # TODO: Eliminate the chips dictionary and just leave the query_set pandas data frame.
    #  Do the pertinent fixes around the code (PRF).
    def chips_genesis(self, num_cols, savepath="", imgs_ext=""):
        """Method used to create the chips.

        The method takes in the number of columns into which the imaged is
        going to be segmented and then creates the chips. If desired, one can
        provide a directory saving path and the images extension and save the
        created chips.

        Parameters:
            num_cols: (int) (required)
                This is the number of columns into which the image is going
                to be segmented. Another way to think about it is the number
                of chips along the x direction.

            savepath: (str) (optional)
                The directory path were the image chips will be saved.

            imgs_ext: (str) (required with savepath)
                The extension required to save the chip images.

        Returns:
            self: (object)
                It is returned to be able to implement the method
                cascading."""

        # Getting the parameters and creating the chips dictionary
        self.parameters(num_cols)
        self.chips = {}
        self.eds_chips = {}

        # Creating the chips
        # self.grid_points.append((x_coord, y_coord, row_idx, col_idx))
        for x, y, R_idx, C_idx in self.grid_points:
            name = f"R{R_idx}C{C_idx}"  # Naming convention for the chips
            self.chips[name] = self.img[y : y + self.chip_size, x : x + self.chip_size]
            if self.eds is not None:
                raw_averaged_spectrum = self.eds[
                    y : y + self.chip_size,
                    x : x + self.chip_size,
                ].sum(axis=(0, 1), dtype="float64")
                self.eds_chips[name] = raw_averaged_spectrum

        # Creating the query_set (the chips to be classified)
        img_chips = []
        eds_chips = []
        if self.eds is not None:
            for chip_name in self.chips:
                # print('chips genesis, before cv2_to tensor, size:', self.chips[chip_name].shape)
                image = PychipClassifier.cv2_to_PIL(self.chips[chip_name])
                # print('chips genesis, after cv2_to tensor, size:', image.size())
                img_chips.append([image, "no-label-yet", chip_name])
                eds_chips.append([torch.from_numpy(self.eds_chips[chip_name])])
        else:
            for chip_name in self.chips:
                # print('chips genesis, before cv2_to tensor, size:', self.chips[chip_name].shape)
                image = PychipClassifier.cv2_to_PIL(self.chips[chip_name])
                # print('chips genesis, after cv2_to tensor, size:', image.size())
                img_chips.append([image, "no-label-yet", chip_name])

        # Turning the query_set into a pandas dataframe.
        self.query_set = pd.DataFrame(
            img_chips, columns=["images", "labels", "filename"]
        )
        if self.eds is not None:
            self.query_set["EDS"] = eds_chips

        # Saving the chips if desired
        if savepath:
            for name in self.chips:
                save_path = os.path.join(savepath, name + imgs_ext)
                img = np.array(self.chips[name])
                cv2.imwrite(save_path, img)
        return self

    # TODO: Perhaps change the Image.fromarray() to tranform the whole stacked numpy array (PRF)
    # TODO: Add the optional functionality to save the support set when it is created. (PRF)
    def support_genesis(self, support_dict={}, support_path=""):
        """Uses an existing support set or creates one.

        The method either creates a support set by using a given support dictionary
        or uses and existing support set that exists in a given directory path.

        Parameters:
            support_dict: (dict) (required but mutually exclusive with support_path)
                This is the dictionary that tells the method how to create the
                support set. Its keys are suppossed to be the name of the labels
                and the values of the keys a list containing the name of the desired
                chips to be placed inside that label.

            support_path: (str) (required but mutually exclusive with support_dict)
                This is the directory path that tells the method were to find the
                already existing support set. The given directory must contain a
                special structure in which each of its sub-directories is named after
                the label it represents and they must contain the images to be used.

        Returns:
            self: (object)
                It is returned to be able to implement the method cascading."""

        support_set = []
        # Using an already existing support set
        if support_path:
            for label in os.listdir(support_path):
                label_path = os.path.join(support_path, label)
                for support_chip_name in os.listdir(label_path):
                    support_chip_path = os.path.join(label_path, support_chip_name)
                    image = PychipClassifier.cv2_to_PIL(cv2.imread(support_chip_path))
                    support_set.append([image, label, support_chip_name])

        # Creating a support set
        elif support_dict:
            if self.eds is not None:
                for label in support_dict:
                    for chip_name in support_dict[label]:
                        image = PychipClassifier.cv2_to_PIL(self.chips[chip_name])
                        spectra = torch.from_numpy(self.eds_chips[chip_name])
                        support_set.append([image, spectra, label, chip_name])
            else:
                for label in support_dict:
                    for chip_name in support_dict[label]:
                        image = PychipClassifier.cv2_to_PIL(self.chips[chip_name])
                        support_set.append([image, label, chip_name])

        # Raising an error if the path and the dict are not given
        else:
            raise Exception(
                "Please provide a support dictionary to create a support set or a path to an existing support set."
            )

        # Turning the support set into a pandas dataframe.
        if self.eds is not None:
            self.support_set = pd.DataFrame(
                support_set, columns=["images", "spectra", "labels", "filename"]
            )
        else:
            self.support_set = pd.DataFrame(
                support_set, columns=["images", "labels", "filename"]
            )

        return self

    def predict(
        self, encoder="torch101", max_query_size=100, seed=42, savepath="", filename=""
    ):
        """Makes the prediction of the label of the chips.

        The method takes in the desired encoder to use, the maximum batch size, a seed to make
        constant predictions along different runs and an optional directory path and filename to
        save the predictions for each chip if desired.

        Parameters:
            encoder: (str) (required, default value is resnet101)
                This parameter decides which resnet to use. The available values are: resnet18,
                resnet34, resnet50, resnet101 and resnet152.

            max_query_size: (int) (required, default values is 20)
                Determines the maximum batch size to compute at a time.

            seed: (int) (required, default value os 42)
                This is a random seed required to make the predictions constant along
                runs. Notice that the seed default value is 42 as it is the ultimate
                answer to life, the universe and everything.

            savepath: (str) (optional)
                A directory path given to save the results_multimodal of the prediction.

            filename: (str) (required with savepath)
                The name that is going to be given to the save file which will contain
                the results_multimodal of the predictions.

        Returns:
            self: (object)
                It is returned to be able to implement the method cascading."""

        # Selecting the encoder
        # TODO: use the torch versions instead?
        if encoder == "resnet18":
            enkoder = resnet18(pretrained=True, place_on_device=False)
        elif encoder == "resnet34":
            enkoder = resnet34(pretrained=True, place_on_device=False)
        elif encoder == "resnet50":
            enkoder = resnet50(pretrained=True, place_on_device=False)
        elif encoder == "resnet101":
            enkoder = resnet101(pretrained=True, place_on_device=False)
        elif encoder == "resnet152":
            enkoder = resnet152(pretrained=True, place_on_device=False)
        elif encoder == "torch101":
            enkoder = models.resnet101(pretrained=True)
        elif encoder == "shufflenet":
            enkoder = models.shufflenet_v2_x1_0(pretrained=True)
        else:
            raise ValueError(
                "Encoder given not available. Available encoders are:"
                "resnet18, resnet34, resnet50, resnet101, resnet152."
            )

        model = PrototypicalNet(encoder=enkoder, device="cpu")

        # Predicting the labels for chips and timing the prediction.
        start_time = time.time()
        torch.manual_seed(seed)
        norm = Softmax(dim=1)
        output_probabilities = []

        # load the support set data
        support_set = CustomImageDataset(
            self.support_set, transform=resize, size=(255, 255)
        )
        stacked_support_set = {}

        # get class inds
        for label in support_set.labels:
            stacked_support_set[label] = torch.stack(
                [x[0] for i, x in enumerate(support_set) if x[1] == label]
            )

        query_set = CustomImageDataset(
            self.query_set, transform=resize, size=(255, 255)
        )
        query_loader = DataLoader(query_set, batch_size=max_query_size)
        # construct a batches to pass to the model
        # support_set is a dictionary of stacked tensors, query_set is still a Dataset object
        # batch = {'support_examples': support_set, 'query_examples': query_set}
        # model needs to switch to eval mode
        fnames = []
        model.eval()
        with torch.no_grad():
            for i, q in enumerate(query_loader):
                # q[0] is a stack of image data q[1] can be ignored
                print("Computing batch %s" % str(i))
                output = model(stacked_support_set, q[0])
                probabilities = norm(output)
                output_probabilities.append(probabilities)
                fnames.append(q[2])

        # Turning the results_multimodal into a dataframe
        self.results = pd.DataFrame(
            tensor_i.tolist()
            for sublist in output_probabilities
            for tensor_i in sublist
        )

        # Assigning label names to the columns
        self.results.columns = support_set.labels

        # Formatting filenames for each chip
        flat_list = [
            os.path.basename(item).split(".")[0]
            for sublist in fnames
            for item in sublist
        ]

        # Assigning the index row, column position in the convention of R{row_idx}C{col_idx} to the dataframe.
        self.results.index = flat_list

        # Creating the prediction column by comparing the max value between the labels.
        self.results["prediction"] = self.results.idxmax(axis=1)

        # renaming the first column 'chip'
        # self.results_multimodal.reset_index(inplace=True)
        # self.results_multimodal = self.results_multimodal.rename(columns={'index': 'chip'})
        self.results["chip"] = self.results.index

        # TODO:make sure to flip any predictions of the support set that aren't consistent with the support set label
        self.total_time = time.time() - start_time
        print(self.total_time)

        if savepath:
            filepath = os.path.join(savepath, filename)
            self.results.to_csv(filepath)
        return self

    def predict_multimodal(
        self,
        image_encoder="shufflenet",
        eds_encoder="raw",
        max_query_size=100,
        seed=42,
        savepath="",
        filename="",
        cuda_device="cuda:0"
    ):
        """Makes the prediction of the label of the chips.

        The method takes in the desired encoder to use, the maximum batch size, a seed to make
        constant predictions along different runs and an optional directory path and filename to
        save the predictions for each chip if desired.

        Parameters:
            image_encoder: (str) (required, default value is resnet101)
                This parameter decides which resnet to use. The available values are: resnet18,
                resnet34, resnet50, resnet101 and resnet152.

            eds_encoder: (str) (required, default value is 'raw')
                This parameter currently passes raw spectra through an identity

            max_query_size: (int) (required, default values is 20)
                Determines the maximum batch size to compute at a time.

            seed: (int) (required, default value os 42)
                This is a random seed required to make the predictions constant along
                runs. Notice that the seed default value is 42 as it is the ultimate
                answer to life, the universe and everything.

            savepath: (str) (optional)
                A directory path given to save the results_multimodal of the prediction.

            filename: (str) (required with savepath)
                The name that is going to be given to the save file which will contain
                the results_multimodal of the predictions.

        Returns:
            self: (object)
                It is returned to be able to implement the method cascading."""
        # CUDA for PyTorch
        use_cuda = torch.cuda.is_available()
        device = torch.device(cuda_device if use_cuda else "cpu")
        # Selecting the encoder
        # TODO: use the torch versions instead?
        if image_encoder == "micronet":
            # TODO: I hard coded shufflenet because if our encoder was set to torch101 it wouldn't work (needs to be
            # TODO: in the format 'resnet50'... the way torch recognizes it)
            # TODO: make so we dont have to download it every time
            enkoder = torch.hub.load(
                "pytorch/vision:v0.10.0", "resnet50", pretrained=False
            )
            url = pmm.util.get_pretrained_microscopynet_url('resnet50', 'micronet')
            enkoder.load_state_dict(model_zoo.load_url(url, map_location=torch.device(device)))
            # url = pmm.util.get_pretrained_microscopynet_url("resnet101", "micronet")
            # state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
            # Apply the state dict to the model
            # enkoder.load_state_dict(state_dict)
            enkoder.fc = nn.Flatten()
            model = PrototypicalNet(encoder=enkoder, device=device)
        elif image_encoder == "torch101":
            enkoder = models.resnet101(pretrained=True)
            enkoder.fc = nn.Flatten()
            model = PrototypicalNet(encoder=enkoder, device=device)
        elif image_encoder == "shufflenet":
            enkoder = models.shufflenet_v2_x1_0(pretrained=True)
            enkoder.fc = nn.Flatten()
            model = PrototypicalNet(encoder=enkoder, device=device)
        else:
            raise ValueError(
                "Encoder given not available. Available encoders are:"
                "micronet, torch101, and shufflenet"
            )

        if eds_encoder == "raw":
            eds_enkoder = Identity()
            model = MultimodalPrototypicalNet(
                image_encoder=enkoder, eds_encoder=eds_enkoder, device=device
            )
        else:
            raise NotImplementedError
        if use_cuda:
            torch.backends.cudnn.benchmark = True
            model.to(device)
        # Predicting the labels for chips and timing the prediction.
        start_time = time.time()
        torch.manual_seed(seed)
        norm = Softmax(dim=1)
        output_probabilities = []

        # load the support set data
        support_set = CustomImageDataset(
            self.support_set, transform=resize, size=(255, 255)
        )
        stacked_support_set = {}

        # get class inds
        for label in support_set.labels:
            stacked_support_set[label] = {
                "images": torch.stack(
                    [x[0] for i, x in enumerate(support_set) if x[2] == label]
                ).to(device),
                "spectra": torch.stack(
                    [x[1] for i, x in enumerate(support_set) if x[2] == label]
                ).to(device),
            }

        query_set = CustomImageDataset(
            self.query_set, transform=resize, size=(255, 255)
        )
        query_loader = DataLoader(query_set, batch_size=max_query_size)
        # construct a batches to pass to the model
        # support_set is a dictionary of stacked tensors, query_set is still a Dataset object
        # batch = {'support_examples': support_set, 'query_examples': query_set}
        # model needs to switch to eval mode
        fnames = []
        model.eval()
        with torch.no_grad():
            for i, q in enumerate(query_loader):
                # q[0] is a stack of image data q[1] can be ignored
                print("Computing batch %s" % str(i))
                if use_cuda:
                    q[0].to(device)
                output = model(stacked_support_set, q[0], q[1])
                probabilities = norm(output)
                output_probabilities.append(probabilities)
                fnames.append(q[3])

        # Turning the results_multimodal into a dataframe
        self.results = pd.DataFrame(
            tensor_i.tolist()
            for sublist in output_probabilities
            for tensor_i in sublist
        )

        # Assigning label names to the columns
        self.results.columns = support_set.labels

        # Formatting filenames for each chip
        flat_list = [
            os.path.basename(item).split(".")[0]
            for sublist in fnames
            for item in sublist
        ]

        # Assigning the index row, column position in the convention of R{row_idx}C{col_idx} to the dataframe.
        self.results.index = flat_list

        # Creating the prediction column by comparing the max value between the labels.
        self.results["prediction"] = self.results.idxmax(axis=1)

        # renaming the first column 'chip'
        # self.results_multimodal.reset_index(inplace=True)
        # self.results_multimodal = self.results_multimodal.rename(columns={'index': 'chip'})
        self.results["chip"] = self.results.index

        # TODO:make sure to flip any predictions of the support set that aren't consistent with the support set label
        self.total_time = time.time() - start_time
        print(self.total_time)

        if savepath:
            filepath = os.path.join(savepath, filename)
            self.results.to_csv(filepath)
        return self

    # TODO: make the mapping work without any order (PRF).
    # TODO: Add a legend to the image (PRF).
    # TODO: Add and error catcher (PRF)
    def color_image(self, savepath, img_name, color_dict):
        """Creates and saves the color labelled image. .

        This method colors the image sections according to their corresponding labels.
        Then it uses the given savepath and image name to save the color labelled image.

        Parameters:
            savepath: (str) (required)
                Directory path that is going to be used to save the color
                labled image.

            img_name: (str) (required)
                The name to be given to the color labelled image.

        Returns:
            self: (object)
                It is returned to be able to implement the method cascading."""

        # Dictionary identifying the predicted labels with integers. Also they are in order.
        mapping = {
            label: idx for idx, label in enumerate(sorted(set(self.results.prediction)))
        }

        # Creating a zeros numpy array to contain the label for each of the pixels.
        # self.color_labels = np.zeros((self.height - self.pixels_ignored_in_y, self.width - self.pixels_ignored_in_x))

        # give img_og 3 channels if they don't already exist
        if len(self.img_og.shape) == 2:
            new_og = np.zeros((self.img_og.shape[0], self.img_og.shape[1], 3))
            new_og[:, :, 0] = self.img_og  # same value in each channel
            new_og[:, :, 1] = self.img_og
            new_og[:, :, 2] = self.img_og
            self.img_og = new_og

        # creates a version of the output image with only one color overlaid. This is just for presentations and will
        # only run if the following is set to true:
        single_overlay = False
        if single_overlay:
            save = "results_single_overlay.jpg"
            filepath = os.path.join(savepath, save)
            img = self.img_og
            height, width = img.shape[0], img.shape[1]
            top_left = (0, 0)
            bottom_right = (width, height)
            bgr = (20, 20, 20)
            alpha = 0.3
            overlay = img.copy()
            output = img.copy()
            cv2.rectangle(overlay, top_left, bottom_right, bgr, -1)
            cv2.addWeighted(overlay, alpha, output, float(1 - alpha), 0, output)
            cv2.imwrite(filepath, output)
        else:
            pass

        # Assigning colored label sections to the image
        for x_coord, y_coord, row_idx, col_idx in self.grid_points:
            gridpoint = (x_coord, y_coord, col_idx, row_idx)
            results_idx = f"R{row_idx}C{col_idx}"
            label = self.results.loc[results_idx, "prediction"]
            # self.color_labels[y_coord: y_coord + self.chip_size, x_coord: x_coord + self.chip_size] = label
            bgr = color_dict[label]
            self.img_og = create_overlay(self.img_og, gridpoint, bgr, self.chip_size)
        filepath = os.path.join(savepath, img_name)
        cv2.imwrite(filepath, self.img_og)
        # plt.imsave(filepath, color.label2rgb(self.color_labels, self.img, colors=["blue", "red"], kind='overlay'))
        return self

    def stats_plot(self, savepath, img_name, color_dict):
        file = self.results
        filepath = os.path.join(savepath, img_name)
        title = "Abundance (chip count)"
        save_stats_plot(
            file,
            filepath,
            color_dict,
            title,
            show_count=True,
            show_y_axis=False,
            bold_edges=False,
        )
        return self


class CustomImageDataset(Dataset):
    def __init__(self, image_labels, transform=None, target_transform=None, **kwargs):

        super(CustomImageDataset, self).__init__()

        self.imgs_labels = image_labels
        self.labels = image_labels["labels"].unique()
        self.transform = transform
        self.target_transform = target_transform
        self.kwargs = kwargs

    def __len__(self):
        return len(self.imgs_labels)

    def __getitem__(self, idx):
        image = self.imgs_labels.iloc[idx, 0]
        label = self.imgs_labels.iloc[idx, 1]
        filename = self.imgs_labels.iloc[idx, 2]

        if self.transform:
            image = self.transform(image, **self.kwargs)
        if self.target_transform:
            label = self.target_transform(label)
        if "EDS" in self.imgs_labels:
            spectra = self.imgs_labels.iloc[idx, 3]
            return image, spectra, label, filename
        else:
            return image, label, filename


def is_grey(img):
    if len(img.shape) < 3:
        return True
    if img.shape[2] == 1:
        return True
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if (b == g).all() and (b == r).all():
        return True
    return False
