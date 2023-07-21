import pandas as pd
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import shutil
from few_shot.pychip_classifier import PychipClassifier
import hyperspy.api as hs

# data_path is the base directory you will be working out of:
data_path = "/Users/reeh135/Projects/AutoEM/repos/multimodal/multimodal-few-shot/mandm23/"

# ------ load data ------ #
hyperspy_obj = hs.load(data_path+"20230403 2004 SI HAADF-DF4-DF2-BF 5.50 Mx.emd", sum_frames=True,
                       SI_dtype=np.int8, rebin_energy=4)
r = hyperspy_obj[12].isig[5.:10.]
#r.add_lines()
#r.plot(integration_windows='auto')
bw = r.estimate_background_windows(line_width=[5.0, 2.0])
#r.plot(background_windows=bw)
r.get_lines_intensity(background_windows=bw)

# extract spectra
spectra = r.data
# spectra = hyperspy_obj[-1].data  # 3D array of x-ray counts per pixel
# # (x and y of the sample surface, and an energy axis of 4096 energy levels)
# spectra.dtype = 'float64'
# print(spectra.shape)

# img_name is the filename of the image you want to analyze (within the data_path folder)
img_name = "HAADF_image.png"

# results_path is the folder your results_multimodal will save to
results_path = "mandm/results_multimodal_small"

# chips_path is the folder your chips will save to (if you choose to save them)
chips_path = "mandm/results_multimodal_small/chips"

# n_columns is the number of columns your image will be chipped in to
n_columns = 30

# Preview the chip size
# Assumes image is square. If not, there may be distortions in this preview that will not appear in the final chipped image
img_path = os.path.join(data_path, img_name)

img = mpimg.imread(img_path)
width = img.shape[0]
chipsize = width/n_columns
x_grid_lst = np.linspace(chipsize, width-chipsize, n_columns-1)

plt.figure(figsize=(15, 15))
imgplot = plt.imshow(img)
for x_loc in x_grid_lst:
    plt.axvline(x=x_loc, c='lemonchiffon')
    plt.axhline(y=x_loc, c='lemonchiffon')

# classifier = PychipClassifier(data_path=data_path, img_name=img_name, eds=None)
classifier = PychipClassifier(data_path=data_path, img_name=img_name, eds=spectra)

classifier.chips_genesis(n_columns)
#
# #create a support set
fig, axs = plt.subplots(nrows=n_columns, ncols=n_columns, figsize=(10, 10))

for row in range(n_columns):
    for column in range(n_columns):
        chip_name = f"R{row}C{column}"
        chip = classifier.chips[chip_name]
        axs[row, column].imshow(chip)
        # axs[row, column].axis('off')
        axs[row, column].set_yticklabels([])
        axs[row, column].set_yticks([])
        axs[row, column].set_xticklabels([])
        axs[row, column].set_xticks([])

        if row == 0:
            axs[row, column].set_title(f"C{column}")
        if column == 0:
            axs[row, column].set_ylabel(f"R{row}")

plt.show()

support_dict = {'set_1': ['R1C2', "R4C4", 'R0C23'],
                'set_2': ['R13C2', 'R14C3', 'R17C26'],
                'set_3': ['R27C2', 'R22C23']}

color_dict = {'set_1_label': (255, 0, 0),
              'set_2_label': (0, 255, 0),
              'set_3_label': (0, 0, 255),
              'no_label': (255, 255, 255)}

rows = len(support_dict.keys())
columns = max([len(support_dict[x]) for x in list(support_dict.keys())])
#one column will break plotting
if columns <= 1:
    columns = 2
fig, axs = plt.subplots(nrows=rows, ncols=columns, figsize=(10, 10))

for row, sup_set in enumerate(support_dict.keys()):
    for col, chip_name in enumerate(support_dict[sup_set]):
        chip = classifier.chips[chip_name]
        axs[row, col].imshow(chip)
        axs[row, col].axis('off')
        axs[row, col].set_title(f"support set: {sup_set}")
plt.show()

# generate support set
classifier.support_genesis(support_dict=support_dict)

 # predict
 classifier.predict_multimodal(
     savepath=results_path, image_encoder='shufflenet', eds_encoder="raw", seed=75, filename='results_multimodal_small.csv'
 )

# classifier.predict(
#    savepath=results_path, encoder='shufflenet', seed=75, filename='results_unimodal_small.csv'
#)

classifier.color_image(savepath=results_path, img_name='results_unimodal_small.jpg', color_dict=color_dict)

classifier.results.head(10)
