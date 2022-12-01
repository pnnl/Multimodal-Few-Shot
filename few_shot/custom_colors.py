# is it better to create one overlay with multiple rectangles or keep layering overlays with one rectangle each?
# currently, I layer the overlays, and it seems to work fine

import cv2
import math
import pandas as pd
import matplotlib.pyplot as plt


def get_label(x, y, chip_labels_csv_path):
    labels_df = pd.read_csv(chip_labels_csv_path)
    labels_df.rename(columns={'Unnamed: 0': 'chips'}, inplace=True)

    # Change this string to match how your chip names appear in the csv
    # chip_name = f"/home/reeh135/pychip/mandm/chips/query/cropX_{x}_Y_{y}.jpg"
    chip_name = f"R{y}C{x}"

    if chip_name in set(labels_df['chips']):
        label = labels_df.loc[labels_df['chips'] == chip_name, "prediction"].iloc[0]
    else:
        label = "no_label"
    return label


def get_grid(image, chip_size):
    height, width = image.shape[0], image.shape[1]
    pixels_ignored_in_x = width % chip_size
    pixels_ignored_in_y = height % chip_size
    grid_points = []
    for x_idx, x_coord in enumerate(range(0, width - pixels_ignored_in_x, chip_size)):
        for y_idx, y_coord in enumerate(range(0, height - pixels_ignored_in_y, chip_size)):
            grid_points.append((x_coord, y_coord, x_idx, y_idx))

    return grid_points


def create_overlay(image, grid_point, bgr, chip_size):
    overlay = image.copy()
    output = image.copy()

    x_coord = grid_point[0]
    y_coord = grid_point[1]
    top_left = (x_coord, y_coord)
    bottom_right = (x_coord + chip_size-1, y_coord + chip_size-1)

    cv2.rectangle(overlay, top_left, bottom_right, bgr, -1)
    alpha = 0.3

    cv2.addWeighted(overlay, alpha, output, float(1-alpha), 0, output)
    return output


def color_this_image(image_path, chip_labels_csv_path, save, color_dict, n_cols):
    image = cv2.imread(image_path)
    width = image.shape[1]
    chip_size = math.floor(width / n_cols)
    gridpoints = get_grid(image, chip_size)

    for gridpoint in gridpoints:
        label = get_label(gridpoint[2], gridpoint[3], chip_labels_csv_path)
        bgr = color_dict[label]
        image = create_overlay(image, gridpoint, bgr, chip_size)

    cv2.imwrite(save, image)
    return


def get_colors(category, color_dict, alpha):
    if category in color_dict.keys():
        bgr = color_dict[category]
        rgb = tuple(reversed(bgr))
        rgb = tuple(x / 255 for x in rgb)
    else:
        rgb = (0, 0, 0)
    rgba = rgb + (alpha,)
    return rgb, rgba


def save_stats_plot(file, save, color_dict, title, show_count=True, show_y_axis=False, bold_edges=False):
    # read data
    df = pd.read_csv(file)
    categories = df['prediction'].unique().tolist()
    counts = []
    rgb_colors = []
    rgba_colors = []
    for category in categories:
        count = df['prediction'].value_counts()[category]
        counts.append(count)

        # get colors
        alpha = 0.5
        rgb, rgba = get_colors(category, color_dict, alpha)
        rgb_colors.append(rgb)
        rgba_colors.append(rgba)

    # get scale for labels and axis
    max_count = max(counts)
    text_spacing = max_count / 30
    y_max = max_count + (4 * text_spacing)

    # create the figure
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.ylim(0, y_max)
    ax.yaxis.set_visible(show_y_axis)
    ax.set_title(title)
    if show_count:
        for idx, count in enumerate(counts):
            ax.text(idx, count + text_spacing, str(count), ha='center', color='black')
    if bold_edges:
        edge = rgb_colors
    else:
        edge = rgba_colors
    plt.bar(categories, counts, color=rgba_colors, width=0.8, edgecolor=edge, linewidth=5)

    # save the chart
    plt.savefig(save)
    return


if __name__ == '__main__':
    # User Inputs
    i = 14
    csv_path = f'../may_2022_testing/try{i}.csv'
    num_cols = 40  # NOTE: this should match whatever was used to chip the original image
    colors = {'vacuum': (0, 0, 255),
              'particles': (255, 0, 0),
              'mesh': (0, 255, 255),
              'no_label': (255, 255, 255),
              'background': (130, 130, 130)}
    # NOTE: the color_dict values are BGR, not RGB (blame cv2)
    # NOTE: the 'no-label' class will get anything that is not accounted for in your support sets (what was cropped out)

    # Run the following to create the color image
    # NOTE: you may need to modify the chip name string in the 'get_label' function to match your labels csv
    raw_image_path = '../cropped.bmp'
    image_save_path = f"../may_2022_testing/results{i}.jpg"
    #color_this_image(raw_image_path, csv_path, image_save_path, colors, num_cols)

    # Run the following to create a stats bar chart
    stats_save_path = f"../may_2022_testing/stats{i}_Arial.jpg"
    chart_title = "Abundance (chip count)"
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 20, 'font.weight': 'light'})
    save_stats_plot(csv_path, stats_save_path, colors, chart_title, show_count=True, show_y_axis=True, bold_edges=False)
