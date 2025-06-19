import numpy as np
import os

from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image
from skimage.transform import resize

import numpy as np
import os

from PIL import Image


def interpolate_tile_boundaries(tile1, tile2, overlap_width=10):
    """Interpolacja na granicach dwóch tileów poziomo"""
    for i in range(overlap_width):
        alpha = i / float(overlap_width)
        tile1[:, -overlap_width + i] = (1 - alpha) * tile1[:, -overlap_width + i] + alpha * tile2[:, i]
    return tile1


def interpolate_vertical_boundaries(tile1, tile2, overlap_height=10):
    """Interpolacja na granicach dwóch tileów pionowo"""
    for i in range(overlap_height):
        alpha = i / float(overlap_height)
        tile1[-overlap_height + i, :] = (1 - alpha) * tile1[-overlap_height + i, :] + alpha * tile2[i, :]
    return tile1


def stack_and_interpolate(tiles, grid_size, overlap_width=10, overlap_height=10):
    """Układa klocki w gridzie i stosuje interpolację na granicach poziomych i pionowych"""
    rows = []
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            tile = tiles[i * grid_size + j]
            if j > 0:  # Interpolacja pozioma
                tile = interpolate_tile_boundaries(row[-1], tile, overlap_width)
            row.append(tile)
        # Interpolacja pionowa pomiędzy wierszami
        for i in range(1, len(row)):
            row[i] = interpolate_vertical_boundaries(row[i - 1], row[i], overlap_height)
        rows.append(np.hstack(row))

    merged_image = np.vstack(rows)
    return merged_image


def load_gradcam_tiles(file_list, directory):
    """Załaduj kafelki z plików .npy"""
    tiles = []
    for filename in file_list:
        path = os.path.join(directory, filename)
        array = np.load(path)
        tiles.append(array)
    return tiles


def main(directory):
    cam_suffixes = {
        "": "GradCAM",
        "_pp": "GradCAM++"
    }

    overlap_width = 10
    overlap_height = 10

    for suffix, method_name in cam_suffixes.items():
        print(f"Running experiments for {method_name}...")

        gradcam_10x = np.load(os.path.join(directory, f"gradcam_tile_10x{suffix}.npy"))

        gradcam_20x_files = [
            f"gradcam_tile_20x_{i}{suffix}.npy" for i in range(1, 5)
        ]
        gradcam_40x_files = [
            f"gradcam_tile_40x_{i}{suffix}.npy" for i in range(1, 17)
        ]

        gradcam_20x = stack_and_interpolate(
            load_gradcam_tiles(gradcam_20x_files, directory),
            2,
            overlap_width,
            overlap_height
        )
        gradcam_40x = stack_and_interpolate(
            load_gradcam_tiles(gradcam_40x_files, directory),
            4,
            overlap_width,
            overlap_height
        )

        # Run experiments for different resolutions
        experiment(
            gradcam_10x,
            resize(gradcam_20x, (256, 256), anti_aliasing=True),
            resize(gradcam_40x, (256, 256), anti_aliasing=True),
            f"{method_name}_256",
            f"tile_10x.png"
        )
        experiment(
            resize(gradcam_10x, (512, 512), anti_aliasing=True),
            gradcam_20x,
            resize(gradcam_40x, (512, 512), anti_aliasing=True),
            f"{method_name}_512",
            f"tile_20x.png"
        )
        experiment(
            resize(gradcam_10x, (1024, 1024), anti_aliasing=True),
            resize(gradcam_20x, (1024, 1024), anti_aliasing=True),
            gradcam_40x,
            f"{method_name}_1024",
            f"tile_40x.png"
        )


def experiment(gradcam_10x, gradcam_20x, gradcam_40x, suffix, img):
    gradcam_sum = gradcam_10x + gradcam_20x + gradcam_40x
    gradcam_sum /= 3

    image = np.array(Image.open(os.path.join(directory, img)))
    image_rgb = image / 255.0

    mean_value = np.mean(gradcam_sum)
    std_value = np.std(gradcam_sum)
    print(f"Mean: {mean_value}")
    print(f"Standard Deviation: {std_value}")
    threshold = mean_value

    def sigmoid_thresholding(arr, threshold, steepness=10):
        return 1 / (1 + np.exp(-steepness * (arr - threshold)))

    def soft_linear_thresholding(arr, threshold, max_value=1.0):
        result = np.copy(arr)
        result[arr < threshold] = 0
        result[arr >= threshold] = (arr[arr >= threshold] - threshold) / (max_value - threshold)
        return result

    gradcam_union = np.maximum.reduce([gradcam_10x, gradcam_20x, gradcam_40x])
    iou_threshold = 0.5
    binarized_10x = gradcam_10x > iou_threshold
    binarized_20x = gradcam_20x > iou_threshold
    binarized_40x = gradcam_40x > iou_threshold
    intersection = binarized_10x & binarized_20x & binarized_40x
    union = binarized_10x | binarized_20x | binarized_40x
    iou_map = intersection.astype(float) / (union.astype(float) + 1e-8)

    output_maps = [
        ("merged_gradcam_arithmetic_sum", gradcam_sum),
        ("smooth_thresholded_gradcam_sum", gradcam_sum * sigmoid_thresholding(gradcam_sum, threshold)),
        ("soft_thresholded_gradcam_sum", soft_linear_thresholding(gradcam_sum, threshold)),
        ("simple_threshold_gradcam_sum", np.where(gradcam_sum > threshold, gradcam_sum, 0)),
        ("gradcam_intersection", np.minimum.reduce([gradcam_10x, gradcam_20x, gradcam_40x])),
        ("gradcam_union", gradcam_union),
        ("smooth_thresholded_gradcam_union", gradcam_union * sigmoid_thresholding(gradcam_union, 0.7)),
        ("gradcam_mean", gradcam_sum),
        ("gradcam_product", gradcam_10x * gradcam_20x * gradcam_40x),
        ("iou_map", iou_map),
    ]

    for name, data in output_maps:
        visualization = show_cam_on_image(image_rgb, data, use_rgb=True)
        imagePil = Image.fromarray(visualization)
        imagePil.save(os.path.join(directory, f"{name}_{suffix}.png"))


if __name__ == '__main__':
    directory = "D:/DiagSet/run_results_0BCCB273_7808_34688"  # Upewnij się, że ścieżka jest poprawna
    main(directory)
