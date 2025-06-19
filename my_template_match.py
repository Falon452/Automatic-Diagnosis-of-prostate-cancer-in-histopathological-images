import argparse
import os
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from PIL import Image
from pytorch_grad_cam import (
    GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, XGradCAM,
    EigenCAM, LayerCAM
)
from torch import nn
from torchvision import models
import cv2
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity

import logging
from itertools import combinations

global ref_points
ref_points = []
selecting = False

PREDEFINED_REGIONS = {
    r"dataWSI\wsis\0BCCB273-17EA-4DE0-B530-DB721AA010E5.ndpi": [((34680, 7550), (38500, 11600))],
    r"dataWSI\wsis\AA70AF78-A731-4E3C-B423-B73EA3841EF9.ndpi": [((23644, 3328), (25948, 4452)),
                                                                ((25488, 3790), (27331, 4998)),
                                                                ((27100, 4168), (28742, 5659)),
                                                                ((28512, 4641), (29836, 5880)),
                                                                ((29664, 4830), (30931, 6079)),
                                                                ((30700, 5271), (31852, 6625)), ((31737
                                                                                                  , 5764),
                                                                                                 (32601, 7024)),
                                                                ((32486, 6111), (33062, 7455)),
                                                                ((33033, 6709), (33465, 7528)),
                                                                ((33350, 6321), (34156, 7434)),
                                                                ((34185, 6174), (35942, 8872)),
                                                                ((35222, 5901), (36633, 6510)),
                                                                ((42825, 1806), (44784, 3780)),
                                                                ((44668, 3202), (46224, 4116)),
                                                                ((46108, 3664), (47347, 4830)),
                                                                ((45734, 4053), (46368, 4389)), ((
                                                                                                     47174, 4116),
                                                                                                 (48441, 5166)),
                                                                ((48240, 4389), (49392, 5428)),
                                                                ((49248, 4672), (50198, 5985)),
                                                                ((50112, 5229), (51840, 6373)),
                                                                ((51667, 5502), (52934, 6447)),
                                                                ((52848, 5848), (53971, 6562)),
                                                                ((53798, 5365), (54835, 7056)),
                                                                ((53280, 6919), (54489, 8295))]
}


def load_images(image_paths):
    images = []
    for path in image_paths:
        img = Image.open(path)
        images.append(img)
    return images


def plot_image_grid(images, grid_size, title='Image Grid', save_path=""):
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 12))
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            ax.imshow(images[idx])
        ax.axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(save_path)
    plt.close()


def apply_cam(model, image_path, target_layer, heatmap_save_path, target_class=1,
              save_path=None, show_image=True, cam_class=GradCAM):
    image = plt.imread(image_path)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    input_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    target_layers = [model.features[target_layer]]
    targets = [ClassifierOutputTarget(target_class)]

    with cam_class(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        for i in range(len(grayscale_cam)):
            grayscale_cam_image = grayscale_cam[i, :]
            visualization = show_cam_on_image(image.astype(np.float32) / 255.0, grayscale_cam_image, use_rgb=True)

            if save_path:
                Image.fromarray(visualization.astype(np.uint8)).save(save_path)
                print(f"Image saved to {save_path}")

            if heatmap_save_path:
                np.save(heatmap_save_path, grayscale_cam_image)
                print(f"Heatmap saved to {heatmap_save_path}")

            if show_image:
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(image)
                plt.title("Original Image")
                plt.subplot(1, 2, 2)
                plt.imshow(visualization)
                plt.title(f"{cam_class.__name__} Visualization")
                plt.show()


def resize_heatmap(heatmap, target_shape):
    return cv2.resize(heatmap, (target_shape[1], target_shape[0]))


def compare_heatmaps(heatmap_path1, heatmap_path2, logger):
    heatmap1 = np.load(heatmap_path1)
    heatmap2 = np.load(heatmap_path2)
    heatmap2_resized = resize_heatmap(heatmap2, heatmap1.shape)

    heatmap1_flat = heatmap1.flatten()
    heatmap2_flat = heatmap2_resized.flatten()
    similarity = cosine_similarity([heatmap1_flat], [heatmap2_flat])[0][0]
    mse = np.mean((heatmap1_flat - heatmap2_flat) ** 2)
    heatmap1_normalized = (heatmap1 - heatmap1.min()) / (heatmap1.max() - heatmap1.min())
    heatmap2_normalized = (heatmap2_resized - heatmap2_resized.min()) / (
            heatmap2_resized.max() - heatmap2_resized.min())
    data_range = heatmap1.max() - heatmap1.min()
    score = ssim(heatmap1_normalized, heatmap2_normalized, data_range=data_range)
    logger.info(
        f"Heatmap1: {heatmap_path1} Heatmap2: {heatmap_path2}. SSIM: {score:.3f} Cosine Similarity: {similarity:.3f} Mean Squared Error: {mse:.3f}")


def load_heatmap(file_path):
    """Loads a heatmap from the given file path."""
    return np.load(file_path)


def merge_heatmaps(heatmaps, grid_shape):
    """Merges a list of heatmaps into a grid shape (rows, cols)."""
    rows, cols = grid_shape
    heatmap_height, heatmap_width = heatmaps[0].shape
    merged_heatmap = np.zeros((rows * heatmap_height, cols * heatmap_width))

    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < len(heatmaps):
                merged_heatmap[i * heatmap_height:(i + 1) * heatmap_height,
                j * heatmap_width:(j + 1) * heatmap_width] = heatmaps[idx]
    return merged_heatmap


def click_and_crop(event, x, y, flags, param):
    global ref_points, selecting

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_points.append([(x, y)])
        selecting = True

    elif event == cv2.EVENT_LBUTTONUP:
        ref_points[-1].append((x, y))
        selecting = False


def show_thumbnail_and_select_regions(slide_path, level=2, display_size=(1920, 1024)):
    global ref_points

    slide = openslide.OpenSlide(slide_path)
    thumbnail = slide.get_thumbnail(slide.level_dimensions[level])

    thumbnail = thumbnail.resize(display_size)
    thumbnail_np = np.array(thumbnail)

    thumbnail_cv = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2BGR)

    clone = thumbnail_cv.copy()
    cv2.namedWindow("Thumbnail")
    cv2.setMouseCallback("Thumbnail", click_and_crop, thumbnail_cv)

    while True:
        temp_image = clone.copy()

        for rect in ref_points:
            if len(rect) == 2:
                cv2.rectangle(temp_image, rect[0], rect[1], (0, 255, 0), 2)

        cv2.imshow("Thumbnail", temp_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            ref_points = []

        elif key == ord("c"):
            break

    cv2.destroyAllWindows()

    selected_regions = []
    scale_x = slide.level_dimensions[level][0] / display_size[0]
    scale_y = slide.level_dimensions[level][1] / display_size[1]

    for rect in ref_points:
        if len(rect) == 2:
            top_left = (int(rect[0][0] * scale_x), int(rect[0][1] * scale_y))
            bottom_right = (int(rect[1][0] * scale_x), int(rect[1][1] * scale_y))
            selected_regions.append((top_left, bottom_right))

    return selected_regions


def load_vgg16_model(checkpoint_path, device):
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(4096, 2)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # Remove 'module.' prefix if model was trained with DataParallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    return model


def find_patch_with_tile_based_matching(
        selected_regions,
        slide_path,
        target_dir,
        output_folder,
        model_40x_path,
        model_20x_path,
        model_10x_path,
        logger,
        level=2,
):
    target_hashmap = {}
    layer_40 = 28
    layer_20 = 28
    layer_10 = 28
    threshold = 100
    for label in sorted(os.listdir(target_dir)):
        label_dir = os.path.join(target_dir, label)
        if not os.path.isdir(label_dir):
            continue

        # for every .npy file in this label folder
        for fname in os.listdir(label_dir):
            if not fname.endswith('.npy'):
                continue
            parts = fname.split('.')
            blob_idx = int(parts[-2])  # the "<BLOB_IDX>" segment
            npy_path = os.path.join(label_dir, fname)
            blobs = np.load(npy_path)
            num_patches = blobs.shape[0]
            logger.info(f"Found {num_patches} patches in file: {label}/{fname}")

            for patch_idx, arr in enumerate(blobs):
                sig = tuple(tuple(arr[d, d]) for d in range(threshold))
                target_hashmap[sig] = (False, blob_idx, label, patch_idx)

    logger.info(f"Total unique signatures indexed: {len(target_hashmap)}")

    for selected_region in selected_regions:
        if selected_region:
            logger.info(f"Region: {selected_region}")

            slide = openslide.OpenSlide(slide_path)

            region_width = slide.level_dimensions[level][0]
            region_height = slide.level_dimensions[level][1]
            slide_image = slide.read_region((0, 0), level, (region_width, region_height))
            slide_image = np.array(slide_image.convert("RGB"))
            top_left, bottom_right = selected_region

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model40 = load_vgg16_model(model_40x_path, device)
            model20 = load_vgg16_model(model_20x_path, device)
            model10 = load_vgg16_model(model_10x_path, device)

            for i in range(top_left[1], bottom_right[1] - threshold):
                for j in range(top_left[0], bottom_right[0] - threshold):
                    current_pixels = tuple(tuple(slide_image[i + d, j + d]) for d in range(threshold))

                    if current_pixels in target_hashmap and target_hashmap[current_pixels][0] == False:
                        logger.info(current_pixels)
                        logger.info(f"BLOB IS ${target_hashmap[current_pixels][1]}")
                        found, blob_idx, label, patch_idx = target_hashmap[current_pixels]
                        target_hashmap[current_pixels] = (True, blob_idx, label, patch_idx)

                        lbl = label or "nolabel"
                        output_folder_k = os.path.join(
                            output_folder,
                            f"{lbl}_blob{blob_idx}_patch{patch_idx}_{i}_{j}"
                        )
                        os.makedirs(output_folder_k, exist_ok=True)

                        logger.info(f"Matched label={lbl!r}, blob={blob_idx}, patch={patch_idx} @ ({i},{j})")

                        best_top_left_corner = (
                            int(j * slide.level_downsamples[level]), int(i * slide.level_downsamples[level]))
                        output_folder_k = output_folder + f"_{i}_{j}"
                        logger.info(f"Found match over threshold at {i} {j}")
                        logger.info(f"Matched label={lbl!r}, blob={blob_idx}")

                        extract_and_save_tiles(slide_path, best_top_left_corner, output_folder_k)

                        cam_methods = {
                            "": GradCAM,
                            "PlusPlus": GradCAMPlusPlus,
                            "Layer": LayerCAM
                        }

                        NUM_CAM_LAYERS = 3

                        for i in range(1, 17):
                            for step in range(NUM_CAM_LAYERS):
                                target_layer_idx = layer_40 - 2 * step
                                for suffix, cam_cls in cam_methods.items():
                                    full_suffix = f"{suffix}_L{target_layer_idx}"
                                    apply_cam(
                                        model=model40,
                                        image_path=f"{output_folder_k}/tile_40x_{i}.png",
                                        target_layer=target_layer_idx,
                                        target_class=1,
                                        save_path=f"{output_folder_k}/gradcam_tile_40x_{i}{full_suffix}.png",
                                        heatmap_save_path=f"{output_folder_k}/gradcam_tile_40x_{i}{full_suffix}.npy",
                                        show_image=False,
                                        cam_class=cam_cls
                                    )

                        for i in range(1, 5):
                            for step in range(NUM_CAM_LAYERS):
                                target_layer_idx = layer_20 - 2 * step
                                for suffix, cam_cls in cam_methods.items():
                                    full_suffix = f"{suffix}_L{target_layer_idx}"
                                    apply_cam(
                                        model=model20,
                                        image_path=f"{output_folder_k}/tile_20x_{i}.png",
                                        target_layer=target_layer_idx,
                                        target_class=1,
                                        save_path=f"{output_folder_k}/gradcam_tile_20x_{i}{full_suffix}.png",
                                        heatmap_save_path=f"{output_folder_k}/gradcam_tile_20x_{i}{full_suffix}.npy",
                                        show_image=False,
                                        cam_class=cam_cls
                                    )

                        for step in range(NUM_CAM_LAYERS):
                            target_layer_idx = layer_10 - 2 * step
                            for suffix, cam_cls in cam_methods.items():
                                full_suffix = f"{suffix}_L{target_layer_idx}"
                                apply_cam(
                                    model=model10,
                                    image_path=f"{output_folder_k}/tile_10x.png",
                                    target_layer=target_layer_idx,
                                    target_class=1,
                                    save_path=f"{output_folder_k}/gradcam_tile_10x{full_suffix}.png",
                                    heatmap_save_path=f"{output_folder_k}/gradcam_tile_10x{full_suffix}.npy",
                                    show_image=False,
                                    cam_class=cam_cls
                                )

                        for suffix in cam_methods.keys():
                            for step in range(NUM_CAM_LAYERS):
                                target_layer_idx = layer_40 - 2 * step
                                full_suffix = f"{suffix}_L{target_layer_idx}"

                                image_paths_40x = [f'{output_folder_k}/gradcam_tile_40x_{i}{full_suffix}.png' for i in
                                                   range(1, 17)]
                                heatmaps_40x = [load_heatmap(f"{output_folder_k}/gradcam_tile_40x_{i}{full_suffix}.npy")
                                                for i
                                                in range(1, 17)]
                                merged_40x = merge_heatmaps(heatmaps_40x, (4, 4))
                                np.save(f'{output_folder_k}/merged_heatmap_40x{full_suffix}.npy', merged_40x)
                                plot_image_grid(load_images(image_paths_40x), (4, 4),
                                                f'40x Magnification Grid {full_suffix}',
                                                f'{output_folder_k}/40x_magnification_grid{full_suffix}.png')

                        for suffix in cam_methods.keys():
                            for step in range(NUM_CAM_LAYERS):
                                target_layer_idx = layer_20 - 2 * step
                                full_suffix = f"{suffix}_L{target_layer_idx}"

                                image_paths_20x = [f'{output_folder_k}/gradcam_tile_20x_{i}{full_suffix}.png' for i in
                                                   range(1, 5)]
                                heatmaps_20x = [load_heatmap(f"{output_folder_k}/gradcam_tile_20x_{i}{full_suffix}.npy")
                                                for i
                                                in range(1, 5)]
                                merged_20x = merge_heatmaps(heatmaps_20x, (2, 2))
                                np.save(f'{output_folder_k}/merged_heatmap_20x{full_suffix}.npy', merged_20x)
                                plot_image_grid(load_images(image_paths_20x), (2, 2),
                                                f'20x Magnification Grid {full_suffix}',
                                                f'{output_folder_k}/20x_magnification_grid{full_suffix}.png')

                        for suffix in cam_methods.keys():
                            for step in range(NUM_CAM_LAYERS):
                                layer_idx = layer_10 - 2 * step
                                full_suffix = f"{suffix}_L{layer_idx}"

                                compare_heatmaps(
                                    f'{output_folder_k}/merged_heatmap_40x{full_suffix}.npy',
                                    f'{output_folder_k}/gradcam_tile_10x{full_suffix}.npy',
                                    logger=logger
                                )
                                compare_heatmaps(
                                    f'{output_folder_k}/merged_heatmap_20x{full_suffix}.npy',
                                    f'{output_folder_k}/gradcam_tile_10x{full_suffix}.npy',
                                    logger=logger
                                )
                                compare_heatmaps(
                                    f'{output_folder_k}/merged_heatmap_40x{full_suffix}.npy',
                                    f'{output_folder_k}/merged_heatmap_20x{full_suffix}.npy',
                                    logger=logger
                                )

                        for step in range(NUM_CAM_LAYERS):
                            layer_idx = layer_10 - 2 * step
                            logger.info(f"==== Comparing CAM Methods at Layer {layer_idx} ====")

                            for suffix_a, suffix_b in combinations(cam_methods.keys(), 2):
                                full_suffix_a = f"{suffix_a}_L{layer_idx}"
                                full_suffix_b = f"{suffix_b}_L{layer_idx}"

                                logger.info(f"Comparing {suffix_a or 'GradCAM'} vs {suffix_b or 'GradCAM'}")

                                try:
                                    compare_heatmaps(
                                        f'{output_folder_k}/gradcam_tile_10x{full_suffix_a}.npy',
                                        f'{output_folder_k}/gradcam_tile_10x{full_suffix_b}.npy',
                                        logger=logger
                                    )
                                except Exception as e:
                                    logger.error(f"Error comparing {full_suffix_a} vs {full_suffix_b}: {e}")

                                try:
                                    compare_heatmaps(
                                        f'{output_folder_k}/merged_heatmap_20x{full_suffix_a}.npy',
                                        f'{output_folder_k}/merged_heatmap_20x{full_suffix_b}.npy',
                                        logger=logger
                                    )
                                except Exception as e:
                                    logger.error(f"Error comparing 20x merged {full_suffix_a} vs {full_suffix_b}: {e}")

                                try:
                                    compare_heatmaps(
                                        f'{output_folder_k}/merged_heatmap_40x{full_suffix_a}.npy',
                                        f'{output_folder_k}/merged_heatmap_40x{full_suffix_b}.npy',
                                        logger=logger
                                    )
                                except Exception as e:
                                    logger.error(f"Error comparing 40x merged {full_suffix_a} vs {full_suffix_b}: {e}")


def extract_and_compare_tile(slide_path, target_image_path, blob_index, match_coords, level, tile_size=256):
    slide = openslide.OpenSlide(slide_path)
    target_image = np.load(target_image_path)[blob_index]
    extracted_tile = slide.read_region(match_coords, level, (tile_size, tile_size))
    extracted_tile = np.array(extracted_tile.convert("RGB"))
    original_10x_tile = slide.read_region(match_coords, 2, (tile_size, tile_size))
    original_10x_tile = np.array(original_10x_tile.convert("RGB"))

    target_image_tile_sized = target_image[:tile_size, :tile_size]

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Target Image")
    plt.imshow(target_image_tile_sized)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Extracted Tile")
    plt.imshow(extracted_tile)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Original 10x Tile")
    plt.imshow(original_10x_tile)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def save_tile(slide, level, coords, tile_size, output_folder, filename):
    """Save a tile extracted from the slide."""
    tile = slide.read_region(coords, level, (tile_size, tile_size))
    tile = np.array(tile.convert("RGB"))
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))


def extract_and_save_tiles(slide_path, match_coords, output_folder, tile_size=256):
    """Extract tiles at different magnifications and save them."""
    slide = openslide.OpenSlide(slide_path)
    logger.info(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    save_tile(slide, level=2, coords=match_coords, tile_size=tile_size, output_folder=output_folder,
              filename="tile_10x.png")
    save_tile(slide, level=1, coords=match_coords, tile_size=512, output_folder=output_folder,
              filename="tile_20x.png")
    save_tile(slide, level=0, coords=match_coords, tile_size=1024, output_folder=output_folder,
              filename="tile_40x.png")

    offsets_level_1 = [(0, 0), (tile_size * 2, 0), (0, tile_size * 2), (tile_size * 2, tile_size * 2)]
    for idx, (dx, dy) in enumerate(offsets_level_1):
        tile_coords = (match_coords[0] + dx, match_coords[1] + dy)
        save_tile(slide, level=1, coords=tile_coords, tile_size=tile_size, output_folder=output_folder,
                  filename=f"tile_20x_{idx + 1}.png")

    offsets_level_0 = [(x * tile_size, y * tile_size) for y in range(4) for x in range(4)]
    for idx, (dx, dy) in enumerate(offsets_level_0):
        tile_coords = (match_coords[0] + dx, match_coords[1] + dy)
        save_tile(slide, level=0, coords=tile_coords, tile_size=tile_size, output_folder=output_folder,
                  filename=f"tile_40x_{idx + 1}.png")


def visualize_tiles(output_folder, grid_dim, tile_prefix):
    """Visualize saved tiles in a grid."""
    fig, axes = plt.subplots(*grid_dim, figsize=(12, 12))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        tile_path = os.path.join(output_folder, f"{tile_prefix}_{idx + 1}.png")
        if os.path.exists(tile_path):
            tile = cv2.imread(tile_path)
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            ax.imshow(tile)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def main(slide_path, target_image_path, output_folder, model_40x_path, model_20x_path, model_10x_path, logger):
    if slide_path in PREDEFINED_REGIONS:
        selected_regions = PREDEFINED_REGIONS[slide_path]
        logger.info("Using predefined region to speed up the template match")
    else:
        selected_regions = show_thumbnail_and_select_regions(slide_path, level=2)
        logger.info(f"ADD to PREDEFINED REGIONS {selected_regions}")

    find_patch_with_tile_based_matching(
        selected_regions,
        slide_path,
        target_image_path,
        level=2,
        output_folder=output_folder,
        model_40x_path=model_40x_path,
        model_20x_path=model_20x_path,
        model_10x_path=model_10x_path,
        logger=logger,
    )


r"""
python my_template_match.py --slide_path dataWSI\download\5E3DA3E2-8C48-401C-9DBB-15FD224C20A4.ndpi --target_image_path data\test\40x\2A05B17B-E3DF-4931-8C11-03B6CDC64182\R4\2A05B17B-E3DF-4931-8C11-03B6CDC64182.blob.0.npy --blob_index 1 --output_folder output_slide1 --model_40x_path final_40x_model.pth --model_20x_path final_20x_model.pth --model_10x_path best_vgg16_model_10x.pth
python my_template_match.py --gradcam_target_layer 28 --slide_path dataWSI\download\43A626A2-EF5B-411E-A7B7-79FF2E1CFC58.ndpi --target_image_path data\40x\43A626A2-EF5B-411E-A7B7-79FF2E1CFC58\R3\43A626A2-EF5B-411E-A7B7-79FF2E1CFC58.blob.0.npy --blob_index 12 --output_folder output28 --model_40x_path final_40x_model.pth --model_20x_path final_20x_model.pth --model_10x_path best_vgg16_model_10x.pth
python my_template_match.py --layer_40 28 --layer_20 26 --layer_10 24 --slide_path dataWSI\download\43A626A2-EF5B-411E-A7B7-79FF2E1CFC58.ndpi --target_image_path data\10x\43A626A2-EF5B-411E-A7B7-79FF2E1CFC58\R3\43A626A2-EF5B-411E-A7B7-79FF2E1CFC58.blob.0.npy --blob_index 2 --output_folder same_model_different_gradcam_layers --model_40x_path final_40x_model.pth --model_20x_path final_40x_model.pth --model_10x_path final_40x_model.pth
python my_template_match.py --layer_40 28 --layer_20 26 --layer_10 24 --slide_path dataWSI\download\43A626A2-EF5B-411E-A7B7-79FF2E1CFC58.ndpi --target_image_path data\10x\43A626A2-EF5B-411E-A7B7-79FF2E1CFC58\R3\43A626A2-EF5B-411E-A7B7-79FF2E1CFC58.blob.0.npy --blob_index 7 --output_folder same_model_different_gradcam_layers_7 --model_40x_path final_40x_model.pth --model_20x_path final_40x_model.pth --model_10x_path final_40x_model.pth

python my_template_match.py --layer_40 28 --layer_20 26 --layer_10 24 --slide_path dataWSI\download\0BCCB273-17EA-4DE0-B530-DB721AA010E5.ndpi --target_image_path data\10x\0BCCB273-17EA-4DE0-B530-DB721AA010E5\R5\0BCCB273-17EA-4DE0-B530-DB721AA010E5.blob.0.npy --blob_index 0 --output_folder same_model_different_gradcam_layers_R5_0 --model_40x_path final_40x_model.pth --model_20x_path final_40x_model.pth --model_10x_path final_40x_model.pth  
python my_template_match.py --layer_40 28 --layer_20 26 --layer_10 24 --slide_path dataWSI\download\0BCCB273-17EA-4DE0-B530-DB721AA010E5.ndpi --target_image_path data\10x\0BCCB273-17EA-4DE0-B530-DB721AA010E5\R5\0BCCB273-17EA-4DE0-B530-DB721AA010E5.blob.0.npy --blob_index 1 --output_folder same_model_different_gradcam_layers_R5_1 --model_40x_path final_40x_model.pth --model_20x_path final_40x_model.pth --model_10x_path final_40x_model.pth  
python my_template_match.py --layer_40 28 --layer_20 26 --layer_10 24 --slide_path dataWSI\download\0BCCB273-17EA-4DE0-B530-DB721AA010E5.ndpi --target_image_path data\10x\0BCCB273-17EA-4DE0-B530-DB721AA010E5\R5\0BCCB273-17EA-4DE0-B530-DB721AA010E5.blob.0.npy --blob_index 2 --output_folder same_model_different_gradcam_layers_R5_2 --model_40x_path final_40x_model.pth --model_20x_path final_40x_model.pth --model_10x_path final_40x_model.pth  

python my_template_match.py --layer_40 28 --layer_20 26 --layer_10 24 --slide_path dataWSI\wsis\0BCCB273-17EA-4DE0-B530-DB721AA010E5.ndpi --target_image_path data\10x\0BCCB273-17EA-4DE0-B530-DB721AA010E5\R4\0BCCB273-17EA-4DE0-B530-DB721AA010E5.blob.0.npy --blob_index 0 --output_folder doesItWork --model_40x_path results\final_40x_model.pth --model_20x_path results\final_40x_model.pth --model_10x_path results\final_40x_model.pth  
python my_template_match.py --layer_40 28 --layer_20 28 --layer_10 28 --slide_path dataWSI\wsis\0BCCB273-17EA-4DE0-B530-DB721AA010E5.ndpi --target_image_path data\10x\0BCCB273-17EA-4DE0-B530-DB721AA010E5\R4\0BCCB273-17EA-4DE0-B530-DB721AA010E5.blob.0.npy --output_folder run_results_0BCCB273 --model_40x_path run_40/checkpoint_epoch_56.pth --model_20x_path run_20/checkpoint_epoch_46.pth --model_10x_path run_10/checkpoint_epoch_89.pth  
python my_template_match.py --layer_40 28 --layer_20 28 --layer_10 28 --slide_path dataWSI\wsis\2A05B17B-E3DF-4931-8C11-03B6CDC64182.ndpi --target_image_path data/10x/2A05B17B-E3DF-4931-8C11-03B6CDC64182/R3/2A05B17B-E3DF-4931-8C11-03B6CDC64182.blob.0.npy --output_folder run_results --model_40x_path run_40/checkpoint_epoch_38.pth --model_20x_path run_20/checkpoint_epoch_25.pth --model_10x_path run_10/checkpoint_epoch_50.pth  


python my_template_match.py --layer_40 28 --layer_20 28 --layer_10 28 --slide_path dataWSI\wsis\0BCCB273-17EA-4DE0-B530-DB721AA010E5.ndpi --target_image_path data\10x\0BCCB273-17EA-4DE0-B530-DB721AA010E5\R4\0BCCB273-17EA-4DE0-B530-DB721AA010E5.blob.0.npy --output_folder run_0BCCB273 --model_40x_path run_40/checkpoint_epoch_56.pth --model_20x_path run_20/checkpoint_epoch_46.pth --model_10x_path run_10/checkpoint_epoch_89.pth  
python my_template_match.py --layer_40 28 --layer_20 28 --layer_10 28 --slide_path dataWSI\wsis\AA70AF78-A731-4E3C-B423-B73EA3841EF9.ndpi --target_image_path data\10x\AA70AF78-A731-4E3C-B423-B73EA3841EF9\R4\AA70AF78-A731-4E3C-B423-B73EA3841EF9.blob.0.npy --output_folder run_AA70AF78 --model_40x_path run_40/checkpoint_epoch_56.pth --model_20x_path run_20/checkpoint_epoch_46.pth --model_10x_path run_10/checkpoint_epoch_89.pth  
python my_template_match.py --slide_path dataWSI\wsis\AA70AF78-A731-4E3C-B423-B73EA3841EF9.ndpi --target_image_path data\10x\AA70AF78-A731-4E3C-B423-B73EA3841EF9 --output_folder run_AA70AF78 --model_40x_path run_40/checkpoint_epoch_56.pth --model_20x_path run_20/checkpoint_epoch_46.pth --model_10x_path run_10/checkpoint_epoch_89.pth  

python my_template_match.py --slide_path dataWSI\wsis\9623685A-F976-4C28-8C1C-6B1B5F217631.ndpi --target_image_path data\10x\9623685A-F976-4C28-8C1C-6B1B5F217631 --output_folder run_9623685A --model_40x_path run_40/checkpoint_epoch_56.pth --model_20x_path run_20/checkpoint_epoch_46.pth --model_10x_path run_10/checkpoint_epoch_89.pth  
python my_template_match.py --slide_path dataWSI\wsis\0BCCB273-17EA-4DE0-B530-DB721AA010E5.ndpi --target_image_path data\10x\0BCCB273-17EA-4DE0-B530-DB721AA010E5 --output_folder run_0BCCB273 --model_40x_path run_40/checkpoint_epoch_56.pth --model_20x_path run_20/checkpoint_epoch_46.pth --model_10x_path run_10/checkpoint_epoch_89.pth  
python my_template_match.py --slide_path dataWSI/wsis/5AE6D5CD-E5EC-4621-A6A9-97A2AB7BCC42.ndpi --target_image_path data\10x\5AE6D5CD-E5EC-4621-A6A9-97A2AB7BCC42 --output_folder run_5AE6D5CD --model_40x_path run_40/checkpoint_epoch_56.pth --model_20x_path run_20/checkpoint_epoch_46.pth --model_10x_path run_10/checkpoint_epoch_89.pth  

python my_template_match.py --slide_path dataWSI\wsis\626FD5D9-6947-46FF-A3E8-B73C8D1D335D.ndpi --target_image_path data\10x\626FD5D9-6947-46FF-A3E8-B73C8D1D335D --output_folder run_626FD5D9 --model_40x_path run_40/checkpoint_epoch_56.pth --model_20x_path run_20/checkpoint_epoch_46.pth --model_10x_path run_10/checkpoint_epoch_89.pth  

python my_template_match.py --slide_path dataWSI\wsis\626FD5D9-6947-46FF-A3E8-B73C8D1D335D.ndpi --target_image_path data\10x\626FD5D9-6947-46FF-A3E8-B73C8D1D335D --output_folder new_626FD5D9 --model_40x_path 40x/checkpoint_epoch_62.pth --model_20x_path 20x/checkpoint_epoch_61.pth --model_10x_path 10x/checkpoint_epoch_88.pth  

python my_template_match.py --slide_path dataWSI\wsis\626FD5D9-6947-46FF-A3E8-B73C8D1D335D.ndpi --target_image_path data\10x\626FD5D9-6947-46FF-A3E8-B73C8D1D335D --output_folder last_626FD5D9 --model_40x_path 40xnew/checkpoint_epoch_97.pth --model_20x_path 20xnew/checkpoint_epoch_173.pth --model_10x_path 10xnew/checkpoint_epoch_320.pth  
python my_template_match.py --slide_path dataWSI\wsis\0340F85A-E6C4-4305-91F9-DAE1E7E9730D.ndpi --target_image_path data\10x\0340F85A-E6C4-4305-91F9-DAE1E7E9730D --output_folder last_0340F85A --model_40x_path 40xnew/checkpoint_epoch_97.pth --model_20x_path 20xnew/checkpoint_epoch_173.pth --model_10x_path 10xnew/checkpoint_epoch_320.pth  
python my_template_match.py --slide_path dataWSI\wsis\2315BEF6-3832-4AB9-8F6D-3514AD4E6925.ndpi --target_image_path data\10x\2315BEF6-3832-4AB9-8F6D-3514AD4E6925 --output_folder last_2315BEF6 --model_40x_path 40xnew/checkpoint_epoch_97.pth --model_20x_path 20xnew/checkpoint_epoch_173.pth --model_10x_path 10xnew/checkpoint_epoch_320.pth  
python my_template_match.py --slide_path dataWSI\wsis\28FB7B71-B9E7-482D-96FB-E03FE965E087.ndpi --target_image_path data\10x\28FB7B71-B9E7-482D-96FB-E03FE965E087 --output_folder last_28FB7B71-B9E7-482D-96FB-E03FE965E087 --model_40x_path 40xnew/checkpoint_epoch_97.pth --model_20x_path 20xnew/checkpoint_epoch_173.pth --model_10x_path 10xnew/checkpoint_epoch_320.pth  

"""

"""
target must be 10x
when we select wiht mouse do it from top left corner to bottom right corner.
"""

r"""
2025-06-08 00:14:59,904 - INFO - ADD to PREDEFINED REGIONS [((2628, 3444), (4857, 4963)), ((5233, 3255), (6523, 4258)), ((6500, 3038),
 (8565, 4095)), ((8541, 3309), (10090, 4340)), ((6359, 4746), (8049, 5777)), ((8002, 5018), (9152, 5967)), ((9105, 4557), (10161, 5669
)), ((10090, 3363), (11709, 5045)), ((11592, 3010), (14643, 4394)), ((14666, 3010), (16520, 4285)), ((14478, 4312), (16661, 5614)), ((
16684, 4095), (18843, 5669)), ((28136, 5018), (30882, 5886)), ((28089, 7378), (29567, 8056)), ((29403, 6971), (30764, 7730)), ((30788,
 6727), (32031, 7595)), ((30811, 5370), (32125, 6238)), ((32102, 5533), (33369, 6374)), ((32008, 6347), (33510, 7052)), ((33440, 5289)
, (34777, 6428)), ((34003, 4258), (35552, 5533)), ((34871, 3146), (36490, 4855)), ((35739, 2170), (37617, 3906)), ((37476, 1871), (392
12, 3689)), ((39212, 2115), (40386, 3282)), ((11639, 17766), (12742, 20994)), ((12648, 17143), (15746, 18309)), ((13868, 18336), (1476
0, 19882)), ((14737, 19367), (16051, 20425)), ((16051, 19611), (17646, 20343)), ((15769, 17739), (16684, 18797)), ((16731, 18336), (17
388, 19231)), ((17388, 18689), (19477, 19502)), ((17623, 19475), (19453, 20045)), ((19477, 18797), (21073, 20072)), ((21073, 18607), (
22129, 19421)), ((21800, 18092), (22692, 18797)), ((22668, 17604), (23795, 18445)), ((23748, 17115), (25038, 17685)), ((20462, 22947),
 (21636, 24656)), ((20580, 21401), (21495, 22866)), ((21143, 19882), (22152, 21266)), ((19829, 2658), (20721, 3526)), ((19242, 4068),
(20110, 5425)), ((25156, 5343), (26001, 6021)), ((25930, 4936), (27455, 5750)), ((40386, 2495), (41958, 3363)), ((41911, 2359), (42756
, 3227))]

"""


def setup_global_logger(log_path):
    logger = logging.getLogger("match_logger")
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process slide and blob data.")
    parser.add_argument('--slide_path', type=str, required=True, help="Path to the slide file.")
    parser.add_argument('--target_image_path', type=str, required=True, help="Path to the target image file.")
    parser.add_argument('--output_folder', type=str, required=True, help="Path to the output folder.")
    parser.add_argument('--model_40x_path', type=str, required=True, help="Path to the 40x VGG16 model.")
    parser.add_argument('--model_20x_path', type=str, required=True, help="Path to the 20x VGG16 model.")
    parser.add_argument('--model_10x_path', type=str, required=True, help="Path to the 10x VGG16 model.")

    args = parser.parse_args()

    import pathlib

    wsi_id = pathlib.Path(args.slide_path).stem
    log_path = f"logs/{wsi_id}.log"
    os.makedirs("logs", exist_ok=True)

    logger = setup_global_logger(log_path)

    main(
        slide_path=args.slide_path,
        target_image_path=args.target_image_path,
        output_folder=args.output_folder,
        model_40x_path=args.model_40x_path,
        model_20x_path=args.model_20x_path,
        model_10x_path=args.model_10x_path,
        logger=logger
    )
