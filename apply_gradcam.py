import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity


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
    logger.info(f"Heatmap1: {heatmap_path1} Heatmap2: {heatmap_path2}. SSIM: {score:.3f} Cosine Similarity: {similarity:.3f} Mean Squared Error: {mse:.3f}")

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


if __name__ == '__main__':
    heatmap_paths_40x = [
        f"output_tiles/gradcam_tile_40x_{i}.npy" for i in range(1, 17)
    ]
    heatmaps_40x = [load_heatmap(path) for path in heatmap_paths_40x]

    heatmap_paths_20x = [
        f"output_tiles/gradcam_tile_20x_{i}.npy" for i in range(1, 5)
    ]
    heatmaps_20x = [load_heatmap(path) for path in heatmap_paths_20x]

    merged_heatmap_40x = merge_heatmaps(heatmaps_40x, grid_shape=(4, 4))
    merged_heatmap_20x = merge_heatmaps(heatmaps_20x, grid_shape=(2, 2))

    np.save('results/output_tiles/merged_heatmap_40x.npy', merged_heatmap_40x)
    np.save('results/output_tiles/merged_heatmap_20x.npy', merged_heatmap_20x)

    print("Comparing merged_heatmap_40x.npy and gradcam_tile_10x.npy")
    compare_heatmaps('output_tiles/merged_heatmap_40x.npy', "output_tiles\gradcam_tile_10x.npy", metric='mse')
    compare_heatmaps('output_tiles/merged_heatmap_40x.npy', "output_tiles\gradcam_tile_10x.npy", metric='cosine')
    compare_heatmaps('output_tiles/merged_heatmap_40x.npy', "output_tiles\gradcam_tile_10x.npy", metric='ssim')
    print()
    print("Comparing merged_heatmap_20x.npy and gradcam_tile_10x.npy")
    compare_heatmaps('output_tiles/merged_heatmap_20x.npy', "output_tiles\gradcam_tile_10x.npy", metric='mse')
    compare_heatmaps('output_tiles/merged_heatmap_20x.npy', "output_tiles\gradcam_tile_10x.npy", metric='cosine')
    compare_heatmaps('output_tiles/merged_heatmap_20x.npy', "output_tiles\gradcam_tile_10x.npy", metric='ssim')
