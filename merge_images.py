import matplotlib.pyplot as plt
from PIL import Image


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


if __name__ == '__main__':
    # Image paths for 20x (2x2 grid) and 40x (4x4 grid)
    image_paths_20x = [
        'output_tiles\gradcam_tile_20x_1.png',
        'output_tiles\gradcam_tile_20x_2.png',
        'output_tiles\gradcam_tile_20x_3.png',
        'output_tiles\gradcam_tile_20x_4.png',
    ]
    image_paths_40x = [
        'output_tiles\gradcam_tile_40x_1.png',
        'output_tiles\gradcam_tile_40x_2.png',
        'output_tiles\gradcam_tile_40x_3.png',
        'output_tiles\gradcam_tile_40x_4.png',
        'output_tiles\gradcam_tile_40x_5.png',
        'output_tiles\gradcam_tile_40x_6.png',
        'output_tiles\gradcam_tile_40x_7.png',
        'output_tiles\gradcam_tile_40x_8.png',
        'output_tiles\gradcam_tile_40x_9.png',
        'output_tiles\gradcam_tile_40x_10.png',
        'output_tiles\gradcam_tile_40x_11.png',
        'output_tiles\gradcam_tile_40x_12.png',
        'output_tiles\gradcam_tile_40x_13.png',
        'output_tiles\gradcam_tile_40x_14.png',
        'output_tiles\gradcam_tile_40x_15.png',
        'output_tiles\gradcam_tile_40x_16.png',
    ]
    # Load images for each grid
    images_20x = load_images(image_paths_20x)
    images_40x = load_images(image_paths_40x)

    # Plot 20x images in a 2x2 grid
    plot_image_grid(images_20x, grid_size=(2, 2), title='20x Magnification Grid',
                    save_path='results/output_tiles/20x_magnification_grid.png')

    # Save 40x images in a 4x4 grid
    plot_image_grid(images_40x, grid_size=(4, 4), title='40x Magnification Grid',
                    save_path='results/output_tiles/40x_magnification_grid.png')
