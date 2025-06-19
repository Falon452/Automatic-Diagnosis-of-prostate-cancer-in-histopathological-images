import openslide
from PIL import Image
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python show_wsi_thumbnail.py <path_to_ndpi_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        # Open the NDPI file
        slide = openslide.OpenSlide(file_path)

        # Get a thumbnail of the slide
        thumbnail = slide.get_thumbnail((1024, 1024))  # Resize to 1024x1024 or any desired dimensions

        # Convert to a PIL Image and display
        plt.imshow(thumbnail)
        plt.title("Whole Slide Image Thumbnail")
        plt.axis("off")
        plt.show()

    except Exception as e:
        print(f"Error loading file: {e}")
