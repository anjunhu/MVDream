import os
from PIL import Image

def pngs_to_gif(input_directory, output_gif_name, duration=200):
    # Get all PNG images from the directory
    png_files = [f for f in os.listdir(input_directory) if f.endswith('.png')]
    print(png_files)
    # png_files.sort()  # Sort images to keep the order

    # Load images
    images = [Image.open(os.path.join(input_directory, file)) for file in png_files]

    # Save the images as a GIF
    if images:
        images[0].save(
            output_gif_name,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved as {output_gif_name}")
    else:
        print("No PNG files found in the directory.")

input_directory = 'ddim_inv_artefacts'  # Replace with your directory
output_gif_name = 'rendered_standing_cow_add_noise.gif.gif'  # Replace with desired output GIF name
pngs_to_gif(input_directory, output_gif_name)

input_directory = 'forward_cache_artefacts'  # Replace with your directory
output_gif_name = 'rendered_standing_cow_denoise.gif'  # Replace with desired output GIF name
pngs_to_gif(input_directory, output_gif_name)
