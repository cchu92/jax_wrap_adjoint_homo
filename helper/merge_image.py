from PIL import Image

# Set the path where your images are stored
image_folder_path = 'D:/Dropbox/Dropbox/Fraunhofer_Cluster/JAX_GPU/code/ProgMat_v1/collec_image/c3_chirality/binary/'

nb_of_images = 260
# Get a list of image paths
image_paths = [f"{image_folder_path}rve_opt{i}.png" for i in range(nb_of_images)]

# Load all images
images = [Image.open(path) for path in image_paths]

# Get the size of one image (assuming all images are the same size)
img_width, img_height = images[0].size

# Create a blank canvas with a white background
grid_width = 13
grid_height = 20
canvas_width = img_width * grid_width
canvas_height = img_height * grid_height
canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')

# Loop through the images and paste them into the canvas
for y in range(grid_height):
    for x in range(grid_width):
        img_index = y * grid_width + x
        canvas.paste(images[img_index], (x * img_width, y * img_height))

# Save the final image
save_data_path = 'D:/Dropbox/Dropbox/Fraunhofer_Cluster/JAX_GPU/code/ProgMat_v1/collec_image/c3_chirality/binary/'
canvas.save(image_folder_path+'merged_image.png')
