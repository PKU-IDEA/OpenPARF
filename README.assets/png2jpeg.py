import glob
from PIL import Image

png_file = './openparf_logo.png'


image = Image.open(png_file)
white_background = Image.new('RGB', image.size, (255, 255, 255))
white_background.paste(image, mask=image.split()[3])
white_background.save(png_file.replace(".png", ".jpg"))