from PIL import Image
import os, shutil

image_dir = './distortedData/'
gt_dir = './gtData/'
images = sorted(os.listdir(image_dir))
gt = sorted(os.listdir(gt_dir))

for image in images:
    im = Image.open(os.path.join(image_dir, image))
    width, height = im.size
    os.makedirs(f'./{width}x{height}/distortedData', exist_ok=True)
    os.makedirs(f'./{width}x{height}/gtData', exist_ok=True)
    shutil.copy2(os.path.join(image_dir, image), f'./{width}x{height}/distortedData/{image}')
    shutil.copy2(os.path.join(gt_dir, image), f'./{width}x{height}/gtData/{image}')