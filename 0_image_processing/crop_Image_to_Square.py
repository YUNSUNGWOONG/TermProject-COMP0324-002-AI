from PIL import Image
import os, re

def crop_to_square(image_path, output_path, cnt, total, rescaleWidth):
    img = Image.open(image_path)
    width, height = img.size
    
    size = min(width, height)

    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size

    
    cropped_img = img.crop((left, top, right, bottom))
    if (rescaleWidth > 0):
        size = rescaleWidth
        cropped_img = cropped_img.resize((size, size), Image.LANCZOS)
    
    cropped_img.save(output_path)
    
    print(f"[{cnt}/{total}] Saved cropped image ({size}x{size}) | from {image_path} to {output_path}")

width = int(input("Enter width of image (n x n) / Enter ""0"" -> Not edit: "))

directory_dir = './'
directoryList = sorted(os.listdir(directory_dir))

os.makedirs(f'./distortedData', exist_ok=True)
os.makedirs(f'./gtData', exist_ok=True)

cntDistortedImg = {}
cntGTImg = {}
direcCnt = 0
for directory in directoryList:   
    if (not re.compile(r'^\d+x\d+$').match(directory)):
        continue
    direcCnt += 1
    target_dir = os.path.join(directory_dir, directory)
    target_List = sorted(os.listdir(os.path.join(target_dir, 'distortedData')))

    cntDistortedImg[directory] = 0
    for target in target_List:
        cntDistortedImg[directory] += 1
        crop_to_square(os.path.join(target_dir, 'distortedData', target), os.path.join('./distortedData', target), cntDistortedImg[directory], len(target_List), width)
    
    target_List = sorted(os.listdir(os.path.join(target_dir, 'gtData')))
    cntGTImg[directory] = 0
    for target in target_List:
        cntGTImg[directory] += 1
        crop_to_square(os.path.join(target_dir, 'gtData', target), os.path.join('./gtData', target), cntGTImg[directory], len(target_List), width)

keys = list(cntDistortedImg.keys())
print("\n\n---------------------------------------------------------")
print("Converted: (Resolution | Distorted | Ground Truth)")
for res in keys:
    print(f'{res} | {cntDistortedImg[res]} | {cntGTImg[res]}')
print(f'\nTotal | {sum(cntDistortedImg.keys())} | {sum(cntGTImg.keys())}\n\n')