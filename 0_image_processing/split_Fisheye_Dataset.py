import os, shutil
from sklearn.model_selection import train_test_split

imgCnt = {'train': 1, 'test': 1}

distorted_dir = './dataset/distortedData/'
gt_dir = './dataset/gtData/'
linemap_npy_dir = './dataset/lineHeatMap_npy'
linemap_img_dir = './dataset/lineHeatMap_img'
distorted_images = sorted(os.listdir(distorted_dir))
gt_images = sorted(os.listdir(gt_dir))
lineMap_npy = sorted(os.listdir(linemap_npy_dir))
lineMap_img = sorted(os.listdir(linemap_img_dir))

train_img, test_img = train_test_split(distorted_images, test_size=0.2, random_state=42)

for split, files in [('train', train_img), ('test', test_img)]:
    os.makedirs(f'./dataset/{split}/distortedData/', exist_ok=True)
    os.makedirs(f'./dataset/{split}/gtData/', exist_ok=True)
    os.makedirs(f'./dataset/{split}/lineHeatMap_npy/', exist_ok=True)
    os.makedirs(f'./dataset/{split}/lineHeatMap_img/', exist_ok=True)
    for file in files:
        shutil.copy2(os.path.join(distorted_dir, file), f'./dataset/{split}/distortedData/{imgCnt[split]:06}.png')
        shutil.copy2(os.path.join(gt_dir, file), f'./dataset/{split}/gtData/{imgCnt[split]:06}.png')
        shutil.copy2(os.path.join(linemap_npy_dir, file[:-4]) + '.npy', f'./dataset/{split}/lineHeatMap_npy/{imgCnt[split]:06}.npy')
        shutil.copy2(os.path.join(linemap_img_dir, file), f'./dataset/{split}/lineHeatMap_img/{imgCnt[split]:06}.png')
        imgCnt[split] += 1