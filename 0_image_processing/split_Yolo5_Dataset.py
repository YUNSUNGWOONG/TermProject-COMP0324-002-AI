import os, shutil
from sklearn.model_selection import train_test_split

# category = ['campus', 'city', 'residential', 'road']
imgCnt = {'train': 1, 'val': 1}


image_dir = './images/train2017/'
label_dir = './labels/train2017/'
images = sorted(os.listdir(image_dir))
labels = sorted(os.listdir(label_dir))

labelIdx = 0
for image in images:
    if (image[:-4] != labels[labelIdx][:-4]):
        f=open(os.path.join(label_dir, (image[:-4] + '.txt')), "w+")
        print(f'Crated TXT files: {image[:-4]}.txt')
        f.close()
    else:
        if (len(labels) > labelIdx + 1):
            labelIdx += 1
    
labels = sorted(os.listdir(label_dir))

train_img, test_img = train_test_split(images, test_size=0.2, random_state=42)

for split, files in [('train', train_img), ('val', test_img)]:
    os.makedirs(f'./{split}/images/train2017/', exist_ok=True)
    os.makedirs(f'./{split}/labels/train2017/', exist_ok=True)
    for file in files:
        shutil.copy2(os.path.join(image_dir, file), f'./{split}/images/train2017/{imgCnt[split]:06}.png')
        shutil.copy2(os.path.join(label_dir, file)[:-4] + '.txt', f'./{split}/labels/train2017/{imgCnt[split]:06}.txt')
        imgCnt[split] += 1