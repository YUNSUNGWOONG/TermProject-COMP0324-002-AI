import cv2, os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

def check_path(output_path):
    if not os.path.exists(output_path):
        return output_path
    
    base, ext = os.path.splitext(output_path)
    idx = 1
    new_path = f"{base}_{idx}{ext}"
    while os.path.exists(new_path):
        idx += 1
        new_path = f"{base}_{idx}{ext}"
    return new_path

def augmentation(image_path, output_path, cnt, total):
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),  # 밝기, 대비 ±20%
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),  # 색조, 채도, 명도 조절
    ])

    image = cv2.imread(image_path)

    augmented = transform(image=image)
    image_aug = augmented['image']

    save_path = check_path(output_path=output_path)
    cv2.imwrite(save_path, image_aug)
    
    print(f"[{cnt}/{total}] Saved augmented image | from {image_path} to {save_path}")

repeat = int(input("The number of repeatition?  "))

os.makedirs('./distortedData_augmented', exist_ok=True)
os.makedirs('./gtData_augmented', exist_ok=True)

cnt = 0
for i in range(repeat):
    target_List = sorted(os.listdir('./distortedData'))
    for target in target_List:
        cnt += 1
        augmentation(os.path.join('./distortedData', target), os.path.join('./distortedData_augmented', 'augmented_' + target), cnt, len(target_List) * repeat * 2)

    target_List = sorted(os.listdir('./gtData'))
    for target in target_List:
        cnt += 1
        augmentation(os.path.join('./gtData', target), os.path.join('./gtData_augmented', 'augmented_' + target), cnt, len(target_List) * repeat * 2)

print("\n\nCompleted!")