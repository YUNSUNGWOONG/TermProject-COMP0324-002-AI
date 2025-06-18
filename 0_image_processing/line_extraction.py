import cv2
import numpy as np
import os

def get_line_information(source, dest_img, dest_npy):
    img = cv2.imread(source, cv2.IMREAD_GRAYSCALE) # 1. 이미지 불러오기 (흑백)
    edges = cv2.Canny(img, 50, 150, apertureSize=3) # 2. 엣지 검출 (Canny)

    lines = cv2.HoughLinesP( # 3. HoughLinesP로 직선 검출
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=5,
        maxLineGap=5
        )  # 파라미터는 이미지 특성에 맞게 조정[2][3][5]
    
    line_heatmap = np.zeros_like(img, dtype=np.float32) # 4. line_heatmap 생성 (float32, 0~1)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0] # 직선 부분을 1.0으로 표시
            cv2.line(line_heatmap, (x1, y1), (x2, y2), 1.0, 1)

    line_heatmap = np.clip(line_heatmap, 0, 1)
    line_heatmap_uint8 = (line_heatmap * 255).astype(np.uint8)
    cv2.imwrite(dest_img, line_heatmap_uint8) # 5. 히트맵을 이미지로 저장 (PNG)   
    np.save(dest_npy, line_heatmap) # 6. (선택) numpy 배열로 저장
    
def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")
        
rootPath = './dataset/gtData'
dest_img = './dataset/lineHeatMap_img'
dest_npy = './dataset/lineHeatMap_npy'
lists = os.listdir(rootPath)
lists = [file for file in lists if (file.endswith(".png") or file.endswith(".jpg"))]
createDirectory(dest_img)
createDirectory(dest_npy)
count = 0
for i in lists:
    source = os.path.join(rootPath, i)
    get_line_information(source, os.path.join(dest_img, i), os.path.join(dest_npy, i[:-4]) + '.npy')
    count += 1
    print(f'{count}/{len(lists)}... {source}')