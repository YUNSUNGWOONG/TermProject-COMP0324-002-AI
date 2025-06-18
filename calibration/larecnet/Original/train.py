import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.larecnet import LaRecNet
from utils.loss import LaRecNetLoss
from utils.warp import warp_image
from utils.dataLoader import FisheyeDataset
import os

def save_pth(model, optimizaer, epoch, loss, phase, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    checkpoint = {
        'epoch': epoch,
        'pahse': phase,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizaer.state_dict(),
        'loss': loss
    }
    
    filename = os.path.join(save_path, f'larecnet_{phase.replace(" ", "_")}_epoch{epoch}.pth')
    torch.save(checkpoint, filename)
    print(f'Saved Checkpoint: {filename}')

# 하이퍼파라미터
BATCH_SIZE = 4
IMG_SIZE = (512, 512)
NUM_PARAMS = 4
STPES = { "Line Heatmap Only": 30, "Parameter Only": 20, "Full": 100 }
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
GT_IMG_DIR = './dataset/train/gtData'
DISTORTED_IMG_DIR = './dataset/train/distortedData'
GT_LINEHEATMAP_NPY_DIR = './dataset/train/lineHeatMap_npy'
CHECKPOINT_PTH = './checkpoints'

train_dataset = FisheyeDataset(
    origin_img_dir=GT_IMG_DIR,  # 정답 이미지 파일 리스트
    fisheye_img_dir=DISTORTED_IMG_DIR,  # 왜곡 이미지 파일 리스트
    origin_line_dir=GT_LINEHEATMAP_NPY_DIR,  # 정답 히트맵 npy 파일 리스트
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# 모델, 손실 함수, 옵티마이저 준비
model = LaRecNet(img_size=IMG_SIZE, num_params=NUM_PARAMS).to(DEVICE)
criterion = LaRecNetLoss().to(DEVICE)

print(f"\nTraining Device: {DEVICE}\n")
for step, epoch in STPES.items():
    print("========= {step} =========")
    match step:
        case "Line Heatmap Only":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.line_extractor.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.line_extractor.parameters(), lr=1e-4)
            
        case "Parameter Only":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.param_estimator.parameters():
                param.requires_grad = True
            for param in model.rectify_layer.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(
                list(model.param_estimator.parameters()) + list(model.rectify_layer.parameters()), lr=1e-4)
            
        case "Full":
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            
    for repeat in range(epoch):
        model.train()
        total_loss = 0
        
        num_processed = 0
        for i, (fish_img, gt_img, line_heatmap) in enumerate(train_loader):
            batch_size = fish_img.size(0)
            num_processed += batch_size
            
            fish_img = fish_img.to(DEVICE)
            gt_img = gt_img.to(DEVICE)
            line_heatmap = line_heatmap.to(DEVICE)
            optimizer.zero_grad()
            
            uv_map, line_heatmap_pred, params = model(fish_img)

            # 입력 이미지를 예측된 uv_map으로 보정 (rectified)
            rectified_img = warp_image(fish_img, uv_map)

            # 손실 계산
            loss, loss_dict = criterion(
                pred_img=rectified_img,
                gt_img=gt_img,
                pred_uv=uv_map,
                line_heatmap=line_heatmap,
                pred_line_heatmap=line_heatmap_pred,
                distortion_param=params
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % 10 == 0:
                    print(f"Stage: {step}, Epoch [{repeat + 1}/{epoch}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                    loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
                    print(f"{loss_str}\n")
                    
        avg_loss = total_loss / len(train_loader)
        print("-----------------------------------------------")
        print(f"Stage: {step}, Epoch [{repeat+1}] - Avg_Loss: {avg_loss:.4f}\n"
            f"Reconstruction: {loss_dict['Reconst']:.4f} | Perceptual: {loss_dict['Perceptual']:.4f} | "
            f"Line_Straight: {loss_dict['Line_Straight']:.4f} | Param_Reg: {loss_dict['Param_Reg']:.4f} | Smoothness: {loss_dict['Smoothness']:.4f}")
        print("-----------------------------------------------\n\n")
    save_pth(model, optimizer, epoch, avg_loss, step, CHECKPOINT_PTH)
    print("\nComplete!\n\n")
