import torch
import os
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from models.larecnet import EnhancedLaRecNet
from utils.loss import LaRecNetLoss
from utils.dataLoader import FisheyeDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import numpy as np

def train():
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # 1. LaRecNet 논문 기반 데이터 로더 설정
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = FisheyeDataset(
        fisheye_img_dir='./dataset/train/distortedData/',
        origin_img_dir='./dataset/train/gtData/',
        transform=transform
    )
    
    # LaRecNet은 더 작은 배치 크기가 효과적
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, 
                            num_workers=4, pin_memory=True)
    
    # 2. LaRecNet 3단계 모듈 구조 모델 초기화
    model = EnhancedLaRecNet(backbone='resnet18').to(device)
    
    # 3. LaRecNet 논문 기반 손실 함수 설정
    criterion = LaRecNetLoss(
        lambda_recon=1.0,           # 재구성 손실
        lambda_line=0.5,            # 직선 검출 손실  
        lambda_param=0.2,           # 파라미터 정규화 손실
        lambda_perceptual=0.01,      # 지각적 손실
        lambda_smoothness=0.02,     # 스무드니스 손실
        lambda_straightness=0.8,    # 직선성 보존 손실
        use_perceptual=True,
        use_straightness=True
    )
    
    # 4. 최적화기 및 스케줄러 설정
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    
    # 5. LaRecNet 3단계 훈련 과정 구현
    training_stages = [
        {'stage': 'line_only', 'epochs': 20, 'description': '직선 검출 모듈만 훈련'},
        {'stage': 'param_only', 'epochs': 10, 'description': '파라미터 추정 모듈 훈련'}, 
        {'stage': 'full', 'epochs': 100, 'description': '전체 end-to-end 훈련'}
    ]
    
    total_epochs = 0
    best_loss = float('inf')
    
    for stage_info in training_stages:
        stage = stage_info['stage']
        stage_epochs = stage_info['epochs']
        print(f"\n{'='*50}")
        print(f"훈련 단계: {stage_info['description']}")
        print(f"{'='*50}")
        
        # 훈련 단계별 모델 및 손실 함수 설정
        model.set_training_stage(stage)
        criterion.set_training_stage(stage)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError(f"{stage} 단계에서 훈련 가능한 파라미터가 없습니다!")

        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
        
        # 단계별 학습률 조정
        if stage == 'line_only':
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
        elif stage == 'param_only':
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
        else:  # full
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
        
        # 단계별 훈련 루프
        for epoch in range(stage_epochs):
            model.train()
            total_loss = 0
            stage_losses = {}
            for i, (fisheye_img, origin_img) in enumerate(train_loader):
                fisheye_img = fisheye_img.to(device)
                origin_img = origin_img.to(device)
                
                # LaRecNet 3단계 모듈 순전파
                rectified_img, line_heatmap, distortion_params = model(fisheye_img)
                
                # 원본 해상도로 맞춤
                rectified_img = F.interpolate(rectified_img, size=origin_img.shape[-2:], 
                                            mode='bilinear', align_corners=True)
                line_heatmap = F.interpolate(line_heatmap, size=origin_img.shape[-2:], 
                                           mode='bilinear', align_corners=True)
                
                # LaRecNet 논문 기반 종합 손실 계산
                loss, loss_components = criterion(
                    rectified_img=rectified_img,
                    line_heatmap=line_heatmap, 
                    distortion_params=distortion_params,
                    fisheye_img=fisheye_img,
                    gt_img=origin_img
                )
                
                # 역전파 및 최적화
                optimizer.zero_grad()
                loss.backward()
                
                # 그래디언트 클리핑 (LaRecNet의 안정적 훈련을 위해)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                
                # 손실 구성요소 기록
                for key, value in loss_components.items():
                    if key not in stage_losses:
                        stage_losses[key] = []
                    stage_losses[key].append(value)
                
                # 진행 상황 출력
                if (i + 1) % 10 == 0:
                    print(f"Stage: {stage}, Epoch [{total_epochs + epoch + 1}], "
                          f"Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                    
                    # 손실 구성요소 상세 출력
                    loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in loss_components.items()])
                    hybrid_weight = model.get_hybrid_weights()
                    print(f"  상세 손실: {loss_str} | neuropil weight: {hybrid_weight['alpha']:.4f}, geometrical weight: {hybrid_weight['beta']:.4f}")
            
            avg_loss = total_loss / len(train_loader)
            print(f"\nStage: {stage}, Epoch [{total_epochs + epoch + 1}] "
                  f"Average Loss: {avg_loss:.4f}")
            
            # 손실 구성요소 평균 출력
            avg_stage_losses = {k: np.mean(v) for k, v in stage_losses.items()}
            print(f"평균 손실 구성요소: {avg_stage_losses}")
            
            # 스케줄러 업데이트
            scheduler.step(avg_loss)
            
            # 모델 저장 (최고 성능 모델)
            if avg_loss < best_loss:
                best_loss = avg_loss
                os.makedirs('./checkpoints', exist_ok=True)
                torch.save({
                    'epoch': total_epochs + epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'stage': stage
                }, f"./checkpoints/larecnet_best_{stage}.pth")
                print(f"최고 성능 모델 저장: {stage} 단계, Loss: {avg_loss:.4f}")
            
            # 주기적 체크포인트 저장
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': total_epochs + epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'stage': stage
                }, f"./checkpoints/larecnet_{stage}_epoch{total_epochs + epoch + 1}.pth")
        
        total_epochs += stage_epochs
        print(f"{stage} 단계 훈련 완료\n")
    
    # 최종 모델 저장
    torch.save({
        'epoch': total_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_loss,
        'stage': 'final'
    }, "./checkpoints/larecnet_final.pth")
    
    print(f"LaRecNet 훈련 완료! 총 {total_epochs} 에포크")
    print(f"최고 성능: {best_loss:.4f}")

def validate_model(model, criterion, val_loader, device):
    """검증 함수 (선택적 사용)"""
    model.eval()
    total_val_loss = 0
    val_loss_components = {}
    
    with torch.no_grad():
        for fisheye_img, origin_img in val_loader:
            fisheye_img = fisheye_img.to(device)
            origin_img = origin_img.to(device)
            
            rectified_img, line_heatmap, distortion_params = model(fisheye_img)
            
            rectified_img = F.interpolate(rectified_img, size=origin_img.shape[-2:], 
                                        mode='bilinear', align_corners=True)
            line_heatmap = F.interpolate(line_heatmap, size=origin_img.shape[-2:], 
                                       mode='bilinear', align_corners=True)
            
            val_loss, loss_components = criterion(
                rectified_img=rectified_img,
                line_heatmap=line_heatmap,
                distortion_params=distortion_params, 
                fisheye_img=fisheye_img,
                gt_img=origin_img
            )
            
            total_val_loss += val_loss.item()
            
            for key, value in loss_components.items():
                if key not in val_loss_components:
                    val_loss_components[key] = []
                val_loss_components[key].append(value)
    
    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_components = {k: np.mean(v) for k, v in val_loss_components.items()}
    
    return avg_val_loss, avg_val_components

def resume_training(checkpoint_path, model, optimizer, scheduler):
    """훈련 재개 함수"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']
        stage = checkpoint['stage']
        
        print(f"체크포인트 로드: Epoch {start_epoch}, Loss {best_loss:.4f}, Stage {stage}")
        return start_epoch, best_loss, stage
    else:
        print("체크포인트를 찾을 수 없습니다. 처음부터 훈련을 시작합니다.")
        return 0, float('inf'), 'line_only'

if __name__ == '__main__':
    train()
