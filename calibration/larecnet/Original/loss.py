import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from torchvision import models

class LaRecNetLoss(nn.Module):
    def __init__(self, lambda_rec=1.0, lambda_per=0.1, lambda_line=1.0, lambda_param_reg=0.01, lambda_smooth=0.01):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_per = lambda_per
        self.lambda_line = lambda_line
        self.lambda_param_reg = lambda_param_reg
        self.lambda_smooth = lambda_smooth

        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        self.perceptual_loss_net = LossNetwork(nn.Sequential(*list(vgg.features)[:16]).eval())
        for p in self.perceptual_loss_net.parameters():
            p.requires_grad = False
            
    def set_training_stage(self, stage):
        """훈련 단계에 따른 손실 가중치 조정"""
        if stage == 'Line Heatmap Only':
            # 1단계: 직선 검출만 훈련
            self.lambda_rec = 0.0
            self.lambda_line = 1.0
            self.lambda_param_reg = 0.0
            self.lambda_per = 0.0
            self.lambda_smooth = 0.0

        elif stage == 'Parameter Only':
            # 2단계: 파라미터 추정 훈련           
            self.lambda_rec = 0.5
            self.lambda_line = 0.0
            self.lambda_param_reg = 0.15
            self.lambda_per = 0.01
            self.lambda_smooth = 0.2
            
        elif stage == 'Full':
            # 3단계: 전체 end-to-end 훈련
            self.lambda_rec = 1.0
            self.lambda_line = 1.0
            self.lambda_param_reg = 0.05
            self.lambda_per = 0.1
            self.lambda_smooth = 0.01
                
    def parameter_regularization_loss(self, distortion_param):
        return distortion_param.pow(2).mean() # L2 norm의 제곱의 평균 (배치 전체 평균)

    def smoothness_loss(self, uv_map): # uv_map: (B, 2, H, W)
        dx = torch.abs(uv_map[:, :, :, :-1] - uv_map[:, :, :, 1:])
        dy = torch.abs(uv_map[:, :, :-1, :] - uv_map[:, :, 1:, :])
        return dx.mean() + dy.mean()

    def forward(self, pred_img, gt_img, pred_uv, line_heatmap, pred_line_heatmap, distortion_param):
        loss_rec = F.l1_loss(pred_img, gt_img)  # 1. Reconstruction Loss - Mean Absolute Error
        loss_per = self.perceptual_loss_net(pred_img, gt_img) # 2. Perceptual Loss

        # 3. Linearity Loss (직선성)
        # pred_uv: 네트워크가 예측한 rectification map (UV 맵)
        # line_heatmap: ground truth line heatmap (직선 정보) npy 파일
        loss_line = F.l1_loss(pred_line_heatmap, line_heatmap) # 정답 line heatmap(직선 형태)과 비교
    
        loss_param_reg = self.parameter_regularization_loss(distortion_param) # 4. Parameter Regularization Loss
        loss_smooth = self.smoothness_loss(pred_uv) # 5. Smoothness Loss

        # 최종 손실
        total_loss = (
            self.lambda_rec * loss_rec +
            self.lambda_per * loss_per +
            self.lambda_line * loss_line +
            self.lambda_param_reg * loss_param_reg +
            self.lambda_smooth * loss_smooth
        )
        return total_loss, {
            'Reconst': loss_rec.item(),
            'Perceptual': loss_per.item(),
            'Line_Straight': loss_line.item(),
            'Param_Reg': loss_param_reg.item(),
            'Smoothness': loss_smooth.item()
        }

class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_img, gt_img):
        loss = []
        pred_features = self.output_features(pred_img)
        gt_features = self.output_features(gt_img)
        for pred_f, gt_f in zip(pred_features, gt_features):
            loss.append(F.mse_loss(pred_f, gt_f))
        return sum(loss) / len(loss)
    
    
