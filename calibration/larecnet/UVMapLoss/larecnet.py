import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import utils.warp as warp
import utils.loss as loss

class LineDetectionModule(nn.Module):
    """
    모듈 1: 직선 검출 모듈
    어안렌즈 이미지에서 왜곡된 직선을 검출하여 히트맵으로 출력
    """
    def __init__(self, backbone='resnet18'):
        super(LineDetectionModule, self).__init__()
        
        # 백본 네트워크 (ResNet 기반)
        if backbone == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            backbone_channels = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # 직선 검출을 위한 헤드
        self.line_head = nn.Sequential(
            nn.Conv2d(backbone_channels, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),  # 직선 히트맵 출력
        )
        
        # 업샘플링을 위한 디코더
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 32, 4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 백본을 통한 특징 추출
        features = self.backbone(x)  # (B, 512, H/32, W/32)
        
        # 직선 히트맵 생성
        line_features = self.line_head(features)  # (B, 1, H/32, W/32)
        
        # 원본 해상도로 업샘플링
        line_heatmap = self.decoder(line_features)  # (B, 1, H, W)
        
        # 입력 크기와 정확히 맞추기
        line_heatmap = F.interpolate(line_heatmap, size=x.shape[-2:], 
                                   mode='bilinear', align_corners=True)
        
        # 경계 강화를 위한 후처리
        line_heatmap = self.enhance_edges(line_heatmap)
        
        return torch.clamp(line_heatmap, 0.0, 1.0)
    
    def enhance_edges(self, heatmap):
        """경계 부분의 직선 검출 강화 - 적응적 경계 폭"""
        _, _, h, w = heatmap.shape
        edge_mask = torch.zeros_like(heatmap)
        
        # 이미지 크기에 비례하는 edge_width 계산
        edge_width = min(20, h//10, w//10)  # 이미지 크기의 10% 또는 최대 20픽셀
        
        # 경계 영역에 가중치 적용
        edge_mask[:, :, :edge_width, :] = 1.2
        edge_mask[:, :, -edge_width:, :] = 1.2
        edge_mask[:, :, :, :edge_width] = 1.2
        edge_mask[:, :, :, -edge_width:] = 1.2
        
        return heatmap * (1 + 0.2 * edge_mask)

class ParameterEstimationModule(nn.Module):
    """
    모듈 2: 왜곡 파라미터 추정 모듈
    이미지와 직선 히트맵을 결합하여 물리적 왜곡 파라미터 추정
    """
    def __init__(self):
        super(ParameterEstimationModule, self).__init__()
        
        # 입력: RGB 이미지(3채널) + 직선 히트맵(1채널) = 4채널
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(4, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # ResNet 블록들
            self._make_layer(64, 64, 2),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
        )
        
        # 전역 평균 풀링
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 파라미터 추정 헤드
        self.param_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 4)  # [k1, k2, u0, v0]
        )
        
        # 파라미터 초기화
        self._initialize_params()
    
    def _make_layer(self, inplanes, planes, blocks, stride=1):
        """ResNet 기본 블록 생성"""
        layers = []
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            downsample = None
            
        layers.append(BasicBlock(inplanes, planes, stride, downsample))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
            
        return nn.Sequential(*layers)
    
    def _initialize_params(self):
        """파라미터 초기화 - 왜곡 파라미터의 일반적인 범위 고려"""
        with torch.no_grad():
            # 마지막 레이어의 바이어스를 일반적인 어안렌즈 파라미터로 초기화
            self.param_head[-1].bias.data = torch.tensor([0.0, 0.0, 0.0, 0.0])
            self.param_head[-1].weight.data *= 0.01  # 작은 가중치로 초기화
    
    def forward(self, img, line_heatmap):
        # 이미지와 라인 히트맵 결합
        combined_input = torch.cat([img, line_heatmap], dim=1)  # (B, 4, H, W)
        
        # 특징 추출
        features = self.feature_extractor(combined_input)  # (B, 512, H', W')
        
        # 전역 특징
        global_features = self.global_pool(features)  # (B, 512, 1, 1)
        global_features = global_features.flatten(1)  # (B, 512)
        
        # 왜곡 파라미터 추정
        distortion_params = self.param_head(global_features)  # (B, 4)
        
        return distortion_params

class BasicBlock(nn.Module):
    """ResNet 기본 블록"""
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out

class DifferentiableRectificationLayer(nn.Module):
    """
    모듈 3: 미분 가능한 교정 레이어
    물리적 어안렌즈 왜곡 모델을 사용하여 이미지 교정
    """
    def __init__(self):
        super(DifferentiableRectificationLayer, self).__init__()
    
    def forward(self, fisheye_img, distortion_params):
        """
        어안렌즈 왜곡 보정 수행
        Args:
            fisheye_img: (B, C, H, W) 어안렌즈 이미지
            distortion_params: (B, 4) [k1, k2, u0, v0] 왜곡 파라미터
        """
        B, C, H, W = fisheye_img.shape
        device = fisheye_img.device
        
        # 왜곡 파라미터 분리
        k1 = distortion_params[:, 0].view(-1, 1, 1)  # (B, 1, 1)
        k2 = distortion_params[:, 1].view(-1, 1, 1)  # (B, 1, 1)
        u0 = distortion_params[:, 2].view(-1, 1, 1)  # (B, 1, 1)
        v0 = distortion_params[:, 3].view(-1, 1, 1)  # (B, 1, 1)
        
        # 정규화된 좌표 그리드 생성 [-1, 1]
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # 배치 차원 확장
        grid_x = grid_x.unsqueeze(0).repeat(B, 1, 1)  # (B, H, W)
        grid_y = grid_y.unsqueeze(0).repeat(B, 1, 1)  # (B, H, W)
        
        # 중심점 보정
        grid_x_centered = grid_x + u0
        grid_y_centered = grid_y + v0
        
        # 극좌표 변환
        r = torch.sqrt(grid_x_centered**2 + grid_y_centered**2)  # 반지름
        theta = torch.atan2(grid_y_centered, grid_x_centered)    # 각도
        
        # 어안렌즈 왜곡 모델 적용: r_d = r * (1 + k1*r^2 + k2*r^4)
        # 작은 값에서의 수치적 안정성을 위한 클램핑
        r_clamped = torch.clamp(r, min=1e-8)
        distortion_factor = 1 + k1 * r_clamped**2 + k2 * r_clamped**4
        r_distorted = r_clamped * distortion_factor
        
        # 직교좌표로 복원
        x_distorted = r_distorted * torch.cos(theta)
        y_distorted = r_distorted * torch.sin(theta)
        
        # 샘플링 그리드 구성 (grid_sample 형식에 맞춤)
        # grid_sample은 (x, y) 순서를 기대함
        sample_grid = torch.stack([x_distorted, y_distorted], dim=-1)  # (B, H, W, 2)
        
        # 범위를 [-1, 1]로 클램핑 (grid_sample 요구사항)
        sample_grid = torch.clamp(sample_grid, -1, 1)
        
        # 이미지 교정 수행
        rectified_img = F.grid_sample(
            fisheye_img, sample_grid, 
            mode='bilinear', 
            align_corners=True, 
            padding_mode='border'
        )
        
        return torch.clamp(rectified_img, 0.0, 1.0)
    
class LaRecNet(nn.Module):
    """
    완전한 LaRecNet 아키텍처
    논문의 3단계 모듈 구조를 정확히 구현
    """
    def __init__(self, backbone='resnet18'):
        super(LaRecNet, self).__init__()
        
        # 3개의 순차적 모듈
        self.line_detection_module = LineDetectionModule(backbone=backbone)
        self.parameter_estimation_module = ParameterEstimationModule()
        self.rectification_layer = DifferentiableRectificationLayer()
        
        # 훈련 단계 제어
        self.training_stage = 'full'  # 'line_only', 'param_only', 'full'
    
    def set_training_stage(self, stage):
        """훈련 단계 설정"""
        assert stage in ['line_only', 'param_only', 'full']
        self.training_stage = stage
        
        if stage == 'line_only':
            # 직선 검출 모듈만 학습
            for param in self.line_detection_module.parameters():
                param.requires_grad = True
            for param in self.parameter_estimation_module.parameters():
                param.requires_grad = False
            # rectification_layer는 미분 가능한 레이어이므로 gradient 필요 없음
            
        elif stage == 'param_only':
            # 파라미터 추정 모듈만 학습
            for param in self.line_detection_module.parameters():
                param.requires_grad = False
            for param in self.parameter_estimation_module.parameters():
                param.requires_grad = True
                
        else:  # 'full'
            # 전체 end-to-end 학습
            for param in self.line_detection_module.parameters():
                param.requires_grad = True
            for param in self.parameter_estimation_module.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        """
        전방향 전파
        Args:
            x: (B, 3, H, W) 어안렌즈 이미지
        Returns:
            rectified_img: (B, 3, H, W) 교정된 이미지
            line_heatmap: (B, 1, H, W) 직선 히트맵
            distortion_params: (B, 4) 왜곡 파라미터 [k1, k2, u0, v0]
        """
        # 1단계: 직선 검출
        line_heatmap = self.line_detection_module(x)
        
        # 2단계: 왜곡 파라미터 추정
        distortion_params = self.parameter_estimation_module(x, line_heatmap)
        
        # 3단계: 이미지 교정
        rectified_img = self.rectification_layer(x, distortion_params)
        
        return rectified_img, line_heatmap, distortion_params
    
    def get_line_heatmap_only(self, x):
        """직선 히트맵만 반환 (1단계 훈련용)"""
        return self.line_detection_module(x)
    
    def get_distortion_params_only(self, x, line_heatmap):
        """왜곡 파라미터만 반환 (2단계 훈련용)"""
        return self.parameter_estimation_module(x, line_heatmap)
    
class HybridParameterEstimationModule(nn.Module):
    def forward(self, img, line_heatmap):
        # 1. 신경망 기반 추정
        nn_params = self.nn_estimator(img, line_heatmap)
        
        # 2. 라인 기반 추정 (예외 처리 강화)
        try:
            # 배치별 처리
            batch_size = line_heatmap.size(0)
            geo_params_list = []
            
            for b in range(batch_size):
                # 개별 배치 처리
                heatmap_np = line_heatmap[b,0].detach().cpu().numpy()
                lines = loss.extract_lines_from_heatmap(heatmap_np)
                
                # 라인 기반 파라미터 추정
                if len(lines) > 0:
                    geo_params = warp.estimate_distortion_from_lines(lines, img.shape[-2:])
                    geo_params_list.append(geo_params)
                else:
                    geo_params_list.append(torch.zeros_like(nn_params[b]))
            
            geo_params = torch.stack(geo_params_list).to(img.device)
            
        except Exception as e:
            print(f"라인 기반 파라미터 추정 오류: {e}")
            geo_params = torch.zeros_like(nn_params)
        
        # 3. 혼합 가중치 적용
        hybrid_params = self.alpha * nn_params + (1 - self.alpha) * geo_params
        
        return hybrid_params

class EnhancedLaRecNet(nn.Module):
    """
    개선된 LaRecNet 아키텍처
    """
    def __init__(self, backbone='resnet18'):
        super(EnhancedLaRecNet, self).__init__()
        
        # 3개의 순차적 모듈
        self.line_detection_module = LineDetectionModule(backbone=backbone)
        self.parameter_estimation_module = HybridParameterEstimationModule()  # 변경된 부분
        self.rectification_layer = DifferentiableRectificationLayer()
        
        # 훈련 단계 제어
        self.training_stage = 'full'

    def set_training_stage(self, stage):
        """훈련 단계 설정 (확장 버전)"""
        assert stage in ['line_only', 'param_only', 'full']
        self.training_stage = stage
        
        # 모듈별 학습 가능 여부 설정
        line_trainable = stage in ['line_only', 'full']
        param_trainable = stage in ['param_only', 'full']
        
        # 직선 검출 모듈
        for param in self.line_detection_module.parameters():
            param.requires_grad = line_trainable
            
        # 파라미터 추정 모듈
        for param in self.parameter_estimation_module.parameters():
            if isinstance(param, nn.Parameter) and param is self.parameter_estimation_module.alpha:
                param.requires_grad = param_trainable  # alpha는 항상 학습
            else:
                param.requires_grad = param_trainable

    def forward(self, x):
        # 1단계: 직선 검출
        line_heatmap = self.line_detection_module(x)
        
        # 2단계: 하이브리드 파라미터 추정
        distortion_params = self.parameter_estimation_module(x, line_heatmap)
        
        # 3단계: 이미지 교정
        rectified_img = self.rectification_layer(x, distortion_params)
        
        return rectified_img, line_heatmap, distortion_params

    def get_hybrid_weights(self):
        """혼합 가중치 확인용"""
        return {
            'alpha': self.parameter_estimation_module.alpha.item(),
            'beta': 1 - self.parameter_estimation_module.alpha.item()
        }