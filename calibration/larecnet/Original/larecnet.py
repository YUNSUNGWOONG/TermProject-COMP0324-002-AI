import torch
import torch.nn as nn
import torch.nn.functional as F

class LineFeatureExtractor(nn.Module):
    """입력 이미지에서 line heatmap을 예측하는 CNN (U-Net 스타일)"""
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.Conv2d(32, out_channels, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x  # (B, 1, H, W)

class DistortionParamEstimator(nn.Module):
    """이미지와 예측 line heatmap을 결합해 distortion parameter를 예측하는 CNN"""
    def __init__(self, img_channels=3, heatmap_channels=1, num_params=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels + heatmap_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_params)
        )

    def forward(self, img, line_heatmap):
        x = torch.cat([img, line_heatmap], dim=1)
        x = self.conv(x)
        x = self.fc(x)
        return x  # (B, num_params)

class RectificationLayer(nn.Module):
    """예측된 distortion parameter로 rectification map(UV 맵)을 생성"""
    def __init__(self, img_size, num_params=4):
        super().__init__()
        self.img_size = img_size
        self.num_params = num_params

    def forward(self, params):
        B = params.shape[0]
        H, W = self.img_size
        device = params.device

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)   # (B, H, W, 2)

        x = grid[..., 0]
        y = grid[..., 1]
        r = torch.sqrt(x ** 2 + y ** 2)

        # 파라미터 개수에 맞게 factor 계산
        factor = 1.0
        for i in range(self.num_params):
            factor += params[:, i].view(B, 1, 1) * r ** (2 * (i + 1))

        x_distorted = x * factor
        y_distorted = y * factor
        grid_distorted = torch.stack((x_distorted, y_distorted), dim=-1)
        return grid_distorted

class LaRecNet(nn.Module):
    def __init__(self, img_size=(512, 512), num_params=4):
        super().__init__()
        self.line_extractor = LineFeatureExtractor(in_channels=3, out_channels=1)
        self.param_estimator = DistortionParamEstimator(img_channels=3, heatmap_channels=1, num_params=num_params)
        self.rectify_layer = RectificationLayer(img_size=img_size, num_params=num_params)

    def forward(self, x):
        line_heatmap_pred = self.line_extractor(x)  # (B, 1, H, W)
        params = self.param_estimator(x, line_heatmap_pred)  # (B, num_params)
        uv_map = self.rectify_layer(params)  # (B, H, W, 2)
        return uv_map, line_heatmap_pred, params
