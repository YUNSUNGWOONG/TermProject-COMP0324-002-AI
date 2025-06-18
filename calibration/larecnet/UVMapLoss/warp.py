import torch.nn.functional as F

def warp_image(image, flow_field, mode='bilinear', align_corners=True):
    """
    LaRecNet 논문에서 채택한 방법으로 이미지 또는 히트맵을 UV 맵(flow_field)으로 warp하는 함수

    Args:
        image (torch.Tensor): (B, C, H, W) 형태의 입력 이미지 또는 히트맵
        flow_field (torch.Tensor): (B, H, W, 2) 형태의 UV 맵, [-1, 1] 범위로 정규화된 grid
        mode (str): interpolation 모드, 기본은 'bilinear'
        align_corners (bool): grid_sample의 align_corners 옵션

    Returns:
        torch.Tensor: (B, C, H, W) 형태의 warp된 이미지
    """
    # flow_field는 이미 [-1, 1] 범위로 정규화되어 있다고 가정
    warped = F.grid_sample(image, flow_field, mode=mode, align_corners=align_corners)
    return warped
