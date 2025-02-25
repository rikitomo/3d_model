import torch
import json
from pathlib import Path
from typing import Tuple, Dict

class PointCloudNormalizer:
    """点群データの正規化を行うクラス"""
    
    def __init__(self):
        self.center = None
        self.scale = None
    
    def fit(self, coords: torch.Tensor) -> None:
        """正規化パラメータの計算"""
        # 重心の計算
        self.center = coords.mean(dim=0)
        
        # スケールの計算（最大絶対値）
        centered = coords - self.center
        self.scale = centered.abs().max()
    
    def transform(self, coords: torch.Tensor) -> torch.Tensor:
        """座標の正規化を実行"""
        if self.center is None or self.scale is None:
            raise ValueError("normalize_paramsが設定されていません。先にfitを実行してください。")
        
        # 重心位置での正規化とスケーリング
        centered = coords - self.center
        normalized = centered / self.scale
        
        return normalized
    
    def inverse_transform(self, normalized_coords: torch.Tensor) -> torch.Tensor:
        """正規化された座標を元のスケールに戻す"""
        if self.center is None or self.scale is None:
            raise ValueError("normalize_paramsが設定されていません。")
        
        # スケールと位置を元に戻す
        scaled = normalized_coords * self.scale
        original = scaled + self.center
        
        return original
    
    def save_params(self, save_path: Path) -> None:
        """正規化パラメータの保存"""
        params = {
            "center": self.center.tolist(),
            "scale": self.scale.item()
        }
        
        with open(save_path, "w") as f:
            json.dump(params, f, indent=4)
    
    @classmethod
    def load_params(cls, load_path: Path) -> "PointCloudNormalizer":
        """正規化パラメータの読み込み"""
        normalizer = cls()
        
        with open(load_path, "r") as f:
            params = json.load(f)
        
        normalizer.center = torch.tensor(params["center"])
        normalizer.scale = params["scale"]
        
        return normalizer
