import os
import argparse
from dataclasses import dataclass
from pathlib import Path

import json
import torch
import torch.nn as nn

import sys
sys.path.append("pcn/src")
from train_pcn_xyz import PCN
from module.stl_file_loader import STLFileLoader

@dataclass
class PredictConfig:
    """予測設定を管理するクラス"""
    # モデル設定
    model_path: Path
    device: str
    
    # ディレクトリ設定
    after_dir: Path
    before_dir: Path
    output_dir: Path
    
    def __post_init__(self):
        """出力ディレクトリの初期化"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

class ModelPredictor:
    """PCNモデルの予測を管理するクラス"""
    
    def __init__(self, cfg: PredictConfig):
        self.cfg = cfg
        self.device = cfg.device
    
    def predict_stl_files(self):
        """STLファイルの予測を実行"""
        # lastを含むSTLファイルを検索
        test_files = [f for f in os.listdir(self.cfg.after_dir) if "last" in f and f.endswith(".STL")]
        
        if not test_files:
            print("予測対象となる'last'を含むSTLファイルが見つかりませんでした。")
            return

        total_loss = 0.0
        
        for test_file in test_files:
            print(f"\n{test_file}の予測を開始します...")
            test_path = self.cfg.after_dir / test_file
            base_name = test_file[:-9]  # '_last.STL'を除去
            before_file = base_name + "_first.STL"
            before_path = self.cfg.before_dir / before_file
            
            loss = self._process_file(test_path, before_path)
            total_loss += loss

        # 平均lossの計算と保存
        avg_loss = total_loss / len(test_files)
        self._save_average_loss(avg_loss)
    
    def _process_file(self, test_path: Path, before_path: Path):
        """個別のSTLファイルを処理"""
        # STLファイルの読み込み
        after_nodes, before_nodes, faces = STLFileLoader.load_file_pair(test_path, before_path)
    
        # モデルの準備
        num_nodes = after_nodes.shape[0]
        model = self._prepare_model(num_nodes)
        criterion = nn.MSELoss()

        # 予測の実行と評価
        predicted_coords = self._predict_coordinates(model, after_nodes).to("cpu")

        # lossの計算
        before_coords = before_nodes[:, 1:].float().contiguous()
        before_coords = before_coords.unsqueeze(0)  # バッチ次元を追加
        loss = criterion(predicted_coords, before_coords).item()
        
        # 予測結果の保存
        self._save_prediction(predicted_coords, after_nodes, faces, test_path.stem[:-5])
        return loss
    
    def _save_average_loss(self, avg_loss: float):
        """平均lossをtraining_log.jsonに保存"""
        log_path = self.cfg.model_path.parent / "training_logs.json"

        with open(log_path, 'r') as f:
            logs = json.load(f)
        
        logs["prediction_loss"] = avg_loss
        
        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=4)
        print(f"平均Loss ({avg_loss:.6f})をtraining_logs.jsonに保存しました")

    def _prepare_model(self, num_nodes: int) -> PCN:
        """モデルの準備"""
        model = PCN(num_dense=num_nodes)
        model.load_state_dict(torch.load(self.cfg.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def _predict_coordinates(self, model: PCN, after_nodes: torch.Tensor) -> torch.Tensor:
        """座標の予測を実行"""
        # 座標データのみを抽出 (x, y, z)
        after_coords = after_nodes[:, 1:].float().contiguous()
        
        # バッチ次元を追加してGPUに転送
        after_coords = after_coords.unsqueeze(0).to(self.device)
        print(f"入力データの形状: {after_coords.shape}")
        
        # 予測の実行
        with torch.no_grad():
            predicted_coords = model(after_coords)
        
        print(f"予測データの形状: {predicted_coords.shape}")
        
        return predicted_coords

    def _save_prediction(self, predicted_coords: torch.Tensor, after_nodes: torch.Tensor,
                        faces: torch.Tensor, base_name: str):
        """予測結果の保存"""
        predicted_coords = predicted_coords.squeeze(0)  # バッチ次元を削除
        predicted_nodes = torch.cat([after_nodes[:, 0:1], predicted_coords], dim=1)
        output_file = f"predicted_{base_name}_first.stl"
        output_path = self.cfg.output_dir / output_file
        STLFileLoader.save_to_stl(predicted_nodes, faces, output_path)
        print(f"予測結果を保存しました: {output_path}")

def parse_args() -> PredictConfig:
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description='PCNモデルを使用した予測の実行')
    
    # パスの設定
    parser.add_argument("--model-path", type=str, required=True, help="学習済みモデルのパス")
    parser.add_argument("--after-dir", type=str, required=True, help="予測対象のSTLファイルがあるディレクトリ")
    parser.add_argument("--before-dir", type=str, required=True, help="比較用の初期状態STLファイルがあるディレクトリ")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="予測結果の出力先ディレクトリ")
    
    # デバイスの設定
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"],  
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="使用するデバイス（cuda/cpu）")
    
    args = parser.parse_args()
        
    # 設定オブジェクトの作成
    cfg = PredictConfig(
        model_path=Path(args.model_path),
        device=args.device,
        after_dir=Path(args.after_dir),
        before_dir=Path(args.before_dir),
        output_dir=Path(args.output_dir)
    )
    
    return cfg

def main():
    """メイン処理"""
    cfg = parse_args()
    predictor = ModelPredictor(cfg)
    predictor.predict_stl_files()

if __name__ == "__main__":
    main()
