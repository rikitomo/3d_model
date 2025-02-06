import os
import json
import ast
import time
import argparse
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from module.stl_file_loader import STLFileLoader

@dataclass
class CFG:
    """実験設定を管理するクラス"""
    # 実験設定
    experiment_name: str
    output_dir: Path
    
    # モデルパラメータ
    hidden_dims: List[int]
    
    # 学習パラメータ
    batch_size: int
    num_epochs: int
    learning_rate: float
    
    # 早期終了の設定
    early_stopping_patience: int
    early_stopping_min_delta: float
    early_stopping_min_epochs: int
    
    # データ分割比率
    train_val_split_ratio: float

    # 乱数シード値
    seed: int
    
    def __post_init__(self):
        """出力ディレクトリの初期化"""
        self.experiment_dir = self.output_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # モデルの保存ディレクトリ
        self.model_dir = self.experiment_dir / "models"
        self.model_dir.mkdir(exist_ok=True)

    def _set_seed(self):
        """乱数シード値の設定"""
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class AfterBeforeDataset(Dataset):
    """変形前後の3D形状データを管理するデータセット"""
    
    def __init__(self, after_dir: str, before_dir: str):
        if not os.path.exists(after_dir) or not os.path.exists(before_dir):
            raise FileNotFoundError("指定されたディレクトリが存在しません")

        self.after_dir = after_dir
        self.before_dir = before_dir
        
        # ファイルリストの取得
        self.after_files = sorted([f for f in os.listdir(after_dir) if f.endswith("_last.STL")])
        self.before_files = sorted([f for f in os.listdir(before_dir) if f.endswith("_first.STL")])
        
        # 対応するファイルのペアを確認
        self.file_pairs = []
        for after_file in self.after_files:
            base_name = after_file[:-9]  # "_last.STL"を除去
            before_file = base_name + "_first.STL"
            if before_file in self.before_files:
                self.file_pairs.append((after_file, before_file))
        
        if not self.file_pairs:
            raise ValueError("対応するファイルペアが見つかりません")
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx: int):
        after_file, before_file = self.file_pairs[idx]
        after_path = os.path.join(self.after_dir, after_file)
        before_path = os.path.join(self.before_dir, before_file)
        
        # ファイルの読み込み
        after_nodes, before_nodes, faces = STLFileLoader.load_file_pair(after_path, before_path)

        # 座標データのみを抽出 (x, y, z)
        after_coords = after_nodes[:, 1:].float()
        before_coords = before_nodes[:, 1:].float()
        
        return after_coords, before_coords, faces

class NodeMLP(nn.Module):
    """節点座標を予測するための多層パーセプトロンモデル"""

    def __init__(self, num_nodes: int, hidden_dims: List[int]):
        super().__init__()

        layers = []
        input_dim = num_nodes * 3  # 全節点のxyz座標
        
        # 隠れ層の構築
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            input_dim = hidden_dim
        
        # 出力層
        layers.append(nn.Linear(input_dim, num_nodes * 3))
        
        self.model = nn.Sequential(*layers)
        self.num_nodes = num_nodes
    
    def forward(self, x: torch.Tensor):
        """順伝播の実装"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.model(x)

class EarlyStopping:
    """検証損失の監視による早期学習停止の制御"""

    def __init__(self, cfg: CFG):
        if cfg.early_stopping_patience < 0:
            raise ValueError("patienceは0以上の値である必要があります")
        if cfg.early_stopping_min_delta < 0:
            raise ValueError("min_deltaは0以上の値である必要があります")
        if cfg.early_stopping_min_epochs < 0:
            raise ValueError("min_epochsは0以上の値である必要があります")

        self.patience = cfg.early_stopping_patience
        self.min_delta = cfg.early_stopping_min_delta
        self.min_epochs = cfg.early_stopping_min_epochs
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0

    def should_stop(self, val_loss: float, epoch: int):
        """学習を停止すべきかを判断"""
        if epoch < self.min_epochs:
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
        
        return self.counter >= self.patience

def save_logs(train_losses: List[float], val_losses: List[float], training_time: float, save_path: str):
    """学習履歴をJSONファイルとして保存"""
    logs = {
        "train_losses": train_losses,
        "val_losses": val_losses,
         "training_time_seconds": training_time,
        "training_time_formatted": f"{training_time//3600:.0f}時間{(training_time%3600)//60:.0f}分{training_time%60:.0f}秒"
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(logs, f, indent=4)

def plot_learning_curve(train_losses: List[float], val_losses: List[float], save_path: str):
    """学習曲線をプロットして保存"""
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

class ModelTrainer:
    """MLPモデルの学習を管理するクラス"""

    def __init__(
        self,
        model: nn.Module,
        cfg: CFG,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.cfg = cfg
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_dir: Optional[str] = None
    ):
        """モデルの学習を実行"""
        start_time = time.time()
        training_time = 0.0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        early_stopper = EarlyStopping(self.cfg)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.cfg.num_epochs):
            # 訓練フェーズ
            train_loss = self._train_epoch(train_loader, optimizer)
            train_losses.append(train_loss)
            
            # 検証フェーズ
            val_loss = self._validate_epoch(val_loader)
            val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{self.cfg.num_epochs}:")
            print(f"  Training Loss: {train_loss:.6f}")
            print(f"  Validation Loss: {val_loss:.6f}")
            
            if early_stopper.should_stop(val_loss, epoch):
                print(f"Early stopping at epoch {epoch+1}")
                print(f"Best validation loss was {early_stopper.best_loss:.6f} at epoch {early_stopper.best_epoch+1}")
                break
        
        if model_dir:
            training_time = time.time() - start_time
            self._save_training_results(model_dir, train_losses, val_losses, training_time)
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "training_time": training_time
        }
    
    def _train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer):
        """1エポックの訓練を実行"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for after_coords, before_coords, _ in tqdm(train_loader, desc="Training"):
            after_coords = after_coords.to(self.device)
            before_coords = before_coords.to(self.device)
            
            optimizer.zero_grad()
            pred_coords = self.model(after_coords)
            before_coords_flat = before_coords.view(before_coords.size(0), -1)
            loss = self.criterion(pred_coords, before_coords_flat)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(self, val_loader: DataLoader):
        """1エポックの検証を実行"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for after_coords, before_coords, _ in val_loader:
                after_coords = after_coords.to(self.device)
                before_coords = before_coords.to(self.device)
                
                pred_coords = self.model(after_coords)
                before_coords_flat = before_coords.view(before_coords.size(0), -1)
                loss = self.criterion(pred_coords, before_coords_flat)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _save_training_results(self, model_dir: str, train_losses: List[float], val_losses: List[float], training_time: float):
        """学習結果の保存"""
        os.makedirs(model_dir, exist_ok=True)
        
        # 学習ログの保存
        save_logs(
            train_losses=train_losses,
            val_losses=val_losses,
            training_time=training_time,
            save_path=os.path.join(model_dir, "training_logs.json")
        )
        
        # 学習曲線のプロット
        plot_learning_curve(
            train_losses=train_losses,
            val_losses=val_losses,
            save_path=os.path.join(model_dir, "learning_curve.png")
        )
        
        # モデルの保存
        torch.save(self.model.state_dict(), os.path.join(model_dir, "node_mlp.pth"))

def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description="MLPモデルの学習スクリプト")
    
    # 実験設定
    parser.add_argument("--experiment_name", type=str, required=True, help="実験名")
    parser.add_argument("--output_dir", type=str, default=None, help="出力ディレクトリのパス")
    
    # モデルパラメータ
    parser.add_argument("--hidden_dims", type=str, default="[256, 512, 256]", help="隠れ層のユニット数")
    
    # 学習パラメータ
    parser.add_argument("--batch_size", type=int, default=1, help="バッチサイズ")
    parser.add_argument("--num_epochs", type=int, default=100, help="エポック数")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="学習率")
    
    # 早期終了の設定
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="早期終了の待機エポック数")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0001, help="早期終了の最小変化量")
    parser.add_argument("--early_stopping_min_epochs", type=int, default=0, help="最低実行エポック数")
    
    # データ分割設定
    parser.add_argument("--train_val_split_ratio", type=float, default=0.8, help="訓練:検証データの分割比率")
    
    # データディレクトリ
    parser.add_argument("--after_dir", type=str, required=True, help="変形後のSTLファイルが格納されているディレクトリパス")
    parser.add_argument("--before_dir", type=str, required=True, help="変形前のSTLファイルが格納されているディレクトリパス")

    # seed値の設定
    parser.add_argument("--seed", type=int, default=42, help="乱数シード値の設定")
    
    args = parser.parse_args()
    
    # hidden_dimsの文字列をリストに変換
    try:
        args.hidden_dims = ast.literal_eval(args.hidden_dims)
        if not isinstance(args.hidden_dims, list):
            raise ValueError("hidden_dimsはリスト形式で指定してください")
    except:
        raise ValueError("hidden_dimsの形式が不正です。例: [256, 512, 256]")
    
    return args

def main():
    """メイン処理"""
    # コマンドライン引数の解析
    args = parse_args()
    
    # 設定の初期化
    cfg = CFG(
        experiment_name=args.experiment_name,
        output_dir=Path(args.output_dir) if args.output_dir else Path(__file__).parent.parent / "experiments",
        hidden_dims=args.hidden_dims,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_min_epochs=args.early_stopping_min_epochs,
        train_val_split_ratio=args.train_val_split_ratio,
        seed=args.seed
    )
    cfg._set_seed()
    
    # データセットの作成
    dataset = AfterBeforeDataset(
        after_dir=args.after_dir,
        before_dir=args.before_dir
    )
    
    # サンプルデータから節点数を取得
    sample_after, _, _ = dataset[0]
    num_nodes = sample_after.size(0)
    
    # データの分割（訓練:検証）
    train_size = int(cfg.train_val_split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # DataLoaderの作成
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    # デバイスの設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # モデルの作成と学習
    model = NodeMLP(num_nodes=num_nodes, hidden_dims=cfg.hidden_dims)
    trainer = ModelTrainer(model, cfg, device)
    
    # 学習の実行
    print("学習を開始します...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        model_dir=str(cfg.model_dir)
    )
    
    print("\n学習が完了しました。")
    print(f"最終訓練損失: {history['train_losses'][-1]:.6f}")
    print(f"最終検証損失: {history['val_losses'][-1]:.6f}")
    training_time = history['training_time']
    print(f"学習時間: {training_time//3600:.0f}時間{(training_time%3600)//60:.0f}分{training_time%60:.0f}秒")
    print(f"\n結果は以下のディレクトリに保存されました：\n{cfg.experiment_dir}")

if __name__ == "__main__":
    main()
