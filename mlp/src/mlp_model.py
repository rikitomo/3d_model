import os
import json
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from module.stl_file_loader import STLFileLoader

@dataclass
class CFG:
    """実験設定を管理するクラス"""
    # 実験設定
    experiment_name: str
    output_dir: Path = Path(__file__).parent.parent / "experiments"
    
    # モデルパラメータ
    hidden_dims: List[int] = field(
        default_factory=lambda: [256, 512, 256]
    )  # 隠れ層のユニット数
    
    # 学習パラメータ
    batch_size: int = 1
    num_epochs: int = 100
    learning_rate: float = 0.001
    
    # 早期終了の設定
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0001
    
    # データ分割比率
    train_val_split_ratio: float = 0.8
    
    def __post_init__(self):
        """出力ディレクトリの初期化"""
        self.experiment_dir = self.output_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # モデルの保存ディレクトリ
        self.model_dir = self.experiment_dir / "models"
        self.model_dir.mkdir(exist_ok=True)

class AfterBeforeDataset(Dataset):
    """変形前後の3D形状データを管理するデータセット"""
    def __init__(self, after_dir: str, before_dir: str):
        """
        Args:
            after_dir: 変形後のモデルが格納されているディレクトリパス
            before_dir: 変形前のモデルが格納されているディレクトリパス
        
        Raises:
            FileNotFoundError: 指定されたディレクトリが存在しない場合
            ValueError: 対応するファイルペアが見つからない場合
        """
        if not os.path.exists(after_dir) or not os.path.exists(before_dir):
            raise FileNotFoundError("指定されたディレクトリが存在しません")

        self.after_dir = after_dir
        self.before_dir = before_dir
        
        # ファイルリストの取得
        self.after_files = sorted([f for f in os.listdir(after_dir) if f.endswith('_last.stl')])
        self.before_files = sorted([f for f in os.listdir(before_dir) if f.endswith('_first.stl')])
        
        # 対応するファイルのペアを確認
        self.file_pairs = []
        for after_file in self.after_files:
            base_name = after_file[:-9]  # '_last.stl'を除去
            before_file = base_name + '_first.stl'
            if before_file in self.before_files:
                self.file_pairs.append((after_file, before_file))
        
        if not self.file_pairs:
            raise ValueError("対応するファイルペアが見つかりません")
    
    def __len__(self) -> int:
        return len(self.file_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        指定されたインデックスのデータペアを返す

        Args:
            idx: データインデックス

        Returns:
            変形後の座標、変形前の座標、メッシュデータのタプル
        """
        after_file, before_file = self.file_pairs[idx]
        after_path = os.path.join(self.after_dir, after_file)
        before_path = os.path.join(self.before_dir, before_file)
        
        # ファイルの読み込み（メッシュ情報も含む）
        after_nodes, before_nodes, mesh_data = STLFileLoader.load_file_pair(after_path, before_path)
        
        # node_idでソート
        after_nodes = after_nodes[torch.argsort(after_nodes[:, 0])]
        before_nodes = before_nodes[torch.argsort(before_nodes[:, 0])]
        
        # 座標データのみを抽出 (x, y, z)
        after_coords = after_nodes[:, 1:].float()
        before_coords = before_nodes[:, 1:].float()
        
        return after_coords, before_coords, mesh_data

class NodeMLP(nn.Module):
    """節点座標を予測するための多層パーセプトロンモデル"""

    def __init__(self, num_nodes: int, hidden_dims: List[int]):
        """
        Args:
            num_nodes: モデルの節点数
            hidden_dims: 隠れ層のユニット数のリスト

        Raises:
            ValueError: hidden_dimsが空リストの場合
        """
        super().__init__()
        
        if not hidden_dims:
            raise ValueError("hidden_dimsは少なくとも1つの要素を含む必要があります")

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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播の実行

        Args:
            x: 入力テンソル (batch_size, num_nodes, 3)

        Returns:
            予測された座標テンソル (batch_size, num_nodes * 3)
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return self.model(x)

class EarlyStopping:
    """検証損失の監視による早期学習停止の制御"""

    def __init__(self, cfg: CFG):
        """
        Args:
            cfg: 設定オブジェクト

        Raises:
            ValueError: patienceまたはmin_deltaが負の値の場合
        """
        if cfg.early_stopping_patience < 0:
            raise ValueError("patienceは0以上の値である必要があります")
        if cfg.early_stopping_min_delta < 0:
            raise ValueError("min_deltaは0以上の値である必要があります")

        self.patience = cfg.early_stopping_patience
        self.min_delta = cfg.early_stopping_min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0

    def should_stop(self, val_loss: float, epoch: int) -> bool:
        """
        学習を停止すべきかを判断

        Args:
            val_loss: 現在の検証損失値
            epoch: 現在のエポック数

        Returns:
            学習を停止すべきかどうか
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
        
        return self.counter >= self.patience

def save_logs(train_losses: List[float], val_losses: List[float], save_path: str):
    """
    学習履歴をJSONファイルとして保存

    Args:
        train_losses: 訓練損失の履歴
        val_losses: 検証損失の履歴
        save_path: 保存先のファイルパス
    """
    logs = {
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(logs, f, indent=4)

def plot_learning_curve(train_losses: List[float], val_losses: List[float], save_path: str):
    """
    学習曲線をプロットして保存

    Args:
        train_losses: 訓練損失の履歴
        val_losses: 検証損失の履歴
        save_path: 保存先のファイルパス
    """
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
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: 学習するモデル
            cfg: 設定オブジェクト
            device: 使用するデバイス
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.cfg = cfg
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        モデルの学習を実行

        Args:
            train_loader: 訓練データのDataLoader
            val_loader: 検証データのDataLoader
            num_epochs: エポック数
            learning_rate: 学習率
            model_dir: モデルと学習ログを保存するディレクトリ

        Returns:
            学習履歴を含む辞書
            
        Raises:
            RuntimeError: GPUメモリ不足や学習の収束性に問題がある場合
        """
        try:
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
                
                print(f'Epoch {epoch+1}/{self.cfg.num_epochs}:')
                print(f'  Training Loss: {train_loss:.6f}')
                print(f'  Validation Loss: {val_loss:.6f}')
                
                if early_stopper.should_stop(val_loss, epoch):
                    print(f"Early stopping at epoch {epoch+1}")
                    print(f"Best validation loss was {early_stopper.best_loss:.6f} at epoch {early_stopper.best_epoch+1}")
                    break
            
            if model_dir:
                self._save_training_results(model_dir, train_losses, val_losses)
            
            return {"train_losses": train_losses, "val_losses": val_losses}
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError("GPUメモリが不足しています。バッチサイズを小さくするか、モデルを縮小してください。") from e
            raise
    
    def _train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        """1エポックの訓練を実行"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for after_coords, before_coords, _ in tqdm(train_loader, desc='Training'):
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
    
    def _validate_epoch(self, val_loader: DataLoader) -> float:
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
    
    def _save_training_results(self, model_dir: str, train_losses: List[float], val_losses: List[float]):
        """学習結果の保存"""
        os.makedirs(model_dir, exist_ok=True)
        
        # 学習ログの保存
        save_logs(
            train_losses=train_losses,
            val_losses=val_losses,
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

def main():
    """メイン処理の実行"""
    try:
        # 設定の初期化
        cfg = CFG(experiment_name="exp001")
        
        # データセットの作成
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset = AfterBeforeDataset(
            after_dir=os.path.join(current_dir, "data/stl_train"),
            before_dir=os.path.join(current_dir, "data/stl_train")
        )
        
        # サンプルデータから節点数を取得
        sample_after, _, _ = dataset[0]
        num_nodes = sample_after.size(0)
        
        # データの分割（訓練:検証 = 8:2）
        train_size = int(cfg.train_val_split_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # DataLoaderの作成
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
        
        # モデルの作成と学習
        model = NodeMLP(num_nodes=num_nodes, hidden_dims=cfg.hidden_dims)
        trainer = ModelTrainer(model, cfg)
        
        # 学習の実行
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            model_dir=str(cfg.model_dir)
        )
        
        print("学習が完了しました。")
        print(f"最終訓練損失: {history['train_losses'][-1]:.6f}")
        print(f"最終検証損失: {history['val_losses'][-1]:.6f}")
        
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        print("データディレクトリが存在することを確認してください。")
    except ValueError as e:
        print(f"エラー: {e}")
        print("データセットの設定を確認してください。")
    except RuntimeError as e:
        print(f"エラー: {e}")
        if "out of memory" in str(e):
            print("GPUメモリが不足しています。バッチサイズを小さくするか、モデルを縮小してください。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        raise

if __name__ == "__main__":
    main()
