import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from k_file_loader import KFileLoader
import numpy as np
from typing import Tuple, List
from tqdm import tqdm

class AfterBeforeDataset(Dataset):
    def __init__(self, after_dir: str, before_dir: str):
        """
        Args:
            after_dir (str): model_afterディレクトリのパス
            before_dir (str): model_beforeディレクトリのパス
        """
        self.after_dir = after_dir
        self.before_dir = before_dir
        
        # ファイルリストの取得
        self.after_files = sorted([f for f in os.listdir(after_dir) if f.endswith('.k')])
        self.before_files = sorted([f for f in os.listdir(before_dir) if f.endswith('.k')])
        
        # 対応するファイルのペアを確認
        self.file_pairs = []
        for after_file in self.after_files:
            if after_file in self.before_files:
                self.file_pairs.append(after_file)
    
    def __len__(self) -> int:
        return len(self.file_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filename = self.file_pairs[idx]
        after_path = os.path.join(self.after_dir, filename)
        before_path = os.path.join(self.before_dir, filename)
        
        # ファイルの読み込み
        after_nodes, _ = KFileLoader.load_single_file(after_path)
        before_nodes, _ = KFileLoader.load_single_file(before_path)
        
        # node_idでソート
        after_nodes = after_nodes[torch.argsort(after_nodes[:, 0])]
        before_nodes = before_nodes[torch.argsort(before_nodes[:, 0])]
        
        # 座標データのみを抽出 (x, y, z)
        after_coords = after_nodes[:, 1:].float()
        before_coords = before_nodes[:, 1:].float()
        
        return after_coords, before_coords

class NodeMLP(nn.Module):
    def __init__(self, hidden_dims: List[int] = [64, 128, 64]):
        """
        Args:
            hidden_dims (List[int]): 隠れ層のユニット数のリスト
        """
        super().__init__()
        
        layers = []
        input_dim = 3  # (x, y, z)
        
        # 隠れ層の構築
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim
        
        # 出力層
        layers.append(nn.Linear(input_dim, 3))  # 3次元座標を出力
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[List[float], List[float]]:
    """
    モデルの学習を行う

    Args:
        model: 学習するモデル
        train_loader: 訓練データのDataLoader
        val_loader: 検証データのDataLoader
        num_epochs: エポック数
        learning_rate: 学習率
        device: 使用するデバイス

    Returns:
        Tuple[List[float], List[float]]: 訓練損失と検証損失の履歴
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 訓練フェーズ
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for after_coords, before_coords in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            after_coords = after_coords.to(device)  # (batch_size, num_nodes, 3)
            before_coords = before_coords.to(device)
            
            batch_size, num_nodes, _ = after_coords.shape
            after_coords = after_coords.reshape(-1, 3)  # (batch_size * num_nodes, 3)
            before_coords = before_coords.reshape(-1, 3)
            
            optimizer.zero_grad()
            pred_coords = model(after_coords)  # (batch_size * num_nodes, 3)
            loss = criterion(pred_coords, before_coords)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # 検証フェーズ
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for after_coords, before_coords in val_loader:
                after_coords = after_coords.to(device)
                before_coords = before_coords.to(device)
                
                batch_size, num_nodes, _ = after_coords.shape
                after_coords = after_coords.reshape(-1, 3)
                before_coords = before_coords.reshape(-1, 3)
                
                pred_coords = model(after_coords)
                loss = criterion(pred_coords, before_coords)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Training Loss: {avg_train_loss:.6f}')
        print(f'  Validation Loss: {avg_val_loss:.6f}')
    
    return train_losses, val_losses

def predict_coordinates(
    model: nn.Module,
    after_coords: torch.Tensor,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> torch.Tensor:
    """
    model_afterの座標からmodel_beforeの座標を予測する

    Args:
        model: 学習済みモデル
        after_coords: 予測する座標データ (N, 3)
        device: 使用するデバイス

    Returns:
        torch.Tensor: 予測された座標 (N, 3)
    """
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        after_coords = after_coords.to(device)
        pred_coords = model(after_coords)
    
    return pred_coords.cpu()

if __name__ == "__main__":
    # データセットの作成
    dataset = AfterBeforeDataset(
        after_dir="data/model_after",
        before_dir="data/model_before"
    )
    
    # データの分割（訓練:検証 = 8:2）
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # DataLoaderの作成（バッチサイズを1に設定）
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # モデルの作成と学習
    model = NodeMLP()
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=100,
        learning_rate=0.001
    )
    
    # モデルの保存
    torch.save(model.state_dict(), 'models/node_mlp.pth')
