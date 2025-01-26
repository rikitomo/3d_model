import os
import random
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional


class KFileLoader:
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1,
        shuffle: bool = False,
        file_pattern: str = "*.k"
    ):
        """
        Args:
            data_dir (str): .kファイルが格納されているディレクトリのパス
            batch_size (int): バッチサイズ
            shuffle (bool): データをシャッフルするかどうか
            file_pattern (str): 読み込むファイルのパターン（デフォルトは"*.k"）
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.file_pattern = file_pattern
        
        # .kファイルのリストを取得
        self.file_list = self._get_k_files()
        self.num_files = len(self.file_list)
        
        # イテレーション用の変数
        self.current_index = 0
        
        if self.num_files == 0:
            raise ValueError(f"No .k files found in {data_dir}")
            
        # エポック開始時にシャッフル
        if self.shuffle:
            random.shuffle(self.file_list)

    @staticmethod
    def read_key_file(filename: str) -> Tuple[Dict[int, np.ndarray], Dict[int, Tuple[int, List[int]]]]:
        """
        .kファイルからノードとエレメントデータを読み取る

        Args:
            filename (str): 読み取る.kファイルのパス

        Returns:
            Tuple[Dict[int, np.ndarray], Dict[int, Tuple[int, List[int]]]]:
                - ノードデータ: {node_id: np.array([x, y, z, ...])}
                - エレメントデータ: {element_id: (pid, [node_id1, node_id2, node_id3, node_id4])}
        """
        nodes = {}
        elements = {}
        
        with open(filename, "r") as f:
            lines = f.readlines()
        
        # NODEとELEMENT_SHELLの範囲を読み取る
        is_node = False
        is_element = False
        
        for line in lines:
            line = line.strip()
            
            # キーワードの判定
            if line.startswith("*NODE"):
                is_node = True
                is_element = False
                continue
            elif line.startswith("*ELEMENT_SHELL"):
                is_node = False
                is_element = True
                continue
            elif line.startswith("*"):
                is_node = False
                is_element = False
                continue
                
            # 空行またはコメント行をスキップ
            if not line or line.startswith("$"):
                continue
                
            # データの処理
            items = line.split()
            if is_node and len(items) >= 4:
                try:
                    node_id = int(items[0])
                    data = [float(x) for x in items[1:]]
                    nodes[node_id] = np.array(data)
                except (ValueError, IndexError):
                    continue
                    
            elif is_element and len(items) >= 5:
                try:
                    element_id = int(items[0])
                    pid = int(items[1])
                    node_ids = [int(x) for x in items[2:6]]  # シェル要素は4節点を想定
                    elements[element_id] = (pid, node_ids)
                except (ValueError, IndexError):
                    continue
        
        return nodes, elements

    def _get_k_files(self) -> List[str]:
        """ディレクトリ内の.kファイルのリストを取得"""
        k_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".k"):
                    k_files.append(os.path.join(root, file))
        return k_files

    def __len__(self) -> int:
        """データローダーの長さを返す（バッチ数）"""
        return (self.num_files + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """イテレータとして自身を返す"""
        return self

    def __next__(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """次のバッチを返す

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: 
                - ノードの座標データのリスト [Tensor(N, 4) [node_id, x, y, z], ...]
                - 要素の接続データのリスト [Tensor(M, 6) [element_id, pid, node1, node2, node3, node4], ...]
                各リストの長さはバッチサイズと同じで、各要素は個別のファイルから読み込まれたデータ
        
        Raises:
            StopIteration: バッチの終端に達した場合
            ValueError: 読み込んだファイルにノードまたはエレメントデータがない場合
        """
        if self.current_index >= self.num_files:
            # エポック終了時
            if self.shuffle:
                random.shuffle(self.file_list)
            self.current_index = 0
            raise StopIteration

        batch_nodes = []  # バッチ内の各ファイルのノード
        batch_elements = []  # バッチ内の各ファイルのエレメント
        
        batch_files = self.file_list[self.current_index:self.current_index + self.batch_size]
        
        for file_path in batch_files:
            print(f"Reading file: {file_path}")  # デバッグ用
            nodes, elements = self.read_key_file(file_path)
            
            if not nodes and not elements:
                print(f"Warning: File {file_path} has no valid data")  # デバッグ用
                print(f"Nodes: {len(nodes)}, Elements: {len(elements)}")  # デバッグ用
                continue
            
            # このファイルのノードデータを処理
            file_nodes = []
            for node_id, coords in nodes.items():
                # 座標データを3次元に統一（x, y, z）
                coords_3d = coords[:3]
                file_nodes.append([node_id] + list(coords_3d))
            
            # このファイルのエレメントデータを処理
            file_elements = []
            for element_id, (pid, node_ids) in elements.items():
                file_elements.append([element_id, pid] + node_ids)
            
            # 個別のファイルのデータをTensorに変換してリストに追加
            if file_nodes:
                batch_nodes.append(torch.tensor(file_nodes, dtype=torch.float32))
            if file_elements:
                batch_elements.append(torch.tensor(file_elements, dtype=torch.long))
        
        self.current_index += self.batch_size
        
        if not batch_nodes and not batch_elements:
            return self.__next__()
            
        return batch_nodes, batch_elements

    def reset(self):
        """イテレーションをリセット"""
        self.current_index = 0
        if self.shuffle:
            random.shuffle(self.file_list)

    @classmethod
    def load_single_file(cls, file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """単一の.kファイルを読み込んでTensor形式で返す

        Args:
            file_path (str): 読み込む.kファイルのパス

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - ノードの座標データ (N, 4) [node_id, x, y, z]
                - 要素の接続データ (M, 6) [element_id, pid, node1, node2, node3, node4]
        """
        nodes, elements = cls.read_key_file(file_path)
        
        # ノードデータを処理
        nodes_list = []
        for node_id, coords in nodes.items():
            coords_3d = coords[:3]
            nodes_list.append([node_id] + list(coords_3d))
        
        # エレメントデータを処理
        elements_list = []
        for element_id, (pid, node_ids) in elements.items():
            elements_list.append([element_id, pid] + node_ids)
        
        # Tensorに変換
        nodes_tensor = torch.tensor(nodes_list, dtype=torch.float32)
        elements_tensor = torch.tensor(elements_list, dtype=torch.long)
        
        return nodes_tensor, elements_tensor


if __name__ == "__main__":
    # 単一ファイルの読み込み例
    print("=== 単一ファイルの読み込み ===")
    single_file = "mlp/data/model_before/Manual-chair-geometry-1.k"
    nodes, elements = KFileLoader.load_single_file(single_file)
    print(f"Nodes shape: {nodes.shape}")
    print(f"Elements shape: {elements.shape}")
    print("\nFirst few nodes (node_id, x, y, z):")
    for node in nodes[:3]:
        print(f"Node {int(node[0])}: ({node[1]:.4f}, {node[2]:.4f}, {node[3]:.4f})")
    
    # バッチ処理の例
    print("\n=== バッチ処理の例 ===")
    data_dir = "mlp/data/model_before"  # .kファイルが格納されているルートディレクトリ
    loader = KFileLoader(
        data_dir=data_dir,
        batch_size=2,
        shuffle=True
    )
    
    print(f"Total batches: {len(loader)}")
    for i, (batch_nodes, batch_elements) in enumerate(loader):
        print(f"\nBatch {i+1}:")
        print(f"Number of files in batch: {len(batch_nodes)}")
        for j, (nodes, elements) in enumerate(zip(batch_nodes, batch_elements)):
            print(f"\nFile {j+1} in batch:")
            print(f"  Nodes shape: {nodes.shape}")
            print(f"  Elements shape: {elements.shape}")
            print(f"  First few nodes (node_id, x, y, z):")
            for node in nodes[:3]:
                print(f"    Node {int(node[0])}: ({node[1]:.4f}, {node[2]:.4f}, {node[3]:.4f})")
            for element in elements[:6]:
                print(f"    Element {int(element[0])}: pid={int(element[1])}, nodes={element[2:]}")