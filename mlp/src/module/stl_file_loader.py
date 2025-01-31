import os
import random
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from stl import mesh


class STLFileLoader:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        batch_size: int = 1,
        shuffle: bool = False,
        input_suffix: str = "_last.stl",
        output_suffix: str = "_first.stl"
    ):
        """
        Args:
            input_dir (str): 入力STLファイルが格納されているディレクトリのパス
            output_dir (str): 出力STLファイルが格納されているディレクトリのパス
            batch_size (int): バッチサイズ
            shuffle (bool): データをシャッフルするかどうか
            input_suffix (str): 入力ファイルの接尾辞
            output_suffix (str): 出力ファイルの接尾辞
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.input_suffix = input_suffix
        self.output_suffix = output_suffix
        
        # 入力/出力ペアのファイルリストを取得
        self.file_pairs = self._get_stl_file_pairs()
        self.num_pairs = len(self.file_pairs)
        
        # イテレーション用の変数
        self.current_index = 0
        
        if self.num_pairs == 0:
            raise ValueError(f"No matching STL file pairs found in {input_dir} and {output_dir}")
            
        # エポック開始時にシャッフル
        if self.shuffle:
            random.shuffle(self.file_pairs)

    @staticmethod
    def read_stl_file(filename: str) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """
        STLファイルからノードデータを読み取る

        Args:
            filename (str): 読み取るSTLファイルのパス

        Returns:
            Tuple[Dict[int, np.ndarray], np.ndarray]: 
                - ノードデータ: {node_id: np.array([x, y, z])}
                - メッシュデータ: 元のメッシュの面情報 (N, 3, 3)
        """
        # STLファイルを読み込む
        stl_mesh = mesh.Mesh.from_file(filename)
        
        # メッシュデータを保持
        mesh_data = stl_mesh.vectors.copy()
        
        # 頂点データを抽出し、重複を除去
        vertices = mesh_data.reshape(-1, 3)
        unique_vertices, _ = np.unique(vertices, axis=0, return_inverse=True)
        
        # ノードデータを作成
        nodes = {}
        for i, vertex in enumerate(unique_vertices, start=1):
            nodes[i] = vertex
        
        return nodes, mesh_data

    def _get_stl_file_pairs(self) -> List[Tuple[str, str]]:
        """入力/出力のSTLファイルペアのリストを取得"""
        file_pairs = []
        input_files = set()
        output_files = set()
        
        # 入力ファイルの取得
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith(self.input_suffix.lower()):
                    input_files.add(os.path.join(root, file))
        
        # 出力ファイルの取得
        for root, _, files in os.walk(self.output_dir):
            for file in files:
                if file.lower().endswith(self.output_suffix.lower()):
                    output_files.add(os.path.join(root, file))
        
        # ペアの作成
        for input_file in input_files:
            base_name = os.path.basename(input_file)[:-len(self.input_suffix)]
            output_file = os.path.join(self.output_dir, base_name + self.output_suffix)
            
            if output_file in output_files:
                file_pairs.append((input_file, output_file))
        
        return file_pairs

    def __len__(self) -> int:
        """データローダーの長さを返す（バッチ数）"""
        return (self.num_pairs + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """イテレータとして自身を返す"""
        return self

    def __next__(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """次のバッチを返す

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]:
                - 入力ノードの座標データのリスト [Tensor(N, 4) [node_id, x, y, z], ...]
                - 出力ノードの座標データのリスト [Tensor(N, 4) [node_id, x, y, z], ...]
                各リストはバッチサイズと同じ長さで、各要素は個別のファイルから読み込まれたデータ
        """
        if self.current_index >= self.num_pairs:
            # エポック終了時
            if self.shuffle:
                random.shuffle(self.file_pairs)
            self.current_index = 0
            raise StopIteration

        batch_input_nodes = []  # バッチ内の各ファイルの入力ノード
        batch_output_nodes = []  # バッチ内の各ファイルの出力ノード
        
        batch_pairs = self.file_pairs[self.current_index:self.current_index + self.batch_size]
        
        for input_path, output_path in batch_pairs:
            print(f"Reading files: {input_path} -> {output_path}")  # デバッグ用
            
            # 入力STLの読み込み
            input_nodes = self.read_stl_file(input_path)
            if not input_nodes:
                print(f"Warning: Input file {input_path} has no valid data")
                continue
                
            # 出力STLの読み込み
            output_nodes = self.read_stl_file(output_path)
            if not output_nodes:
                print(f"Warning: Output file {output_path} has no valid data")
                continue
            
            # 入力ノードデータの処理
            input_nodes_list = []
            for node_id, coords in input_nodes.items():
                input_nodes_list.append([node_id] + list(coords))
            
            # 出力ノードデータの処理
            output_nodes_list = []
            for node_id, coords in output_nodes.items():
                output_nodes_list.append([node_id] + list(coords))
            
            # Tensorに変換してリストに追加
            if input_nodes_list and output_nodes_list:
                batch_input_nodes.append(torch.tensor(input_nodes_list, dtype=torch.float32))
                batch_output_nodes.append(torch.tensor(output_nodes_list, dtype=torch.float32))
        
        self.current_index += self.batch_size
        
        if not batch_input_nodes or not batch_output_nodes:
            return self.__next__()
            
        return batch_input_nodes, batch_output_nodes

    def reset(self):
        """イテレーションをリセット"""
        self.current_index = 0
        if self.shuffle:
            random.shuffle(self.file_pairs)

    @classmethod
    def load_file_pair(cls, input_path: str, output_path: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """単一の入力/出力STLファイルペアを読み込んでTensor形式で返す

        Args:
            input_path (str): 入力STLファイルのパス
            output_path (str): 出力STLファイルのパス

        Returns:
            Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
                - 入力ノードの座標データ (N, 4) [node_id, x, y, z]
                - 出力ノードの座標データ (N, 4) [node_id, x, y, z]
                - メッシュデータ (M, 3, 3) 三角形面の頂点座標
        """
        # 入力STLの読み込み
        input_nodes, mesh_data = cls.read_stl_file(input_path)
        input_nodes_list = []
        for node_id, coords in input_nodes.items():
            input_nodes_list.append([node_id] + list(coords))
        
        # 出力STLの読み込み
        output_nodes, _ = cls.read_stl_file(output_path)
        output_nodes_list = []
        for node_id, coords in output_nodes.items():
            output_nodes_list.append([node_id] + list(coords))
        
        # Tensorに変換
        input_tensor = torch.tensor(input_nodes_list, dtype=torch.float32)
        output_tensor = torch.tensor(output_nodes_list, dtype=torch.float32)
        
        return input_tensor, output_tensor, mesh_data

    @staticmethod
    def save_to_stl(nodes: torch.Tensor, mesh_data: np.ndarray, output_path: str):
        """ノードデータとメッシュ構造からSTLファイルを保存する

        Args:
            nodes (torch.Tensor): ノードデータ (N, 4) [node_id, x, y, z]
            mesh_data (np.ndarray): 元のメッシュ構造 (M, 3, 3)
            output_path (str): 出力STLファイルのパス
        """
        # ノードの座標データを抽出 (x, y, z)
        coords = nodes[:, 1:].numpy()
        
        # 元のメッシュ構造の各頂点を、最も近い新しい頂点で置き換える
        updated_mesh = mesh_data.copy()
        mesh_vertices = updated_mesh.reshape(-1, 3)
        
        # 各メッシュ頂点に対して最も近い新しい頂点を見つける
        for i, vertex in enumerate(mesh_vertices):
            # ユークリッド距離を計算
            distances = np.linalg.norm(coords - vertex, axis=1)
            # 最も近い頂点のインデックスを取得
            nearest_idx = np.argmin(distances)
            # 頂点を更新
            mesh_vertices[i] = coords[nearest_idx]
        
        # メッシュ頂点を元の形状に戻す
        updated_mesh = mesh_vertices.reshape(-1, 3, 3)
        
        # メッシュオブジェクトの作成
        stl_mesh = mesh.Mesh(np.zeros(len(updated_mesh), dtype=mesh.Mesh.dtype))
        stl_mesh.vectors = updated_mesh
        
        # STLファイルとして保存
        stl_mesh.save(output_path)


if __name__ == "__main__":
    
    # ファイルペアの読み込み例
    print("\n=== STLファイルペアの読み込み ===")
    input_file = "mlp/data/stl_data/eyeball_last.stl"
    output_file = "mlp/data/stl_data/eyeball_first.stl"
    
    try:
        input_nodes, output_nodes = STLFileLoader.load_file_pair(input_file, output_file)
        print(f"Input nodes shape: {input_nodes.shape}")
        print(f"Output nodes shape: {output_nodes.shape}")
        print("\nFirst few input nodes (node_id, x, y, z):")
        for node in input_nodes[:3]:
            print(f"Node {int(node[0])}: ({node[1]:.4f}, {node[2]:.4f}, {node[3]:.4f})")
        print("\nFirst few output nodes (node_id, x, y, z):")
        for node in output_nodes[:3]:
            print(f"Node {int(node[0])}: ({node[1]:.4f}, {node[2]:.4f}, {node[3]:.4f})")
            
        # バッチ処理の例
        print("\n=== バッチ処理の例 ===")
        input_dir = "mlp/data/stl_data/"
        output_dir = "mlp/data/stl_data/"
        
        loader = STLFileLoader(
            input_dir=input_dir,
            output_dir=output_dir,
            batch_size=2,
            shuffle=True,
            input_suffix="_last.stl",
            output_suffix="_first.stl"
        )
        
        print(f"Total batches: {len(loader)}")
        for i, (batch_inputs, batch_outputs) in enumerate(loader):
            print(f"\nBatch {i+1}:")
            print(f"Number of file pairs in batch: {len(batch_inputs)}")
            for j, (inputs, outputs) in enumerate(zip(batch_inputs, batch_outputs)):
                print(f"\nFile pair {j+1} in batch:")
                print(f"  Input nodes shape: {inputs.shape}")
                print(f"  Output nodes shape: {outputs.shape}")
    
    except FileNotFoundError:
        print(f"ファイルが見つかりません")
        print("テストを実行するには、実際のSTLファイルのパスを指定してください。")
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
