import os
import random
import torch
import numpy as np
import pyvista as pv


class STLFileLoader:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        batch_size: int = 1,
        shuffle: bool = False,
        input_suffix: str = "_last.STL",
        output_suffix: str = "_first.STL"
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
    
    def _get_stl_file_pairs(self):
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

    @staticmethod
    def read_stl_file(filename: str):
        """STLファイルからノードデータを読み取る"""
        # STLファイルを読み込む
        mesh = pv.read(filename)
        
        # メッシュの頂点と面データを取得
        vertices = mesh.points
        faces = mesh.faces.reshape(-1, 4)[:, 1:4]
        
        # ノードデータを作成（1からの連番でノード番号を付与）
        nodes = {}
        for i, vertex in enumerate(vertices, start=1):
            nodes[i] = vertex
        
        return nodes, faces
    
    @classmethod
    def load_file_pair(cls, input_path: str, output_path: str):
        """単一の入力/出力STLファイルペアを読み込んでTensor形式で返す"""
        # 入力STLの読み込み
        input_nodes, faces = cls.read_stl_file(input_path)
        input_nodes_list = []
        for node_id, coords in input_nodes.items():
            input_nodes_list.append([node_id] + list(coords))
        
        # 出力STLの読み込み（メッシュ構造は共通という前提）
        output_nodes, _ = cls.read_stl_file(output_path)
        output_nodes_list = []
        for node_id, coords in output_nodes.items():
            output_nodes_list.append([node_id] + list(coords))

        input_tensor = torch.tensor(input_nodes_list, dtype=torch.float32)
        output_tensor = torch.tensor(output_nodes_list, dtype=torch.float32)
        
        return input_tensor, output_tensor, faces
    
    @staticmethod
    def save_to_stl(nodes: torch.Tensor, mesh_data: np.ndarray, output_path: str):
        """ノードデータとメッシュ構造からSTLファイルを保存する"""
        # ノードの座標データを抽出 (x, y, z)
        coords = nodes[:, 1:].numpy()
        
        # メッシュ頂点をそのまま使用
        surf = pv.PolyData(coords, faces=np.insert(mesh_data, 0, 3, axis=1))
        
        # STLファイルとして保存
        surf.save(output_path)

    def __len__(self):
        """データローダーの長さを返す（バッチ数）"""
        return (self.num_pairs + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """イテレータとして自身を返す"""
        return self

    def __next__(self):
        """次のバッチを返す"""
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
            input_tensor, output_tensor, _ = self.load_file_pair(input_path, output_path)
            batch_input_nodes.append(input_tensor)
            batch_output_nodes.append(output_tensor)
        
        self.current_index += self.batch_size
        return batch_input_nodes, batch_output_nodes

    def reset(self):
        """イテレーションをリセット"""
        self.current_index = 0
        if self.shuffle:
            random.shuffle(self.file_pairs)


if __name__ == "__main__":
    # ファイルペアの読み込み例
    print("\n=== STLファイルペアの読み込み ===")
    input_file = "mlp/data/stl_train/sample_last.STL"
    output_file = "mlp/data/stl_train/sample_first.STL"
    
    input_nodes, output_nodes, mesh_data = STLFileLoader.load_file_pair(input_file, output_file)
    print(f"Input nodes shape: {input_nodes.shape}")
    print(f"Output nodes shape: {output_nodes.shape}")
    print("\nFirst few input nodes (node_id, x, y, z):")
    for node in input_nodes[:2]:
        print(f"Node {int(node[0])}: ({node[1]:.4f}, {node[2]:.4f}, {node[3]:.4f})")
    print("\nFirst few output nodes (node_id, x, y, z):")
    for node in output_nodes[:2]:
        print(f"Node {int(node[0])}: ({node[1]:.4f}, {node[2]:.4f}, {node[3]:.4f})")
    
    print("\nFirst few mesh data (faces):")
    print(mesh_data[:2])
    
    # バッチ処理の例
    print("\n=== バッチ処理の例 ===")
    input_dir = "mlp/data/stl_train/"
    output_dir = "mlp/data/stl_train/"
    
    loader = STLFileLoader(
        input_dir=input_dir,
        output_dir=output_dir,
        batch_size=2,
        shuffle=True,
        input_suffix="_last.STL",
        output_suffix="_first.STL"
    )
    
    print(f"Total batches: {len(loader)}")
    for i, (batch_inputs, batch_outputs) in enumerate(loader):
        print(f"\nBatch {i+1}:")
        print(f"Number of file pairs in batch: {len(batch_inputs)}")
        for j, (inputs, outputs) in enumerate(zip(batch_inputs, batch_outputs)):
            print(f"\nFile pair {j+1} in batch:")
            print(f"  Input nodes shape: {inputs.shape}")
            print(f"  Output nodes shape: {outputs.shape}")
    
    # メッシュの保存例
    print("\n=== STLファイルの保存 ===")
    output_test_file = "mlp/data/stl_train/_output.stl"
    STLFileLoader.save_to_stl(input_nodes, mesh_data, output_test_file)
    print(f"Saved mesh to: {output_test_file}")
