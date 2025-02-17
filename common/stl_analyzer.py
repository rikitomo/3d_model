import os
import numpy as np
import pandas as pd
import pyvista as pv
from typing import Tuple, Dict, List, Optional

class STLAnalyzer:
    def __init__(self):
        """STLファイルを解析・評価・可視化するためのクラス"""
        self.plotter = pv.Plotter()
        self.mesh = None
        self.true_mesh = None
        self.pred_mesh = None
        self.true_coords = None
        self.pred_coords = None
        self.true_faces = None
        self.pred_faces = None
    
    def read_stl_file(self, filename: str) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """STLファイルからノードデータを読み取る"""
        # STLファイルを読み込む
        self.mesh = pv.read(filename)
        
        # メッシュの頂点と面データを取得
        vertices = self.mesh.points
        faces = self.mesh.faces.reshape(-1, 4)[:, 1:4]
        
        # ノードデータを作成
        nodes = {}
        for i, vertex in enumerate(vertices, start=0):
            nodes[i] = vertex
        
        return nodes, faces
    
    def calculate_dimensions(self, nodes: Dict[int, np.ndarray]) -> Tuple[float, float, float]:
        """ノードデータからXYZ方向の長さを計算する"""
        # 全頂点のXYZ座標を配列に変換
        vertices = np.array(list(nodes.values()))
        
        # 各方向の最小値と最大値を計算
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        
        # 各方向の長さを計算
        dimensions = max_coords - min_coords
        
        return tuple(dimensions)

    def load_stl_pair(self, true_path: str, pred_path: str):
        """評価用のSTLファイルペアを読み込む"""
        # STLファイルを読み込み
        self.true_mesh = pv.read(true_path)
        self.pred_mesh = pv.read(pred_path)
        
        # メッシュの頂点と面データを取得
        self.true_coords = self.true_mesh.points
        self.pred_coords = self.pred_mesh.points
        self.true_faces = self.true_mesh.faces.reshape(-1, 4)[:, 1:4]
        self.pred_faces = self.pred_mesh.faces.reshape(-1, 4)[:, 1:4]
    
    def calculate_axis_differences(self) -> np.ndarray:
        """xyz方向それぞれの差分を計算"""
        return self.true_coords - self.pred_coords
    
    def calculate_dimension_differences(self) -> np.ndarray:
        """モデルの縦横高さの差を計算"""
        true_dims = np.ptp(self.true_coords, axis=0)  # 各軸の最大値と最小値の差
        pred_dims = np.ptp(self.pred_coords, axis=0)
        return true_dims - pred_dims

    def calculate_normal_similarity(self) -> Tuple[float, float]:
        """法線ベクトルの類似度を計算"""
        def compute_face_normals(coords, faces):
            normals = []
            for face in faces:
                v0 = coords[face[0]]
                v1 = coords[face[1]]
                v2 = coords[face[2]]
                # 面の法線ベクトルを計算
                normal = np.cross(v1 - v0, v2 - v0)
                # 正規化
                normal = normal / np.linalg.norm(normal)
                normals.append(normal)
            return np.array(normals)

        # 両方の面の法線を計算
        true_normals = compute_face_normals(self.true_coords, self.true_faces)
        pred_normals = compute_face_normals(self.pred_coords, self.pred_faces)

        # 法線ベクトル間の角度を計算
        dot_products = np.abs(np.sum(true_normals[:, np.newaxis] * pred_normals, axis=2))
        angles = np.arccos(np.clip(dot_products, -1.0, 1.0))
        
        # 各面の最小角度を取得（最も近い向きの面との角度）
        min_angles = np.min(angles, axis=1)
        return np.mean(min_angles), np.max(min_angles)

    def calculate_corresponding_distances(self) -> np.ndarray:
        """対応する点同士の距離を計算"""
        return np.linalg.norm(self.true_coords - self.pred_coords, axis=1)
    
    def evaluate(self, true_path: str, pred_path: str) -> Dict:
        """2つのSTLファイルを評価"""
        self.load_stl_pair(true_path, pred_path)
        
        corresponding_dists = self.calculate_corresponding_distances()
        axis_diffs = self.calculate_axis_differences()
        dim_diffs = self.calculate_dimension_differences()
        mean_angle, max_angle = self.calculate_normal_similarity()
        
        metrics = {
            "X方向の寸法差": dim_diffs[0],
            "Y方向の寸法差": dim_diffs[1],
            "Z方向の寸法差": dim_diffs[2],
            "X方向の最大ズレ": np.max(np.abs(axis_diffs[:, 0])),
            "Y方向の最大ズレ": np.max(np.abs(axis_diffs[:, 1])),
            "Z方向の最大ズレ": np.max(np.abs(axis_diffs[:, 2])),
            "法線の平均角度差(rad)": mean_angle,
            "法線の最大角度差(rad)": max_angle,
            "対応点間の平均距離": np.mean(corresponding_dists),
            "対応点間の最大距離": np.max(corresponding_dists),
        }
        
        return metrics

    def visualize_mesh(self, show_edges: bool = True, color: str = "white", edge_color: str = "black", 
                      window_size: Tuple[int, int] = (1024, 768)):
        """メッシュを可視化する"""
        if self.mesh is None:
            raise ValueError("メッシュが読み込まれていません。read_stl_file()を先に実行してください。")
            
        self.plotter.add_mesh(self.mesh, show_edges=show_edges, color=color, edge_color=edge_color)
        self.plotter.show(window_size=window_size)

    def compare_overlaid_meshes(self, file1: str, file2: str, color1: str = "lightblue", color2: str = "lightgreen", 
                              opacity2: float = 0.5, show_edges: bool = True, window_size: Tuple[int, int] = (1024, 768)):
        """2つのSTLファイルを重ねて表示する"""
        # プロッターをリセット
        self.plotter = pv.Plotter()
        
        # 1つ目のメッシュを読み込んで表示
        mesh1 = pv.read(file1)
        self.plotter.add_mesh(mesh1, show_edges=show_edges, color=color1)
        
        # 2つ目のメッシュを読み込んで重ねて表示（半透明）
        mesh2 = pv.read(file2)
        self.plotter.add_mesh(mesh2, show_edges=show_edges, color=color2, opacity=opacity2)
        
        # 表示
        self.plotter.show(window_size=window_size)

def analyze_stl_files_in_directory(directory_path: str, contains_word: str = None) -> pd.DataFrame:
    """指定したディレクトリ内のSTLファイルの寸法を分析"""
    # 分析結果を格納するリスト
    results = []
    
    # STL分析器を初期化
    analyzer = STLAnalyzer()
    
    # ディレクトリ内のSTLファイルを検索
    for filename in os.listdir(directory_path):
        if filename.lower().endswith('.stl'):
            # 指定した単語を含むファイルのみを対象とする
            if contains_word is not None and contains_word not in filename:
                continue
            
            file_path = os.path.join(directory_path, filename)
            
            # STLファイルを読み込んで寸法を計算
            nodes, _ = analyzer.read_stl_file(file_path)
            x_length, y_length, z_length = analyzer.calculate_dimensions(nodes)
            
            # 結果を辞書として保存
            results.append({
                'filename': filename,
                'x_length': x_length,
                'y_length': y_length,
                'z_length': z_length
            })
    
    return pd.DataFrame(results)

def evaluate_files_in_directory(true_dir: str, pred_dir: str, contains_word: str = None) -> Optional[pd.DataFrame]:
    """指定したディレクトリ内のSTLファイルペアを評価"""
    # filenameに特定の文字列が含まれるもののみを対象とする
    def filter_files(files, word):
        if word is None:
            return files
        return {name: path for name, path in files.items() if word in os.path.basename(path)}
    
    # 正解ファイルの一覧を取得
    true_files = {
        os.path.splitext(f)[0]: os.path.join(true_dir, f)
        for f in os.listdir(true_dir)
        if f.lower().endswith(".stl")
    }
    true_files = filter_files(true_files, contains_word)
    
    # 予測ファイルの一覧を取得
    pred_files = {
        os.path.splitext(f)[0]: os.path.join(pred_dir, f)
        for f in os.listdir(pred_dir)
        if f.lower().endswith(".stl")
    }
    pred_files = filter_files(pred_files, contains_word)
    pred_files = {name.replace("predicted_", ""): path for name, path in pred_files.items()}
    
    # 共通のファイルペアを見つける
    common_names = set(true_files.keys()) & set(pred_files.keys())
    file_pairs = [(true_files[name], pred_files[name]) for name in common_names]
    
    # 評価の実行
    analyzer = STLAnalyzer()
    all_results = []
    
    for true_path, pred_path in file_pairs:
        metrics = analyzer.evaluate(true_path, pred_path)
        metrics["filename"] = os.path.basename(true_path)
        all_results.append(metrics)
    
    if not all_results:
        return None
        
    df = pd.DataFrame(all_results)
    cols = ["filename"] + [col for col in df.columns if col != "filename"]
    return df[cols]
