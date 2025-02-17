import numpy as np
import pyvista as pv
from typing import Tuple, Dict, List


class STLDimensionAnalyzer:
    def __init__(self):
        """STLファイルの寸法を解析し可視化するためのクラス"""
        self.plotter = pv.Plotter()
        self.mesh = None
        
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

    def visualize_mesh(self, show_edges: bool = True, color: str = "white", edge_color: str = "black", window_size: Tuple[int, int] = (1024, 768)):
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
