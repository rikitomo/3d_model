import os
import pyvista as pv
import numpy as np
from stl import mesh
from typing import Optional, Tuple, List
from stl_file_loader import STLFileLoader

class STLVisualizer:
    def __init__(self):
        """STLファイルを可視化するためのクラス"""
        self.plotter = pv.Plotter()
        self.mesh = None
        
    def load_stl(self, filename: str) -> None:
        """
        STLファイルを読み込んで表示用のメッシュを作成

        Args:
            filename (str): STLファイルのパス
        """
        # PyVistaでSTLファイルを直接読み込む
        self.mesh = pv.read(filename)
        
    def visualize(self, show_edges: bool = True, color: str = 'white',
                 edge_color: str = 'black', window_size: Tuple[int, int] = (1024, 768)) -> None:
        """
        メッシュを可視化

        Args:
            show_edges (bool): エッジを表示するかどうか
            color (str): メッシュの色
            edge_color (str): エッジの色
            window_size (Tuple[int, int]): ウィンドウサイズ
        """
        if self.mesh is None:
            raise ValueError("メッシュが読み込まれていません。load_stl()を先に実行してください。")
            
        self.plotter.add_mesh(self.mesh, show_edges=show_edges, color=color, edge_color=edge_color)
        self.plotter.show(window_size=window_size)
        
    def compare_meshes(self, input_file: str, output_file: str,
                      show_edges: bool = True, window_size: Tuple[int, int] = (1024, 768)) -> None:
        """
        入力メッシュと出力メッシュを並べて表示

        Args:
            input_file (str): 入力STLファイルのパス
            output_file (str): 出力STLファイルのパス
            show_edges (bool): エッジを表示するかどうか
            window_size (Tuple[int, int]): ウィンドウサイズ
        """
        # 2つのビューを持つプロッターを作成
        plotter = pv.Plotter(shape=(1, 2))
        
        # 入力メッシュを左側に表示
        input_mesh = pv.read(input_file)
        plotter.subplot(0, 0)
        plotter.add_mesh(input_mesh, show_edges=show_edges, color='lightblue')
        plotter.add_text("Mesh1", position='upper_edge')
        
        # 出力メッシュを右側に表示
        output_mesh = pv.read(output_file)
        plotter.subplot(0, 1)
        plotter.add_mesh(output_mesh, show_edges=show_edges, color='lightgreen')
        plotter.add_text("Mesh2", position='upper_edge')
        
        # カメラ位置を同期
        plotter.link_views()
        
        # 表示
        plotter.show(window_size=window_size)
    
    @staticmethod
    def create_animation(input_file: str, output_file: str, output_path: str,
                        n_frames: int = 50) -> None:
        """
        入力メッシュから出力メッシュへの変形アニメーションを作成

        Args:
            input_file (str): 入力STLファイルのパス
            output_file (str): 出力STLファイルのパス
            output_path (str): アニメーションGIFの出力パス
            n_frames (int): アニメーションのフレーム数
        """
        # 入力と出力のメッシュを読み込む
        input_mesh = pv.read(input_file)
        output_mesh = pv.read(output_file)
        
        # プロッターの設定
        plotter = pv.Plotter()
        plotter.open_gif(output_path)
        
        # フレームごとに中間形状を計算して表示
        for i in range(n_frames):
            t = i / (n_frames - 1)  # 0から1までの補間パラメータ
            
            # 頂点座標を線形補間
            vertices = (1 - t) * input_mesh.points + t * output_mesh.points
            intermediate_mesh = pv.PolyData(vertices, faces=input_mesh.faces)
            
            plotter.clear()
            plotter.add_mesh(intermediate_mesh, show_edges=True, color='white')
            plotter.write_frame()
        
        plotter.close()

if __name__ == "__main__":
    # 使用例
    try:
        input_file = "mlp/data/stl_data/eyeball_last.stl"
        output_file = "mlp/data/stl_data/eyeball_first.stl"
        
        # 単一メッシュの可視化
        visualizer = STLVisualizer()
        visualizer.load_stl(input_file)
        visualizer.visualize(show_edges=True, color='lightblue')
        
        # メッシュの比較
        visualizer.compare_meshes(input_file, output_file)
        
        # アニメーションの作成
        # animation_path = "mesh_animation.gif"
        # STLVisualizer.create_animation(input_file, output_file, animation_path)
        # print(f"アニメーションを保存しました: {animation_path}")
        
    except FileNotFoundError:
        print("ファイルが見つかりません。実際のSTLファイルのパスを指定してください。")
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
