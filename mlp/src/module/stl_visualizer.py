import pyvista as pv
from typing import Tuple


class STLVisualizer:
    def __init__(self):
        """STLファイルを可視化するためのクラス"""
        self.plotter = pv.Plotter()
        self.mesh = None
        
    def load_stl(self, filename: str):
        """STLファイルを読み込んで表示用のメッシュを作成"""
        self.mesh = pv.read(filename)
        
    def visualize(self, show_edges: bool = True, color: str = "white", edge_color: str = "black", window_size: Tuple[int, int] = (1024, 768)):
        """メッシュを可視化"""
        if self.mesh is None:
            raise ValueError("メッシュが読み込まれていません。load_stl()を先に実行してください。")
            
        self.plotter.add_mesh(self.mesh, show_edges=show_edges, color=color, edge_color=edge_color)
        self.plotter.show(window_size=window_size)
        
    def compare_meshes(self, input_file: str, output_file: str, show_edges: bool = True, window_size: Tuple[int, int] = (1024, 768)):
        """入力メッシュと出力メッシュを並べて表示"""
        # 2つのビューを持つプロッターを作成
        plotter = pv.Plotter(shape=(1, 2))
        
        # 入力メッシュを左側に表示
        input_mesh = pv.read(input_file)
        plotter.subplot(0, 0)
        plotter.add_mesh(input_mesh, show_edges=show_edges, color="lightblue")
        
        # 出力メッシュを右側に表示
        output_mesh = pv.read(output_file)
        plotter.subplot(0, 1)
        plotter.add_mesh(output_mesh, show_edges=show_edges, color="lightgreen")
        
        # カメラ位置を同期
        plotter.link_views()
        
        # 表示
        plotter.show(window_size=window_size)

if __name__ == "__main__":
    input_file = "mlp/data/stl_train/sample_last.STL"
    output_file = "mlp/data/stl_train/sample_first.STL"
    
    # 単一メッシュの可視化
    visualizer = STLVisualizer()
    visualizer.load_stl(input_file)
    visualizer.visualize(show_edges=True, color="lightblue")
    
    # メッシュの比較
    visualizer.compare_meshes(input_file, output_file)
