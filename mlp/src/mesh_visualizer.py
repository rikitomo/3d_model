import pyvista as pv
import numpy as np
from k_file_loader import KFileLoader
from tqdm import tqdm

def get_part_info(elements):
    """パーツ情報を取得"""
    pid_counts = {}
    for element in elements:
        pid = int(element[1])
        pid_counts[pid] = pid_counts.get(pid, 0) + 1
    return pid_counts

def visualize_k_file(file_path: str, pid_filter: list = None, show_edges: bool = True, debug: bool = False):
    """
    Kファイルのメッシュを可視化する

    Args:
        file_path (str): 可視化する.kファイルのパス
        pid_filter (list): 表示するパーツのPID一覧。Noneの場合は全パーツを表示
        show_edges (bool): エッジを表示するかどうか
        debug (bool): デバッグ情報を出力するかどうか
    """
    def debug_print(*args, **kwargs):
        if debug:
            print(*args, **kwargs)
    
    # PyVistaのプロットテーマを設定
    pv.set_plot_theme('document')
    
    # Kファイルからデータを読み込む
    nodes, elements = KFileLoader.load_single_file(file_path)
    
    debug_print("メッシュデータを処理中...")
    
    # ノードIDとインデックスのマッピングを作成
    node_ids = nodes[:, 0].numpy()
    node_id_to_idx = {int(node_id): idx for idx, node_id in enumerate(node_ids)}
    
    debug_print(f"ノード数: {len(nodes)}")
    debug_print(f"要素数: {len(elements)}")
    
    # パーツ情報を取得
    pid_counts = get_part_info(elements)
    if debug:
        debug_print("\nパーツ情報:")
        for pid, count in pid_counts.items():
            debug_print(f"PID {pid}: {count}要素")
    
    if pid_filter is None:
        pid_filter = list(pid_counts.keys())
    
    # 座標データの準備
    points = nodes[:, 1:].numpy()  # [x, y, z]座標のみ抽出
    
    # 要素の接続情報を0ベースのインデックスに変換
    debug_print("\n要素の接続情報を変換中...")
    cells_list = []
    selected_elements = 0
    
    # tqdmの使用をデバッグモードに依存させる
    elements_iter = tqdm(elements) if debug else elements
    for element in elements_iter:
        pid = int(element[1])
        if pid in pid_filter:
            # 各要素のノードIDを0ベースのインデックスに変換
            node_indices = [node_id_to_idx[int(node_id)] for node_id in element[2:]]
            cells_list.append(node_indices)
            selected_elements += 1
    
    if not cells_list:
        debug_print(f"エラー: 指定されたPID {pid_filter} に該当する要素が見つかりません")
        return
    
    cells = np.array(cells_list)
    debug_print(f"\n選択された要素数: {selected_elements}")
    debug_print(f"Points shape: {points.shape}")
    debug_print(f"Cells shape: {cells.shape}")
    
    # UnstructuredGridを作成
    # VTKのセルタイプ:
    # 9 = VTK_QUAD = 四角形要素
    # cells: 要素の接続情報 [node1, node2, node3, node4] の配列
    # points: 節点座標 [x, y, z] の配列
    debug_print("メッシュを構築中...")
    debug_print("セル構造:")
    if debug and len(cells) > 0:
        debug_print(f"最初の要素の節点インデックス: {cells[0]}")
        debug_print(f"対応する節点座標:")
        for idx in cells[0]:
            debug_print(f"  Node {idx}: {points[idx]}")
    
    # UnstructuredGridオブジェクトの作成
    # - キー9はVTK_QUADを示し、四角形要素を表す
    # - cellsは各要素の節点接続情報
    # - pointsは全節点の座標
    grid = pv.UnstructuredGrid({
        9: cells  # VTK_QUAD (9) for quadrilateral elements
    }, points)
    
    # プロッタの作成
    plotter = pv.Plotter()
    
    # メッシュの表示
    plotter.add_mesh(
        grid,
        show_edges=show_edges,
        line_width=1,
        opacity=0.7,
        color='lightblue',
        smooth_shading=True,
        specular=0.5,
        ambient=0.3
    )
    
    # カメラ位置の設定
    plotter.view_isometric()
    
    # 軸の表示
    plotter.add_axes()
    
    # タイトルの設定
    title = f"Mesh Visualization: {file_path.split('/')[-1]}"
    if pid_filter != list(pid_counts.keys()):
        title += f"\nSelected PIDs: {pid_filter}"
    plotter.add_title(title)
    
    # インタラクティブな表示の開始
    plotter.show()

if __name__ == "__main__":
    # サンプルファイルの可視化
    file_path = "mlp/data/model_before/Manual-chair-geometry-1.k"
    
    # パーツ情報を取得
    _, elements = KFileLoader.load_single_file(file_path)
    pid_counts = get_part_info(elements)

    # 全てのパーツ情報を表示
    visualize_k_file(file_path, pid_filter=None, show_edges=True, debug=True)
    
    # # 要素数が多い順に最初の2つのパーツを表示
    # sorted_pids = sorted(pid_counts.items(), key=lambda x: x[1], reverse=True)
    # selected_pids = [pid for pid, _ in sorted_pids[:2]]
    # print(f"\n要素数の多い順に最初の2つのPIDを表示: {selected_pids}")
    
    # # デバッグ情報を表示してメッシュを可視化
    # visualize_k_file(file_path, pid_filter=selected_pids, show_edges=True, debug=True)
