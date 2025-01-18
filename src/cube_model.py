import numpy as np
from stl import mesh

def create_base_vertices(scale_x=1, scale_y=1, scale_z=1):
    """立方体の基本頂点を生成する

    Args:
        scale_x (float): X軸方向のスケール
        scale_y (float): Y軸方向のスケール
        scale_z (float): Z軸方向のスケール

    Returns:
        np.ndarray: 8個の頂点座標を含む配列 (8x3)
    """
    # スケールを半分にして中心を原点とする
    half_scale_x = scale_x / 2
    half_scale_y = scale_y / 2
    half_scale_z = scale_z / 2

    # 頂点座標の配列を作成
    return np.array([
        [-1 * half_scale_x, -1 * half_scale_y, -1 * half_scale_z],  # 左下前
        [+1 * half_scale_x, -1 * half_scale_y, -1 * half_scale_z],  # 右下前
        [+1 * half_scale_x, +1 * half_scale_y, -1 * half_scale_z],  # 右上前
        [-1 * half_scale_x, +1 * half_scale_y, -1 * half_scale_z],  # 左上前
        [-1 * half_scale_x, -1 * half_scale_y, +1 * half_scale_z],  # 左下後
        [+1 * half_scale_x, -1 * half_scale_y, +1 * half_scale_z],  # 右下後
        [+1 * half_scale_x, +1 * half_scale_y, +1 * half_scale_z],  # 右上後
        [-1 * half_scale_x, +1 * half_scale_y, +1 * half_scale_z]   # 左上後
    ])

def create_cube_faces():
    """立方体の面を構成する三角形を定義する

    Returns:
        np.ndarray: 12個の三角形を構成する頂点インデックスの配列 (12x3)
    """
    # 各面は2つの三角形で構成される（計12個の三角形）
    return np.array([
        [0, 1, 2], [0, 2, 3],  # 前面
        [4, 5, 6], [4, 6, 7],  # 後面
        [1, 5, 6], [1, 6, 2],  # 右面
        [0, 3, 7], [0, 7, 4],  # 左面
        [3, 2, 6], [3, 6, 7],  # 上面
        [0, 4, 5], [0, 5, 1]   # 下面
    ])

def create_mesh_from_vertices_faces(vertices, faces):
    """頂点と面の情報からメッシュを生成する

    Args:
        vertices (np.ndarray): 頂点座標の配列
        faces (np.ndarray): 面を構成する頂点インデックスの配列

    Returns:
        mesh.Mesh: 生成されたメッシュオブジェクト
    """
    # メッシュオブジェクトを初期化
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    cube.remove_duplicate_polygons = True

    # 各面の頂点座標を設定
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = vertices[f[j], :]

    return cube

def cube_model(scale_x=1, scale_y=1, scale_z=1):
    """通常の立方体メッシュを生成する

    Args:
        scale_x (float): X軸方向のスケール
        scale_y (float): Y軸方向のスケール
        scale_z (float): Z軸方向のスケール

    Returns:
        mesh.Mesh: 生成された立方体メッシュ
    """
    vertices = create_base_vertices(scale_x, scale_y, scale_z)
    faces = create_cube_faces()
    return create_mesh_from_vertices_faces(vertices, faces), vertices, faces

