import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import numpy.typing as npt

def plot_model(vertices: npt.NDArray[np.float64], 
              faces: npt.NDArray[np.int64], 
              title: str = "3D Model") -> None:
    """
    3Dモデルを可視化します。

    Args:
        vertices: 頂点座標の配列 (N, 3)
        faces: 面を構成する頂点インデックスの配列 (M, 3)
        title: プロットのタイトル
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    # 各面の頂点座標を取得
    poly3d = []
    for face in faces:
        vertices_face = vertices[face]
        poly3d.append(vertices_face)

    # モデルの表示
    collection = Poly3DCollection(
        poly3d,
        facecolors='blue',
        edgecolors='black',
        alpha=0.3
    )
    ax.add_collection3d(collection)

    # 軸の設定
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    
    margin = 0.2
    ax.set_xlim([x_min - margin, x_max + margin])
    ax.set_ylim([y_min - margin, y_max + margin])
    ax.set_zlim([z_min - margin, z_max + margin])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
