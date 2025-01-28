import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import expm
from collections import deque

def initialize_rotations_with_bfs(neighbors, log_rotation_differences):
    """
    BFSを用いて回転行列を初期化する。

    Args:
        neighbors (dict): 隣接情報
        log_rotation_differences (dict): RIMD特徴量の回転差分（log）

    Returns:
        list: BFSで初期化された回転行列 (n, 3, 3)
    """
    num_vertices = len(neighbors)
    rotations = [None] * num_vertices  # 回転行列を格納するリスト
    visited = [False] * num_vertices  # 訪問フラグ
    queue = deque()

    # 初期頂点を選択し、回転行列を単位行列で初期化
    initial_vertex = 0
    rotations[initial_vertex] = np.eye(3)
    visited[initial_vertex] = True
    queue.append(initial_vertex)

    # 幅優先探索
    while queue:
        current = queue.popleft()

        for neighbor in neighbors[current]:
            if not visited[neighbor]:
                # 回転差分 dRij を適用して隣接頂点の回転行列を初期化
                log_dRij = log_rotation_differences.get((current, neighbor), np.zeros((3, 3)))
                dRij = expm(log_dRij)
                rotations[neighbor] = rotations[current] @ dRij

                visited[neighbor] = True
                queue.append(neighbor)

    return rotations

def compute_cotangent_weights(vertices, faces):
    """
    コタンジェント重みを計算する関数。
    Args:
        vertices (np.ndarray): 頂点座標 (n, 3)
        faces (np.ndarray): 面情報 (m, 3)
    Returns:
        dict: 各エッジ (i, j) に対応するコタンジェント重み
    """
    edge_weights = {}

    for face in faces:
        # 三角形の3つの頂点インデックス
        i, j, k = face

        # 各頂点の座標
        vi, vj, vk = vertices[i], vertices[j], vertices[k]

        # 各辺のベクトル
        e_ij = vj - vi  # 辺 (i, j)
        e_jk = vk - vj  # 辺 (j, k)
        e_ki = vi - vk  # 辺 (k, i)

        # コタンジェントの計算
        def cotangent(v1, v2):
            cos_theta = np.dot(v1, v2)
            sin_theta = np.linalg.norm(np.cross(v1, v2))
            return cos_theta / sin_theta if sin_theta != 0 else 0.0

        # 各エッジに対応する角度の対辺のコタンジェント値を計算
        cot_alpha = cotangent(e_ki, -e_jk)  # 角度 α (辺 (i, j) の対角)
        cot_beta = cotangent(e_ij, -e_ki)  # 角度 β (辺 (j, k) の対角)
        cot_gamma = cotangent(e_jk, -e_ij)  # 角度 γ (辺 (k, i) の対角)

        # エッジごとのコタンジェント重みを計算
        def add_edge_weight(edge_weights, i, j, weight):
            edge = tuple(sorted((i, j)))  # エッジを双方向に対応するキーに変換
            if edge not in edge_weights:
                edge_weights[edge] = 0.0
            edge_weights[edge] += weight

        add_edge_weight(edge_weights, i, j, cot_alpha)
        add_edge_weight(edge_weights, j, k, cot_beta)
        add_edge_weight(edge_weights, k, i, cot_gamma)

    return edge_weights

def global_step(vertices_ref, reconstructed_vertices, rotations, rimd_features, neighbors, cotangent_weights):
    """
    グローバルステップ: 頂点の新しい位置を計算する。

    Args:
        vertices_ref (np.ndarray): 基準モデルの頂点座標 (n, 3)
        reconstructed_vertices (np.ndarray): 再構築された頂点座標 (n, 3)
        rotations (list): 各頂点の回転行列 (n, 3, 3)
        rimd_features (dict): RIMD特徴量（'log_rotation_differences', 'scalings_shears'）
        neighbors (dict): 隣接情報
        cotangent_weights (dict): 各エッジのコタンジェント重み

    Returns:
        np.ndarray: 更新された再構築頂点座標 (n, 3)
    """
    num_vertices = vertices_ref.shape[0]
    A = lil_matrix((3 * num_vertices, 3 * num_vertices))
    b = np.zeros((3 * num_vertices,))

    log_rotation_differences = rimd_features['log_rotation_differences']
    scalings_shears = rimd_features['scalings_shears']

    for i in range(num_vertices):
        if i not in neighbors:
            continue

        for j in neighbors[i]:
            if i < j:
                # コタンジェント重み c_ij を取得
                c_ij = cotangent_weights.get((i, j), 1.0)

                # 基準形状でのエッジベクトル
                e_ij_ref = vertices_ref[j] - vertices_ref[i]

                # ローカル変換 (dRij, Sj) を適用したエッジ
                dRij = expm(log_rotation_differences.get((i, j), np.zeros((3, 3))))
                S_i = scalings_shears[i]
                transformed_e_ij = rotations[i] @ dRij @ S_i @ e_ij_ref

                # 行列Aとベクトルbを更新
                for dim in range(3):
                    A[3 * i + dim, 3 * i + dim] += c_ij
                    A[3 * i + dim, 3 * j + dim] -= c_ij

                    A[3 * j + dim, 3 * i + dim] -= c_ij
                    A[3 * j + dim, 3 * j + dim] += c_ij

                    b[3 * i + dim] += c_ij * transformed_e_ij[dim]
                    b[3 * j + dim] -= c_ij * transformed_e_ij[dim]

    # 線形システムを解いて新しい頂点位置を計算
    x = spsolve(A.tocsr(), b)
    return x.reshape((num_vertices, 3))

def local_step(vertices_ref, reconstructed_vertices, rotations, neighbors):
    """
    ローカルステップ: 各頂点の回転行列を計算する。

    Args:
        vertices_ref (np.ndarray): 基準モデルの頂点座標 (n, 3)
        reconstructed_vertices (np.ndarray): 再構築された頂点座標 (n, 3)
        rotations (list): 各頂点の回転行列 (n, 3, 3)
        neighbors (dict): 隣接情報

    Returns:
        list: 更新された回転行列 (n, 3, 3)
    """
    num_vertices = vertices_ref.shape[0]

    for i in range(num_vertices):
        if i not in neighbors:
            continue

        Ni = neighbors[i]

        # 隣接頂点間のエッジベクトル
        e_ref = np.array([vertices_ref[j] - vertices_ref[i] for j in Ni])
        e_rec = np.array([reconstructed_vertices[j] - reconstructed_vertices[i] for j in Ni])

        # コタンジェント重み
        weights = np.array([1 / len(neighbors[j]) for j in Ni])

        # 最適な回転行列を計算
        S = sum(weights[k] * np.outer(e_rec[k], e_ref[k]) for k in range(len(Ni)))
        U, _, Vt = np.linalg.svd(S)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = U @ Vt

        rotations[i] = R

    return rotations

def compute_reconstruction_error(vertices_ref, reconstructed_vertices):
    """
    再構築エラーを計算する。

    Args:
        vertices_ref (np.ndarray): 基準モデルの頂点座標 (n, 3)
        reconstructed_vertices (np.ndarray): 再構築された頂点座標 (n, 3)

    Returns:
        float: 再構築エラー
    """
    return np.linalg.norm(vertices_ref - reconstructed_vertices)

def reconstruct_model_from_RIMD_optimized(rimd_features, vertices_ref, neighbors, base_faces, max_iterations=100, tolerance=1e-3):
    """
    RIMD特徴量からモデルを再構築する関数。
    BFSによる初期化を使用し、エネルギーの変化量が5回連続で閾値以下になった場合に終了条件を満たす。

    Args:
        rimd_features (dict): RIMD特徴量（'log_rotation_differences', 'scalings_shears'）
        vertices_ref (np.ndarray): 基準モデルの頂点座標 (n, 3)
        neighbors (dict): 隣接情報
        base_faces (np.ndarray): メッシュの三角形インデックス (m, 3)
        max_iterations (int): 最大反復回数
        tolerance (float): エネルギー変化量の終了閾値

    Returns:
        np.ndarray: 再構築された頂点座標 (n, 3)
    """
    num_vertices = vertices_ref.shape[0]
    reconstructed_vertices = np.copy(vertices_ref)  # 初期化

    # BFSを用いた初期化
    rotations = initialize_rotations_with_bfs(neighbors, rimd_features['log_rotation_differences'])

    # コタンジェント重みを計算
    cotangent_weights = compute_cotangent_weights(vertices_ref, base_faces)

    previous_energies = [float('inf')] * 5  # エネルギー変化量を記録するリスト

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")

        # グローバルステップ
        reconstructed_vertices = global_step(vertices_ref, reconstructed_vertices, rotations, rimd_features, neighbors, cotangent_weights)

        # ローカルステップ
        rotations = local_step(vertices_ref, reconstructed_vertices, rotations, neighbors)

        # エネルギーを計算
        current_energy = compute_reconstruction_error(vertices_ref, reconstructed_vertices)
        print(f"\tReconstruction error: {current_energy:.6f}")

        # エネルギー変化量の更新
        energy_change = abs(previous_energies[-1] - current_energy)
        previous_energies.pop(0)
        previous_energies.append(current_energy)

        # 終了条件の判定
        if all(abs(previous_energies[i] - previous_energies[i + 1]) < tolerance for i in range(4)):
            print("\tConverged based on energy change tolerance.")
            break

    return reconstructed_vertices