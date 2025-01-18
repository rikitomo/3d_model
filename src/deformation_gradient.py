from collections import defaultdict
import numpy as np
from numpy.linalg import lstsq
from scipy.linalg import logm, expm


def compute_neighbors_from_faces(faces):
    """
    面情報から隣接情報を計算する関数。
    Args:
        faces (np.ndarray): 面情報 (m, 3)
    Returns:
        dict: 各頂点の隣接情報を保持する辞書
    """
    neighbors = defaultdict(set)  # 隣接情報をセットで保持して重複を排除

    for face in faces:
        # 三角形の各辺をエッジとして追加
        for i in range(3):  # 三角形の場合
            vi = face[i]
            vj = face[(i + 1) % 3]  # 循環的に次の頂点を取得
            neighbors[vi].add(vj)
            neighbors[vj].add(vi)

    # 辞書をリスト形式に変換して返す
    return {key: list(value) for key, value in neighbors.items()}

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
        cot_beta = cotangent(e_jk, -e_ij)  # 角度 β (辺 (j, k) の対角)
        cot_gamma = cotangent(e_ij, -e_ki)  # 角度 γ (辺 (k, i) の対角)

        # エッジごとのコタンジェント重みを計算
        def add_edge_weight(edge_weights, i, j, weight):
            edge = tuple(sorted((i, j)))  # エッジを双方向に対応するキーに変換
            if edge not in edge_weights:
                edge_weights[edge] = 0.0
            edge_weights[edge] += weight

        add_edge_weight(edge_weights, i, j, cot_alpha + cot_beta)
        add_edge_weight(edge_weights, j, k, cot_beta + cot_gamma)
        add_edge_weight(edge_weights, k, i, cot_gamma + cot_alpha)

    return edge_weights

def compute_deformation_gradient(vertices_ref, vertices_def, neighbors, cotangent_weights):
    """
    各頂点の変形勾配を計算する関数。
    Args:
        vertices_ref (np.ndarray): 基準モデルの頂点座標 (n, 3)
        vertices_def (np.ndarray): 変形モデルの頂点座標 (n, 3)
        neighbors (dict): 隣接情報（各頂点に隣接する頂点のリスト）
        cotangent_weights (dict): 各エッジに対するコタンジェント重み
    Returns:
        np.ndarray: 各頂点の変形勾配 (n, 3, 3)
    """
    num_vertices = vertices_ref.shape[0]
    deformation_gradients = np.zeros((num_vertices, 3, 3))  # 各頂点の3x3変形勾配

    for i in range(num_vertices):
        # 頂点 i の隣接頂点
        if i not in neighbors:
            continue  # 隣接頂点がない場合スキップ
        Ni = neighbors[i]

        # エッジベクトル（基準モデルと変形モデル）
        e_ref = np.array([vertices_ref[j] - vertices_ref[i] for j in Ni])
        e_def = np.array([vertices_def[j] - vertices_def[i] for j in Ni])

        # コタンジェント重み行列
        weights = np.array([cotangent_weights.get(tuple(sorted((i, j))), 0.0) for j in Ni])
        C = np.diag(weights)  # 重み行列を対角行列に

        # 最小二乗法で変形勾配 T を計算
        # T_i = argmin || C (e_def - T e_ref) ||^2
        A = e_ref.T @ C @ e_ref
        B = e_ref.T @ C @ e_def
        T_i, _, _, _ = lstsq(A, B, rcond=None)  # 最小二乗法で T を求める

        # 結果を保存
        deformation_gradients[i] = T_i

    return deformation_gradients

import numpy as np

def decompose_deformation_gradient(deformation_gradients):
    """
    変形勾配を回転部分とスケール/シアー部分に分解する関数。
    Args:
        deformation_gradients (np.ndarray): 各頂点の変形勾配 (n, 3, 3)
    Returns:
        tuple: (rotations, scalings_shears)
            rotations (np.ndarray): 各頂点の回転行列 (n, 3, 3)
            scalings_shears (np.ndarray): 各頂点のスケール/シアー行列 (n, 3, 3)
    """
    num_vertices = deformation_gradients.shape[0]
    rotations = np.zeros((num_vertices, 3, 3))
    scalings_shears = np.zeros((num_vertices, 3, 3))

    for i in range(num_vertices):
        T = deformation_gradients[i]

        # 特異値分解 (SVD)
        U, Sigma, Vt = np.linalg.svd(T)

        # 回転行列 R
        R = U @ Vt
        if np.linalg.det(R) < 0:  # 反転が生じた場合の修正
            Vt[2, :] *= -1
            R = U @ Vt

        # スケール/シアー行列 S
        S = Vt.T @ np.diag(Sigma) @ Vt

        # 結果を保存
        rotations[i] = R
        scalings_shears[i] = S

    return rotations, scalings_shears

def compute_rotation_differences(rotations, neighbors):
    """
    隣接する頂点間の回転差を計算する関数。
    Args:
        rotations (np.ndarray): 各頂点の回転行列 (n, 3, 3)
        neighbors (dict): 隣接情報（各頂点に隣接する頂点のリスト）
    Returns:
        dict: 各エッジ (i, j) に対応する回転差行列 (3, 3)
    """
    rotation_differences = {}

    for i, Ri in enumerate(rotations):
        if i not in neighbors:
            continue  # 隣接頂点がない場合はスキップ

        for j in neighbors[i]:
            if i < j:  # エッジ (i, j) の片方向のみ計算
                Rj = rotations[j]
                dRij = Ri.T @ Rj  # 回転差 dR_ij = R_i^T * R_j
                rotation_differences[(i, j)] = dRij

    return rotation_differences

def compute_RIMD_feature(rotations, scalings_shears, neighbors):
    """
    RIMD特徴量を計算する関数。
    Args:
        rotations (np.ndarray): 各頂点の回転行列 (n, 3, 3)
        scalings_shears (np.ndarray): 各頂点のスケール/シアー行列 (n, 3, 3)
        neighbors (dict): 隣接情報（各頂点に隣接する頂点のリスト）
    Returns:
        dict: RIMD特徴量
            - 'log_rotation_differences': 各エッジの回転差行列の対数
            - 'scalings_shears': 各頂点のスケール/シアー行列
    """
    log_rotation_differences = {}

    for i, Ri in enumerate(rotations):
        if i not in neighbors:
            continue  # 隣接頂点がない場合スキップ

        for j in neighbors[i]:
            if i < j:  # エッジ (i, j) の片方向のみ計算
                Rj = rotations[j]
                dRij = Ri.T @ Rj  # 回転差 dR_ij = R_i^T * R_j
                log_dRij = logm(dRij)  # 行列対数を計算
                log_rotation_differences[(i, j)] = log_dRij

    # RIMD特徴量を辞書形式で返す
    return {
        'log_rotation_differences': log_rotation_differences,
        'scalings_shears': scalings_shears
    }

# TODO:エネルギー最小化による再構築が実施できていない
def reconstruct_model_from_RIMD(rimd_features, vertices_ref, neighbors):
    """
    RIMD特徴量からモデルを再構築する関数。
    Args:
        rimd_features (dict): RIMD特徴量（'log_rotation_differences', 'scalings_shears'）
        vertices_ref (np.ndarray): 基準モデルの頂点座標 (n, 3)
        neighbors (dict): 隣接情報（各頂点に隣接する頂点のリスト）
    Returns:
        np.ndarray: 再構築された頂点座標 (n, 3)
    """
    num_vertices = vertices_ref.shape[0]
    reconstructed_vertices = np.zeros_like(vertices_ref)  # 再構築された頂点座標
    rotations = [np.eye(3) for _ in range(num_vertices)]  # 各頂点の回転行列

    # 回転行列を再構築
    log_rotation_differences = rimd_features['log_rotation_differences']
    for i in range(num_vertices):
        if i not in neighbors:
            continue

        for j in neighbors[i]:
            if (i, j) in log_rotation_differences:
                log_dRij = log_rotation_differences[(i, j)]
                dRij = expm(log_dRij)  # 行列指数を計算して dRij を取得
                rotations[j] = rotations[i] @ dRij  # 回転行列を累積

    # 頂点の位置を再構築
    reconstructed_vertices[0] = vertices_ref[0]  # 基準点の位置はそのまま
    scalings_shears = rimd_features['scalings_shears']

    for i in range(num_vertices):
        if i not in neighbors:
            continue

        for j in neighbors[i]:
            if i < j:  # エッジ (i, j) を計算
                S_i = scalings_shears[i]  # 頂点 i のスケール/シアー行列
                e_ij = rotations[i] @ S_i @ (vertices_ref[j] - vertices_ref[i])  # 再構築されたエッジベクトル
                reconstructed_vertices[j] = reconstructed_vertices[i] + e_ij  # 頂点 j の位置を計算

    return reconstructed_vertices

def align_to_reference(reference_vertices, reconstructed_vertices):
    """
    再構築モデルを基準モデルに位置合わせする関数。
    Args:
        reference_vertices (np.ndarray): 基準モデルの頂点座標 (n, 3)
        reconstructed_vertices (np.ndarray): 再構築された頂点座標 (n, 3)
    Returns:
        np.ndarray: 位置合わせされた再構築頂点座標 (n, 3)
    """
    reference_center = np.mean(reference_vertices, axis=0)
    reconstructed_center = np.mean(reconstructed_vertices, axis=0)
    offset = reference_center - reconstructed_center
    return reconstructed_vertices + offset

def compute_reconstruction_error(original_vertices, reconstructed_vertices):
    return np.mean(np.linalg.norm(original_vertices - reconstructed_vertices, axis=1))