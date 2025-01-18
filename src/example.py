from cube_model import cube_model
from deformation_gradient import (
    compute_neighbors_from_faces,
    compute_cotangent_weights,
    compute_deformation_gradient,
    decompose_deformation_gradient,
    compute_rotation_differences,
    compute_RIMD_feature,
    reconstruct_model_from_RIMD,
    align_to_reference,
    compute_reconstruction_error,
)
from visualization import plot_model
import numpy as np

is_debug = True

def main():
    print("Step 1: 基準モデルの生成")
    base_mesh, base_vertices, base_faces = cube_model()
    plot_model(base_vertices, base_faces, "Step 1: Original Cube")

    print("\nStep 2: 変形モデルの生成")
    dented_mesh, deformed_vertices, deformed_faces = cube_model(10, 10, 1)
    plot_model(deformed_vertices, deformed_faces, "Step 2: Deformed Cube")
    
    print("\nStep 3: 隣接情報の計算")
    neighbors = compute_neighbors_from_faces(base_faces)
    if is_debug:
        print("隣接情報:")
        for vertex, adj_vertices in neighbors.items():
            print(f"頂点 {vertex}: 隣接 {adj_vertices}")

    print("\nStep 4: コタンジェント重みの計算")
    # コタンジェント重みを計算
    cotangent_weights = compute_cotangent_weights(base_vertices, base_faces)
    if is_debug:
        print("コタンジェント重み:")
        for edge, weight in cotangent_weights.items():
            print(f"エッジ {edge}: コタンジェント重み {weight}")
    
    print("\nStep 5: 変形勾配の計算")
    deformation_gradients = compute_deformation_gradient(base_vertices, deformed_vertices, neighbors, cotangent_weights)
    if is_debug:
        for i, T_i in enumerate(deformation_gradients):
            print(f"頂点 {i} の変形勾配:\n{T_i}\n")

    print("\nStep 6: 変形勾配を回転とスケール/シアーに極分解")
    rotations, scalings_shears = decompose_deformation_gradient(deformation_gradients)
    if is_debug:
        for i, (R, S) in enumerate(zip(rotations, scalings_shears)):
            print(f"頂点 {i} の回転行列 R:\n{R}\n")
            print(f"頂点 {i} のスケール/シアー行列 S:\n{S}\n")

    print("\nStep 7: 隣接頂点間の回転差の計算")
    rotation_differences = compute_rotation_differences(rotations, neighbors)
    if is_debug:
        for edge, dR in rotation_differences.items():
            print(f"エッジ {edge} の回転差行列 dR:\n{dR}\n")
    
    print("\nStep 8: RIMD特徴量の計算")
    rimd_features = compute_RIMD_feature(rotations, scalings_shears, neighbors)
    if is_debug:
        print("回転差行列の対数 (log_rotation_differences):")
        for edge, log_dR in rimd_features['log_rotation_differences'].items():
            print(f"エッジ {edge}:\n{log_dR}\n")

        print("スケール/シアー行列 (scalings_shears):")
        for i, S in enumerate(rimd_features['scalings_shears']):
            print(f"頂点 {i}:\n{S}\n")
    
    print("\nStep 9: RIMD特徴量からモデルを再構築")
    reconstructed_vertices = reconstruct_model_from_RIMD(rimd_features, base_vertices, neighbors)
    if is_debug:
        print("再構築された頂点座標:")
        print(reconstructed_vertices)

    print("\nStep 10: 基準モデルと再構築モデルの位置合わせ")
    reconstructed_vertices = align_to_reference(base_vertices, reconstructed_vertices)

    print("\nStep 11: 再構築されたモデルの可視化")
    plot_model(reconstructed_vertices, base_faces, "Step 10: Reconstructed Cube")

    print("\nStep 12: 再構築誤差の計算")
    error = compute_reconstruction_error(deformed_vertices, reconstructed_vertices)
    print(f"再構築誤差: {error}")

if __name__ == "__main__":
    main()
