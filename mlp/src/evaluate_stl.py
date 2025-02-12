import os
import argparse
import numpy as np
import pandas as pd
from module.stl_file_loader import STLFileLoader

class STLEvaluator:
    def __init__(self, true_stl_path: str, pred_stl_path: str):
        # STLファイルの読み込み
        self.true_nodes, self.true_faces = STLFileLoader.read_stl_file(true_stl_path)
        self.pred_nodes, self.pred_faces = STLFileLoader.read_stl_file(pred_stl_path)
        
        # ノード座標をnumpy配列に変換
        self.true_coords = np.array([coords for _, coords in self.true_nodes.items()])
        self.pred_coords = np.array([coords for _, coords in self.pred_nodes.items()])
        
    def calculate_node_distances(self):
        """各ノードの座標差（ユークリッド距離）を計算"""
        return np.linalg.norm(self.true_coords - self.pred_coords, axis=1)
    
    def calculate_axis_differences(self):
        """xyz方向それぞれの差分を計算"""
        return self.true_coords - self.pred_coords
    
    def calculate_dimension_differences(self):
        """モデルの縦横高さの差を計算"""
        true_dims = np.ptp(self.true_coords, axis=0)  # 各軸の最大値と最小値の差
        pred_dims = np.ptp(self.pred_coords, axis=0)
        return true_dims - pred_dims

    def calculate_normal_similarity(self):
        """法線ベクトルの類似度を計算"""
        def compute_face_normals(nodes, faces):
            normals = []
            for face in faces:
                v0 = nodes[face[0]]
                v1 = nodes[face[1]]
                v2 = nodes[face[2]]
                # 面の法線ベクトルを計算
                normal = np.cross(v1 - v0, v2 - v0)
                # 正規化
                normal = normal / np.linalg.norm(normal)
                normals.append(normal)
            return np.array(normals)

        # 両方の面の法線を計算
        true_normals = compute_face_normals(self.true_nodes, self.true_faces)
        pred_normals = compute_face_normals(self.pred_nodes, self.pred_faces)

        # 法線ベクトル間の角度を計算
        dot_products = np.abs(np.sum(true_normals[:, np.newaxis] * pred_normals, axis=2))
        angles = np.arccos(np.clip(dot_products, -1.0, 1.0))
        
        # 各面の最小角度を取得（最も近い向きの面との角度）
        min_angles = np.min(angles, axis=1)
        return np.mean(min_angles), np.max(min_angles)

    def calculate_hausdorff_distance(self):
        """ハウスドルフ距離を計算"""
        def directed_hausdorff(source, target):
            distances = np.linalg.norm(source[:, np.newaxis] - target, axis=2)
            return np.max(np.min(distances, axis=1))

        # 双方向のハウスドルフ距離を計算
        forward = directed_hausdorff(self.true_coords, self.pred_coords)
        backward = directed_hausdorff(self.pred_coords, self.true_coords)
        
        return max(forward, backward)

    def calculate_volume_difference(self):
        """モデルの体積差を計算"""
        def compute_volume(nodes, faces):
            volume = 0
            for face in faces:
                v0 = nodes[face[0]]
                v1 = nodes[face[1]]
                v2 = nodes[face[2]]
                # 符号付き四面体の体積を計算
                volume += np.abs(np.dot(v0, np.cross(v1, v2))) / 6.0
            return volume

        true_volume = compute_volume(self.true_nodes, self.true_faces)
        pred_volume = compute_volume(self.pred_nodes, self.pred_faces)
        
        return true_volume - pred_volume, abs(true_volume - pred_volume) / true_volume * 100
    
    def evaluate(self):
        """評価指標を計算"""
        distances = self.calculate_node_distances()
        axis_diffs = self.calculate_axis_differences()
        dim_diffs = self.calculate_dimension_differences()
        mean_angle, max_angle = self.calculate_normal_similarity()
        hausdorff_dist = self.calculate_hausdorff_distance()
        volume_diff, volume_diff_percent = self.calculate_volume_difference()
        
        metrics = {
            "平均座標差": np.mean(distances),
            "最大座標差": np.max(distances),
            "座標差の標準偏差": np.std(distances),
            "X方向の最大ズレ": np.max(np.abs(axis_diffs[:, 0])),
            "Y方向の最大ズレ": np.max(np.abs(axis_diffs[:, 1])),
            "Z方向の最大ズレ": np.max(np.abs(axis_diffs[:, 2])),
            "X方向の寸法差": dim_diffs[0],
            "Y方向の寸法差": dim_diffs[1],
            "Z方向の寸法差": dim_diffs[2],
            "法線の平均角度差(rad)": mean_angle,
            "法線の最大角度差(rad)": max_angle,
            "ハウスドルフ距離": hausdorff_dist,
            "体積差": volume_diff,
            "体積差率(%)": volume_diff_percent
        }
        
        return metrics

def find_stl_pairs(true_dir: str, pred_dir: str):
    """正解と予測のSTLファイルのペアを見つける"""
    # 正解ファイルの一覧を取得
    true_files = {
        os.path.splitext(f)[0].replace("_first", ""): os.path.join(true_dir, f)
        for f in os.listdir(true_dir)
        if f.endswith(".STL") and "_first" in f
    }
    
    # 予測ファイルの一覧を取得
    pred_files = {
        os.path.splitext(f)[0].replace("predicted_", "").replace("_first", ""): os.path.join(pred_dir, f)
        for f in os.listdir(pred_dir)
        if f.endswith(".stl") and "predicted_" in f
    }
    
    # 共通のファイルペアを見つける
    common_names = set(true_files.keys()) & set(pred_files.keys())
    
    return [(true_files[name], pred_files[name]) for name in common_names]

def evaluate_all_pairs(file_pairs: list, output_csv: str):
    """全てのファイルペアを評価"""
    all_results = []
    
    for true_path, pred_path in file_pairs:
        # ファイル名を取得
        true_name = os.path.basename(true_path)
        pred_name = os.path.basename(pred_path)
        print(f"\n評価: {true_name} vs {pred_name}")
        
        # 評価の実行
        evaluator = STLEvaluator(true_path, pred_path)
        metrics = evaluator.evaluate()
        
        # 結果にファイル名を追加
        metrics["ファイル名"] = true_name
        all_results.append(metrics)
        
        # 個別の結果を表示
        print("=== 評価結果 ===")
        for metric, value in metrics.items():
            if metric != "ファイル名":
                print(f"{metric}: {value:.6f}")
    
    if all_results:
        df = pd.DataFrame(all_results)
        cols = ["ファイル名"] + [col for col in df.columns if col != "ファイル名"]
        df = df[cols]

        # CSVファイルに出力
        df.to_csv(output_csv, index=False, encoding="cp932")
        print(f"\n全ての評価結果を {output_csv} に保存しました。")

    else:
        print("評価可能なファイルペアが見つかりませんでした。")

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="STLファイルの形状評価")
    parser.add_argument("--true-dir", type=str, required=True, help="正解のSTLファイルが格納されているディレクトリ")
    parser.add_argument("--pred-dir", type=str, required=True, help="実験結果が格納されているディレクトリ")
    parser.add_argument("--output-name", type=str, default="evaluation_results.csv", help="出力CSVファイルの名前")
    
    args = parser.parse_args()
    
    output_dir = os.path.join(args.pred_dir, "evaluate")
    
    # ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, args.output_name)
    
    # ファイルペアの検索
    print(f"正解ディレクトリ: {args.true_dir}")
    print(f"予測ディレクトリ: {args.pred_dir}/predictions")
    print(f"出力CSV: {output_csv}")
    
    file_pairs = find_stl_pairs(args.true_dir, args.pred_dir + "/predictions")
    print(f"\n{len(file_pairs)}個のファイルペアが見つかりました。")
    
    # 全ペアの評価
    evaluate_all_pairs(file_pairs, output_csv)

if __name__ == "__main__":
    main()
