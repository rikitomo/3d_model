import os
import torch
from mlp_model import NodeMLP
from module.stl_file_loader import STLFileLoader

def main():
    # ディレクトリパスの設定
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(current_dir, "models/node_mlp.pth")
    after_dir = os.path.join(current_dir, "data/stl_test")
    before_dir = os.path.join(current_dir, "data/stl_test")
    output_dir = os.path.join(current_dir, "data/stl_predicted")
    os.makedirs(output_dir, exist_ok=True)

    # lastを含むSTLファイルを検索
    test_files = [f for f in os.listdir(after_dir) if "last" in f and f.endswith(".stl")]
    
    if not test_files:
        print("予測対象となる'last'を含むSTLファイルが見つかりませんでした。")
        return
    
    for test_file in test_files:
        print(f"\n{test_file}の予測を開始します...")
        test_path = os.path.join(after_dir, test_file)
        base_name = test_file[:-9]  # '_last.stl'を除去
        before_file = base_name + "_first.stl"
        before_path = os.path.join(before_dir, before_file)
        
        try:
            # STLファイルの読み込み（メッシュ情報も含む）
            after_nodes, before_nodes, mesh_data = STLFileLoader.load_file_pair(test_path, before_path)
            
            # node_idでソート
            after_nodes = after_nodes[torch.argsort(after_nodes[:, 0])]
            before_nodes = before_nodes[torch.argsort(before_nodes[:, 0])]
        
            # モデルの準備
            num_nodes = after_nodes.shape[0]
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = NodeMLP(num_nodes=num_nodes)
            # モデルの読み込み（weights_only=Trueでセキュリティ警告を回避）
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.to(device)
            model.eval()  # モデルを評価モードにする

            # 予測の実行
            # 座標データを取得して形状を変換
            after_coords = after_nodes[:, 1:].float().contiguous()  # (num_nodes, 3)
            after_coords = after_coords.reshape(1, -1)  # (1, num_nodes * 3)
            after_coords = after_coords.to(device)
            print(f"入力データの形状: {after_coords.shape}")
            
            with torch.no_grad():  # 勾配計算を無効化
                # モデルに入力し、予測を取得
                predicted_coords = model(after_coords)  # (1, num_nodes * 3)
                predicted_coords = predicted_coords.reshape(num_nodes, 3)  # (num_nodes, 3)
            
            print(f"予測データの形状: {predicted_coords.shape}")
            
            # 予測データの形状を確認
            expected_shape = (num_nodes, 3)
            if predicted_coords.shape != expected_shape:
                raise ValueError(f"予測データの形状が想定外です: {predicted_coords.shape}, 期待値: {expected_shape}")
            
            # CPUに移動
            predicted_coords = predicted_coords.cpu()
            print(f"変形後の予測データの形状: {predicted_coords.shape}")
            
            # 予測結果をノードデータに組み込む
            predicted_nodes = torch.cat([after_nodes[:, 0:1], predicted_coords], dim=1)

            # 結果をSTLFileLoaderを使用して保存
            output_file = f"predicted_{base_name}_first.stl"
            output_path = os.path.join(output_dir, output_file)
            STLFileLoader.save_to_stl(predicted_nodes, mesh_data, output_path)
            print(f"予測結果を保存しました: {output_path}")
            
        except Exception as e:
            print(f"エラーが発生しました: {str(e)}")
            continue

if __name__ == "__main__":
    main()
