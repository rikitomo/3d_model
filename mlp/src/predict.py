import os
import torch
from mlp_model import NodeMLP, AfterBeforeDataset
from k_file_loader import KFileLoader

def save_as_k_file(nodes: torch.Tensor, elements: torch.Tensor, output_path: str):
    """
    ノードとエレメントデータをkファイル形式で保存する

    Args:
        nodes: ノードデータ (N, 4) [node_id, x, y, z]
        elements: エレメントデータ (M, 6) [element_id, pid, node1, node2, node3, node4]
        output_path: 出力ファイルパス
    """
    with open(output_path, 'w') as f:
        # ヘッダー
        f.write("$# LS-DYNA Keyword file\n")
        f.write("*KEYWORD\n")
        
        # ノードセクション
        f.write("*NODE\n")
        for node in nodes:
            node_id = int(node[0])
            x, y, z = node[1:4].tolist()
            f.write(f"{node_id:8d}{x:16.8f}{y:16.8f}{z:16.8f}\n")
        
        # エレメントセクション
        f.write("*ELEMENT_SHELL\n")
        for element in elements:
            element_id = int(element[0])
            pid = int(element[1])
            n1, n2, n3, n4 = map(int, element[2:6])
            f.write(f"{element_id:8d}{pid:8d}{n1:8d}{n2:8d}{n3:8d}{n4:8d}\n")
        
        # 終了
        f.write("*END\n")

def main():
    # ディレクトリパスの設定
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(current_dir, "models/node_mlp.pth")
    after_dir = os.path.join(current_dir, "data/model_after")
    output_dir = os.path.join(current_dir, "data/model_predicted")
    os.makedirs(output_dir, exist_ok=True)

    # テストファイルの処理
    test_files = ["Manual-chair-geometry-1.k", "Manual-chair-geometry-2.k"]
    
    for test_file in test_files:
        print(f"\n{test_file}の予測を開始します...")
        test_path = os.path.join(after_dir, test_file)
        nodes, elements = KFileLoader.load_single_file(test_path)
    
        # モデルの準備
        num_nodes = nodes.size(0)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = NodeMLP(num_nodes=num_nodes)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        
        # 予測の実行
        after_coords = nodes[:, 1:].unsqueeze(0)  # バッチ次元を追加
        print(f"入力データの形状: {after_coords.shape}")
        after_coords = after_coords.reshape(1, -1)  # (1, num_nodes * 3)
        print(f"変形後の入力データの形状: {after_coords.shape}")
        
        predicted_coords = model(after_coords.to(device))
        print(f"予測データの形状: {predicted_coords.shape}")
        predicted_coords = predicted_coords.reshape(-1, 3).cpu()
        print(f"変形後の予測データの形状: {predicted_coords.shape}")
        
        # 予測結果をノードデータに組み込む
        predicted_nodes = torch.cat([nodes[:, 0:1], predicted_coords], dim=1)
        
        # 結果の保存
        output_file = f"predicted_{test_file}"
        output_path = os.path.join(output_dir, output_file)
        save_as_k_file(predicted_nodes, elements, output_path)
        print(f"予測結果を保存しました: {output_path}")

if __name__ == "__main__":
    main()
