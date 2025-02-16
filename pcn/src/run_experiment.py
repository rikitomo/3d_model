import subprocess
from pathlib import Path
from typing import Optional
import torch

def run_predict(
    model_path: str,
    after_dir: str,
    before_dir: str,
    output_dir: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """予測を実行する関数"""
    cmd = [
        "python", "predict.py",
        "--model-path", model_path,
        "--after-dir", after_dir,
        "--before-dir", before_dir,
        "--output-dir", output_dir,
        "--device", device
    ]
    
    # カレントディレクトリをスクリプトのディレクトリに変更
    script_dir = Path(__file__).parent
    subprocess.run(cmd, cwd=script_dir)

def run_experiment(
    experiment_name: str,
    after_dir: str,
    before_dir: str,
    batch_size: int = 32,
    num_epochs: int = 200,
    learning_rate: float = 0.0005,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0001,
    early_stopping_min_epochs: int = 10,
    train_val_split_ratio: float = 0.8,
    output_dir: str = None
):
    """学習を実行する関数"""
    cmd = [
        "python", "train_pcn.py",
        "--experiment_name", experiment_name,
        "--after_dir", after_dir,
        "--before_dir", before_dir,
        "--batch_size", str(batch_size),
        "--num_epochs", str(num_epochs),
        "--learning_rate", str(learning_rate),
        "--early_stopping_patience", str(early_stopping_patience),
        "--early_stopping_min_delta", str(early_stopping_min_delta),
        "--early_stopping_min_epochs", str(early_stopping_min_epochs),
        "--train_val_split_ratio", str(train_val_split_ratio)
    ]
    
    if output_dir:
        cmd.extend(["--output_dir", output_dir])
    
    # カレントディレクトリをスクリプトのディレクトリに変更
    script_dir = Path(__file__).parent
    subprocess.run(cmd, cwd=script_dir)

def run_training_and_prediction(
    experiment_name: str,
    train_after_dir: str,
    train_before_dir: str,
    test_after_dir: str,
    test_before_dir: str,
    output_dir: Optional[str] = None,
    batch_size: int = 32,
    num_epochs: int = 200,
    learning_rate: float = 0.0005,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 0.0001,
    early_stopping_min_epochs: int = 10,
    train_val_split_ratio: float = 0.8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """学習と予測を連続して実行する関数"""
    # 学習の実行
    run_experiment(
        experiment_name=experiment_name,
        after_dir=train_after_dir,
        before_dir=train_before_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_min_epochs=early_stopping_min_epochs,
        train_val_split_ratio=train_val_split_ratio,
        output_dir=output_dir
    )
    
    # 実験ディレクトリの設定
    if output_dir:
        exp_dir = Path(output_dir) / experiment_name
    else:
        exp_dir = Path(__file__).parent.parent / "experiments" / experiment_name
    
    # 予測の実行
    model_path = str(exp_dir / "models" / "node_pcn.pth")
    predict_output_dir = str(exp_dir / "predictions")
    
    run_predict(
        model_path=model_path,
        after_dir=test_after_dir,
        before_dir=test_before_dir,
        output_dir=predict_output_dir,
        device=device
    )

def main():
    """実験設定例"""
    # プロジェクトのルートディレクトリを取得
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    # 実験1: 学習と予測
    run_training_and_prediction(
        experiment_name="exp001",
        train_after_dir=str(data_dir / "stl_train"),
        train_before_dir=str(data_dir / "stl_train"),
        test_after_dir=str(data_dir / "stl_test"),
        test_before_dir=str(data_dir / "stl_test"),
        batch_size=16,
        num_epochs=200,
        early_stopping_min_epochs=10,
        learning_rate=0.001
    )

if __name__ == "__main__":
    main()
