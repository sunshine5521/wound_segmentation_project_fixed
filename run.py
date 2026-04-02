import argparse
import sys
import yaml
from src.train import Trainer
from src.inference import WoundAnalyzer

def train_model():
    trainer = Trainer("config.yaml")
    trainer.train()

def predict_single(image_path):
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    analyzer = WoundAnalyzer(checkpoint_path='checkpoints/best_model.pth', config=config)
    result = analyzer.analyze_image(image_path)
    analyzer.visualize_results(result)
    if result['actual_area_mm2']:
        print(f"创面面积: {result['actual_area_mm2']:.2f} mm²")
    else:
        print(f"像素面积: {result['pixel_area']} pixels")

def batch_predict(image_dir):
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    analyzer = WoundAnalyzer(checkpoint_path='checkpoints/best_model.pth', config=config)
    df = analyzer.batch_analyze(image_dir, 'batch_results.csv')
    print(f"处理完成！共分析 {len(df)} 张图像")
    print(f"结果已保存到 batch_results.csv")

def main():
    parser = argparse.ArgumentParser(description="小鼠创面分割系统")
    subparsers = parser.add_subparsers(dest='command', help='命令')
    subparsers.add_parser('train', help='训练模型')
    predict_parser = subparsers.add_parser('predict', help='预测单张图像')
    predict_parser.add_argument('--image', required=True, help='图像路径')
    batch_parser = subparsers.add_parser('batch', help='批量预测')
    batch_parser.add_argument('--dir', required=True, help='图像目录')
    args = parser.parse_args()

    if args.command == 'train':
        train_model()
    elif args.command == 'predict':
        predict_single(args.image)
    elif args.command == 'batch':
        batch_predict(args.dir)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
