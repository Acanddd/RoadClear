import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


# RSCM数据集类别映射到我们的标签（单标签分类）
# 类别索引: 0=fog, 1=rain, 2=snow
RSCM_LABEL_MAP = {
    "haze": 0,    # haze -> fog (类别0)
    "rainy": 1,    # rain -> rain (类别1)
    "snow": 2,    # snow -> snow (类别2)
}

# 类别名称
CLASS_NAMES = ["fog", "rainy", "snow"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNet weather classifier for RoadClear.")
    parser.add_argument(
        "--rscm-dir",
        type=str,
        required=True,
        help="RSCM_dataset 根目录，包含类别子文件夹 (haze, rainy, snow)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./trained_models_rscm",
        help="训练权重保存目录",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenet_v3_small",
        choices=["mobilenet_v3_small", "mobilenet_v3_large", "mobilenet_v2"],
        help="MobileNet backbone",
    )
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    parser.add_argument("--patience", type=int, default=10, help="早停耐心值")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练数据占比")
    parser.add_argument("--img-size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备",
    )
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="是否使用 ImageNet 预训练 backbone",
    )
    return parser.parse_args()


class RSCMDataset(Dataset):
    """RSCM数据集 - 单标签分类"""
    def __init__(self, image_paths: List[Path], transform: transforms.Compose) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = self._infer_label(image_path.parent.name)
        return image, label

    @staticmethod
    def _infer_label(category: str) -> int:
        """返回类别索引"""
        category = category.lower()
        if category in RSCM_LABEL_MAP:
            return RSCM_LABEL_MAP[category]
        raise ValueError(f"Unsupported RSCM category: {category}")


def build_rscm_samples(dataset_root: Path) -> List[Path]:
    """构建RSCM数据集样本列表"""
    samples = []
    for category_dir in dataset_root.iterdir():
        if not category_dir.is_dir():
            continue
        category = category_dir.name.lower()
        if category not in RSCM_LABEL_MAP:
            print(f"跳过不支持的类别: {category}")
            continue

        # 支持jpg和png格式
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for image_path in sorted(category_dir.glob(ext)):
                samples.append(image_path)

    if len(samples) == 0:
        raise RuntimeError("未找到可用于训练的RSCM样本，请检查数据集结构")
    return samples


def build_model(backbone: str, num_classes: int = 3, pretrained: bool = False) -> nn.Module:
    """构建MobileNet模型"""
    backbone = backbone.lower()
    if backbone == "mobilenet_v3_small":
        model = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone == "mobilenet_v3_large":
        model = torchvision.models.mobilenet_v3_large(pretrained=pretrained)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone == "mobilenet_v2":
        model = torchvision.models.mobilenet_v2(pretrained=pretrained)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    return model


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    """计算准确率和F1分数（单标签分类）"""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).float()
    accuracy = correct.mean().item()

    # 计算每个类别的F1分数
    num_classes = logits.size(1)
    f1_scores = []
    for cls in range(num_classes):
        tp = ((preds == cls) & (targets == cls)).sum().float()
        fp = ((preds == cls) & (targets != cls)).sum().float()
        fn = ((preds != cls) & (targets == cls)).sum().float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1.item())
    
    # 返回宏平均F1
    macro_f1 = np.mean(f1_scores)
    return accuracy, macro_f1


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    steps = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        accuracy, f1 = compute_metrics(logits, targets)
        total_loss += loss.item()
        total_acc += accuracy
        total_f1 += f1
        steps += 1

    return total_loss / steps, total_acc / steps, total_f1 / steps


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_f1 = 0.0
    steps = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)
        accuracy, f1 = compute_metrics(logits, targets)

        total_loss += loss.item()
        total_acc += accuracy
        total_f1 += f1
        steps += 1

    return total_loss / steps, total_acc / steps, total_f1 / steps


def main() -> None:
    args = parse_args()
    rscm_root = Path(args.rscm_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建RSCM训练样本
    rscm_samples = build_rscm_samples(rscm_root)
    print(f"RSCM数据集总样本数: {len(rscm_samples)}")

    # 统计每个类别的样本数
    category_counts = {name: 0 for name in CLASS_NAMES}
    for sample in rscm_samples:
        category = sample.parent.name.lower()
        if category in RSCM_LABEL_MAP:
            class_idx = RSCM_LABEL_MAP[category]
            category_counts[CLASS_NAMES[class_idx]] += 1
    
    print("类别分布:")
    for name, count in category_counts.items():
        print(f"  {name}: {count}")

    # 分割训练和验证数据
    np.random.seed(42)
    indices = np.random.permutation(len(rscm_samples))
    num_train = int(len(rscm_samples) * args.train_ratio)
    
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    train_samples = [rscm_samples[i] for i in train_indices]
    val_samples = [rscm_samples[i] for i in val_indices]

    print(f"训练样本数: {len(train_samples)}, 验证样本数: {len(val_samples)}")

    # 数据增强变换
    train_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = RSCMDataset(train_samples, train_transform)
    val_dataset = RSCMDataset(val_samples, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = build_model(args.backbone, num_classes=3, pretrained=args.pretrained_backbone)
    model = model.to(args.device)

    # 计算类别权重（处理不平衡）
    label_counts = np.zeros(3)
    for _, target in train_dataset:
        label_counts[target] += 1
    
    total_samples = len(train_dataset)
    class_weights = torch.tensor(total_samples / (3 * label_counts + 1e-8), dtype=torch.float32).to(args.device)
    
    print(f"类别权重: fog={class_weights[0]:.3f}, rain={class_weights[1]:.3f}, snow={class_weights[2]:.3f}")

    # 使用CrossEntropyLoss（单标签分类）
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = 0.0
    best_path = output_dir / f"weather_classifier_{args.backbone}.pth"
    patience_counter = 0

    print(f"使用设备: {args.device}, backbone: {args.backbone}, pretrained: {args.pretrained_backbone}")
    print(f"损失函数: CrossEntropyLoss (weighted)")
    print("开始训练...\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, torch.device(args.device)
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, torch.device(args.device)
        )
        scheduler.step()

        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
            print(f"  -> 保存最佳权重: {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"早停: 验证 F1 {args.patience} 个 epoch 未提升")
                break

    print(f"\n训练完成，最佳验证 F1={best_f1:.4f}")
    print(f"最终权重保存路径: {best_path}")


if __name__ == "__main__":
    main()
