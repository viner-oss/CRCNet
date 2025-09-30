"""
Metrics
"""
import torch
from torchmetrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUROC,
    AveragePrecision,
    ConfusionMatrix, MatthewsCorrCoef
)

def compute_accuracy(preds: torch.Tensor,
                     target: torch.Tensor,
                     device: torch.device,
                     num_classes: int = 10,
                     average: str = 'macro') -> torch.Tensor:
    metric = Accuracy(task='multiclass',
                      num_classes=num_classes,
                      average=average
    ).to(device)
    return metric(preds, target)

def compute_precision(preds: torch.Tensor,
                      target: torch.Tensor,
                      device: torch.device,
                      num_classes: int = 10,
                      average: str = 'macro') -> torch.Tensor:
    metric = Precision(task='multiclass',
                       num_classes=num_classes,
                       average=average
    ).to(device)
    return metric(preds, target)

def compute_recall(preds: torch.Tensor,
                   target: torch.Tensor,
                   device: torch.device,
                   num_classes: int = 10,
                   average: str = 'macro') -> torch.Tensor:
    metric = Recall(task='multiclass',
                    num_classes=num_classes,
                    average=average
    ).to(device)
    return metric(preds, target)

def compute_f1(preds: torch.Tensor,
               target: torch.Tensor,
               device: torch.device,
               num_classes: int = 10,
               average: str = 'macro') -> torch.Tensor:
    metric = F1Score(task='multiclass',
                     num_classes=num_classes,
                     average=average
    ).to(device)
    return metric(preds, target)

def compute_confusion_matrix(preds: torch.Tensor,
                             target: torch.Tensor,
                             device: torch.device,
                             num_classes: int = 10) -> torch.Tensor:
    metric = ConfusionMatrix(task='multiclass',
                             num_classes=num_classes
    ).to(device)
    return metric(preds, target)

def compute_mcc(preds: torch.Tensor,
                target: torch.Tensor,
                device: torch.device,
                num_classes: int = 10) -> torch.Tensor:
    metric = MatthewsCorrCoef(
        task='multiclass',
        num_classes=num_classes
    ).to(device)
    return metric(preds, target)

def compute_auc_roc(preds: torch.Tensor,
                    target: torch.Tensor,
                    device: torch.device,
                    num_classes: int = 10,
                    average: str = 'macro') -> torch.Tensor:
    metric = AUROC(task='multiclass',
                   num_classes=num_classes,
                   average=average
    ).to(device)
    return metric(preds, target)

def compute_auc_pr(preds: torch.Tensor,
                   target: torch.Tensor,
                   device: torch.device,
                   num_classes: int = 10,
                   average: str = 'macro') -> torch.Tensor:
    metric = AveragePrecision(task='multiclass',
                              num_classes=num_classes,
                              average=average
    ).to(device)
    return metric(preds, target)