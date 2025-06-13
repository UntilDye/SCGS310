# train.py 
import os
import yaml
import torch
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from models.yolo_vehicle import VehicleYOLO
from data.aic_dataset import AICDataset, aic_collate_fn
from utils.loss_compute import YOLOLoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.early_stopping import EarlyStopping

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler


class TrainingConfig:
    """训练配置类"""
    def __init__(self, config_dict):
        self.config = config_dict
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建必要的目录
        self.model_dir = Path(config_dict['training']['save_model_path'])
        self.logs_dir = Path(config_dict['training']['logs_dir'])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志记录"""
        # 清除现有的处理器
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        # 创建带时间戳的日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_dir / f'training_{timestamp}.log'
        
        # 配置日志格式
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        
        # 配置日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, mode='w', encoding='utf-8'),
                logging.StreamHandler()
            ],
            force=True  # 强制重新配置
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"日志文件: {log_file}")
        self.logger.info(f"使用设备: {self.device}")


class DataManager:
    """数据管理类"""
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def create_data_loaders(self):
        """创建数据加载器"""
        aic_classes = {'car': 0, 'truck': 1, 'bus': 2, 'motorbike': 3}
        
        # 创建数据集
        train_dataset = AICDataset(
            images_dir=self.config['dataset']['train'],
            labels_dir=self.config['dataset']['train_labels'],
            img_size=self.config['model']['input_size'][0],
            class_mapping=aic_classes,
            is_training=True
        )
        
        val_dataset = AICDataset(
            images_dir=self.config['dataset']['val'],
            labels_dir=self.config['dataset']['val_labels'], 
            img_size=self.config['model']['input_size'][0],
            class_mapping=aic_classes,
            is_training=False
        )
        
        self.logger.info(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
        
        # 配置数据加载器参数
        num_workers = min(self.config['training'].get('num_workers', 4), os.cpu_count())
        pin_memory = torch.cuda.is_available()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            collate_fn=aic_collate_fn,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training'].get('val_batch_size', self.config['training']['batch_size']),
            shuffle=False,
            num_workers=max(1, num_workers // 2),
            collate_fn=aic_collate_fn,
            pin_memory=pin_memory,
            persistent_workers=max(1, num_workers // 2) > 0
        )
        
        return train_loader, val_loader


class Trainer:
    """训练器类"""
    def __init__(self, config, model, criterion, optimizer, scheduler, scaler):
        self.config = config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(__name__)
        
        # 训练参数
        self.accumulate_steps = config['training'].get('accumulate_grad_batches', 2)
        self.use_amp = config['training'].get('mixed_precision', False) and self.device.type == 'cuda'
        
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        box_losses, obj_losses, cls_losses = [], [], []
        
        # 训练初期禁用混合精度
        use_amp = self.use_amp and epoch >= 3
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        self.optimizer.zero_grad()
        
        for batch_idx, (images, targets, _) in pbar:
            try:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # 检查输入有效性
                if torch.isnan(images).any() or torch.isinf(images).any():
                    self.logger.warning(f"批次 {batch_idx} 输入无效，跳过")
                    continue
                
                with autocast(device_type=self.device.type, enabled=use_amp):
                    predictions = self.model(images)
                    loss, loss_items = self.criterion(predictions, targets)
                    
                    # 检查损失有效性
                    if torch.isnan(loss) or torch.isinf(loss) or loss > 50.0:
                        self.logger.warning(f"批次 {batch_idx} 损失无效: {loss}，跳过")
                        self.optimizer.zero_grad()
                        continue
                    
                    if self.accumulate_steps > 1:
                        loss = loss / self.accumulate_steps
                
                # 记录损失
                if len(loss_items) >= 3:
                    box_losses.append(loss_items[0].item())
                    obj_losses.append(loss_items[1].item())
                    cls_losses.append(loss_items[2].item())
                
                # 反向传播
                if use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # 更新参数
                should_step = (batch_idx + 1) % self.accumulate_steps == 0 or (batch_idx + 1) == len(train_loader)
                
                if should_step:
                    self._update_parameters(use_amp)
                
                total_loss += loss.item() * self.accumulate_steps
                
                # 更新进度条
                if len(loss_items) >= 3:
                    pbar.set_postfix({
                        'loss': f"{loss.item() * self.accumulate_steps:.4f}",
                        'box': f"{loss_items[0].item():.3f}",
                        'obj': f"{loss_items[1].item():.3f}",
                        'cls': f"{loss_items[2].item():.3f}",
                        'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                    })
                
            except Exception as e:
                self.logger.error(f"批次 {batch_idx} 错误: {e}")
                self.optimizer.zero_grad()
                if use_amp:
                    self.scaler.update()
                continue
        
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        
        return {
            'total_loss': avg_loss,
            'box_loss': sum(box_losses) / len(box_losses) if box_losses else 0,
            'obj_loss': sum(obj_losses) / len(obj_losses) if obj_losses else 0,
            'cls_loss': sum(cls_losses) / len(cls_losses) if cls_losses else 0
        }
    
    def _update_parameters(self, use_amp):
        """更新模型参数"""
        max_grad_norm = 1.0
        
        if use_amp:
            try:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                
                if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 10.0:
                    self.logger.warning(f"梯度范数异常: {grad_norm}，跳过更新")
                    self.optimizer.zero_grad()
                else:
                    self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            except RuntimeError as e:
                self.logger.warning(f"梯度处理错误: {e}")
                self.optimizer.zero_grad()
                self.scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm > 10.0:
                self.logger.warning(f"梯度范数异常: {grad_norm}，跳过更新")
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        valid_batches = 0
        box_losses, obj_losses, cls_losses = [], [], []
        
        with torch.no_grad():
            for batch_idx, (images, targets, img_paths) in enumerate(tqdm(val_loader, desc="Validating")):
                try:
                    if images.shape[0] == 0:
                        continue
                        
                    images = images.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    
                    with autocast(device_type=self.device.type, enabled=self.use_amp):
                        predictions = self.model(images)
                        loss, loss_items = self.criterion(predictions, targets)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning(f"验证批次 {batch_idx} 损失无效: {loss}")
                        continue
                    
                    total_loss += loss.item()
                    valid_batches += 1
                    
                    if len(loss_items) >= 3:
                        box_losses.append(loss_items[0].item())
                        obj_losses.append(loss_items[1].item())
                        cls_losses.append(loss_items[2].item())
                    
                except Exception as e:
                    self.logger.error(f"验证批次 {batch_idx} 错误: {e}")
                    continue
        
        if valid_batches == 0:
            self.logger.warning("没有有效的验证批次!")
            return {'total_loss': float('inf'), 'box_loss': 0, 'obj_loss': 0, 'cls_loss': 0}
        
        return {
            'total_loss': total_loss / valid_batches,
            'box_loss': sum(box_losses) / len(box_losses) if box_losses else 0,
            'obj_loss': sum(obj_losses) / len(obj_losses) if obj_losses else 0,
            'cls_loss': sum(cls_losses) / len(cls_losses) if cls_losses else 0
        }


def main():
    parser = argparse.ArgumentParser(description='训练车辆检测模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--resume', type=str, default='', help='从检查点恢复训练')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 初始化配置
    config = TrainingConfig(config_dict)
    logger = config.logger
    
    # 优化CUDA设置
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
    
    # 创建数据加载器
    data_manager = DataManager(config_dict)
    train_loader, val_loader = data_manager.create_data_loaders()
    
    # 创建模型
    model = VehicleYOLO(
        num_classes=config_dict['model']['num_classes'],
        input_shape=tuple(config_dict['model']['input_size'])
    ).to(config.device)
    
    # 创建损失函数
    anchors = [
        [[10, 13], [16, 30], [33, 23]],
        [[30, 61], [62, 45], [59, 119]],
        [[116, 90], [156, 198], [373, 326]]
    ]
    strides = [8, 16, 32]
    
    criterion = YOLOLoss(
        num_classes=config_dict['model']['num_classes'],
        anchors=anchors,
        strides=strides,
        loss_weights=config_dict['training'].get('loss_weights', {"box": 0.05, "obj": 1.0, "cls": 0.5}),
        size_penalty_weight=config_dict['training'].get('size_penalty_weight', 0.1),
        max_box_ratio=config_dict['training'].get('max_box_ratio', 0.7)
    )
    
    # 创建优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config_dict['training']['learning_rate'],
        weight_decay=config_dict['training'].get('weight_decay', 0.0005),
        eps=1e-8
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    scaler = GradScaler(enabled=config_dict['training'].get('mixed_precision', False))
    
    # 创建训练器
    trainer = Trainer(config_dict, model, criterion, optimizer, scheduler, scaler)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(config.logs_dir))
    
    # 恢复训练
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
        logger.info(f"从 epoch {start_epoch} 恢复训练")
    
    # 早停和验证间隔
    early_stopping = EarlyStopping(
        patience=config_dict['training'].get('early_stopping_patience', 15),
        min_delta=config_dict['training'].get('early_stopping_min_delta', 0.001)
    )
    val_interval = config_dict['training'].get('val_interval', 5)
    save_interval = config_dict['training'].get('save_interval', 10)
    
    # 训练循环
    for epoch in range(start_epoch, config_dict['training']['num_epochs']):
        start_time = time.time()
        
        try:
            # 训练
            train_metrics = trainer.train_epoch(train_loader, epoch)
            
            # 验证
            if epoch % val_interval == 0:
                val_metrics = trainer.validate(val_loader)
                
                # 记录日志
                epoch_time = time.time() - start_time
                logger.info(
                    f'Epoch {epoch}: '
                    f'Train Loss: {train_metrics["total_loss"]:.4f} '
                    f'(box: {train_metrics["box_loss"]:.3f}, obj: {train_metrics["obj_loss"]:.3f}, cls: {train_metrics["cls_loss"]:.3f}), '
                    f'Val Loss: {val_metrics["total_loss"]:.4f} '
                    f'(box: {val_metrics["box_loss"]:.3f}, obj: {val_metrics["obj_loss"]:.3f}, cls: {val_metrics["cls_loss"]:.3f}), '
                    f'Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]["lr"]:.6f}'
                )
                
                # TensorBoard记录
                writer.add_scalars('Loss/Total', {
                    'train': train_metrics["total_loss"], 
                    'val': val_metrics["total_loss"]
                }, epoch)
                
                writer.add_scalars('Loss/Detail', {
                    'train_box': train_metrics["box_loss"], 
                    'train_obj': train_metrics["obj_loss"], 
                    'train_cls': train_metrics["cls_loss"],
                    'val_box': val_metrics["box_loss"], 
                    'val_obj': val_metrics["obj_loss"], 
                    'val_cls': val_metrics["cls_loss"]
                }, epoch)
                
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
                
                # 调度器更新
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics["total_loss"])
                else:
                    scheduler.step()
                
                # 保存最佳模型
                if val_metrics["total_loss"] < best_loss:
                    best_loss = val_metrics["total_loss"]
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'loss': best_loss,
                        'config': config_dict
                    }, config.model_dir / 'best_vehicle_model.pth')
                    logger.info(f"保存最佳模型，损失: {best_loss:.4f}")
                
                # 早停检查
                if early_stopping(val_metrics["total_loss"], model):
                    logger.info(f"在 epoch {epoch} 触发早停")
                    early_stopping.restore_best_weights_to_model(model)
                    break
            else:
                epoch_time = time.time() - start_time
                logger.info(
                    f'Epoch {epoch}: Train Loss: {train_metrics["total_loss"]:.4f}, '
                    f'Time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]["lr"]:.6f}'
                )
                if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()
            
            # 定期保存检查点
            if epoch > 0 and epoch % save_interval == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': train_metrics["total_loss"],
                    'config': config_dict
                }, config.model_dir / f'checkpoint_epoch_{epoch}.pth')
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Epoch {epoch} 发生错误: {e}", exc_info=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config_dict
    }, config.model_dir / 'final_vehicle_model.pth')
    
    writer.close()
    logger.info("训练完成!")


if __name__ == '__main__':
    main()