import torch
import torch.nn.functional as F
import os
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from huggingface_hub import upload_file
from sklearn.metrics import precision_score, recall_score, f1_score
from kaggle_secrets import UserSecretsClient
from Model.model import VIT_Model


transform_train = transforms.Compose([
   transforms.Lambda(lambda img: img.convert("RGB")),
   transforms.Resize((72, 72)),
   transforms.RandomHorizontalFlip(),
   transforms.RandomCrop(64, padding=4),
   transforms.RandAugment(num_ops=2, magnitude=9),
   transforms.ToTensor(),
   transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]
   )
])

transform_val = transforms.Compose([
   transforms.Lambda(lambda img: img.convert("RGB")),
   transforms.Resize((64, 64)),
   transforms.ToTensor(),
   transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]
   )
])

def apply_transforms_train(batch):
    batch["image"] = [transform_train(img) for img in batch["image"]]
    return batch

def apply_transforms_val(batch):
    batch["image"] = [transform_val(img) for img in batch["image"]]
    return batch

def top5_accuracy(preds, labels):
   top5 = preds.topk(5, dim=1).indices
   correct = top5.eq(labels.view(-1, 1)).sum().item()
   return correct

def mixup(imgs, labels, num_classes, alpha=0.4):
   lam = np.random.beta(alpha, alpha)
   idx = torch.randperm(imgs.size(0), device=imgs.device)
   mixed_imgs = lam * imgs + (1 - lam) * imgs[idx]
   labels_a = F.one_hot(labels, num_classes).float()
   labels_b = F.one_hot(labels[idx], num_classes).float()
   return mixed_imgs, lam * labels_a + (1 - lam) * labels_b

def cutmix(imgs, labels, num_classes, alpha=1.0):
   lam = np.random.beta(alpha, alpha)
   idx = torch.randperm(imgs.size(0), device=imgs.device)
   B, C, H, W = imgs.shape
   cut_ratio = np.sqrt(1 - lam)
   cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
   cx, cy = np.random.randint(W), np.random.randint(H)
   x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
   y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
   mixed_imgs = imgs.clone()
   mixed_imgs[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
   lam = 1 - (x2 - x1) * (y2 - y1) / (H * W)
   labels_a = F.one_hot(labels, num_classes).float()
   labels_b = F.one_hot(labels[idx], num_classes).float()
   return mixed_imgs, lam * labels_a + (1 - lam) * labels_b

if __name__ == "__main__":

   ds = load_dataset("zh-plus/tiny-imagenet")
   train_ds = ds["train"].with_transform(apply_transforms_train)
   val_ds   = ds["valid"].with_transform(apply_transforms_val)

   train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
   val_loader = DataLoader(val_ds, batch_size=64, num_workers=2, pin_memory=True)

   model = VIT_Model(
      img_size=(3, 64, 64),
      patch_size=4,
      token_len=384,
      preds=200,
      num_heads=6,
      Encoding_hidden_chan_mul=4.,
      depth=6
   )
   device = "cuda" if torch.cuda.is_available() else "cpu"
   model = model.to(device)

   print(f"Device: {device}")
   print(f"CUDA available: {torch.cuda.is_available()}")

   criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
   optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

   warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
      optimizer, start_factor=0.01, end_factor=1.0, total_iters=5
   )
   cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=45)
   scheduler = torch.optim.lr_scheduler.SequentialLR(
      optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5]
   )

   checkpoint_dir = "/kaggle/working/checkpoints"
   os.makedirs(checkpoint_dir, exist_ok=True)
   best_f1 = 0.0

   user_secrets = UserSecretsClient()
   hf_token = user_secrets.get_secret("HF_TOKEN")

   # Для чекпоинтов
   # checkpoint = torch.load('checkpoint_epoch15.pt', map_location=device)
   # model.load_state_dict(checkpoint['model_state_dict'])
   # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   # if 'scheduler_state_dict' in checkpoint:
   #    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
   # start_epoch = checkpoint['epoch']
   # print(f"Продолжаем с эпохи {start_epoch+1}")

#-------------------------------------
# Цикл обучения
#-------------------------------------

   for epoch in range(50):
      model.train()
      total_loss = 0

      for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
         imgs = batch["image"].to(device)
         labels = batch["label"].to(device)

         if random.random() < 0.5:
            imgs, mixed_labels = cutmix(imgs, labels, num_classes=200)
         else:
            imgs, mixed_labels = mixup(imgs, labels, num_classes=200)

         optimizer.zero_grad()
         preds = model(imgs)
         loss = criterion(preds, mixed_labels)

         loss.backward()
         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
         optimizer.step()

         total_loss += loss.item()

      print(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_loader):.4f}", flush=True)

      # Валидация
      model.eval()
      correct_top1 = 0
      correct_top5 = 0
      total = 0

      all_preds = []
      all_labels = []

      with torch.no_grad():
         for batch in val_loader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)

            preds = model(imgs)

            predicted = preds.argmax(dim=1)
            correct_top1 += (predicted == labels).sum().item()
            correct_top5 += top5_accuracy(preds, labels)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total += labels.size(0)

      acc_top1 = correct_top1 / total
      acc_top5 = correct_top5 / total

      precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
      recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
      f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
      scheduler.step()

      print(f"Validation Top‑1 Accuracy: {acc_top1:.4f}")
      print(f"Validation Top‑5 Accuracy: {acc_top5:.4f}")
      print(f"Precision (macro): {precision:.4f}")
      print(f"Recall (macro): {recall:.4f}")
      print(f"F1‑score (macro): {f1:.4f}")

      if (epoch + 1) % 5 == 0:
         torch.save(
            model.state_dict(),
            f"{checkpoint_dir}/epoch_{epoch+1}_weights.pt"
         )
         print(f"Saved epoch {epoch+1} weights")

      if f1 > best_f1:
         best_f1 = f1

         state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': total_loss / len(train_loader),
            'f1': f1,
         }

         tmp = f"{checkpoint_dir}/tmp.pt"
         torch.save(state, tmp)
         os.replace(tmp, f"{checkpoint_dir}/best_checkpoint.pt")

         print("Best checkpoint updated")

         upload_file(
            path_or_fileobj=f"{checkpoint_dir}/best_checkpoint.pt",
            path_in_repo="best_checkpoint.pt",
            repo_id="Krynovv/vit-tiny-imagenet",
            repo_type="model",
            token=hf_token
         )
         print("Uploaded best checkpoint to Hugging Face")
