import torch
import sys
import shutil
import os
from tqdm import tqdm
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

from Model.model import VIT_Model


transform_train = transforms.Compose([
   transforms.Lambda(lambda img: img.convert("RGB")),
   transforms.Resize((72, 72)),
   transforms.RandomHorizontalFlip(),
   transforms.RandomCrop(64, padding=4),
   transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
   transforms.ToTensor(),
   transforms.Normalize(
      mean=[0.485,0.456,0.406],
      std=[0.229,0.224,0.225]
   )
])

transform_val = transforms.Compose([
   transforms.Lambda(lambda img: img.convert("RGB")),
   transforms.Resize((64,64)),
   transforms.ToTensor(),
   transforms.Normalize(
      mean=[0.485,0.456,0.406],
      std=[0.229,0.224,0.225]
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

if __name__ == "__main__":

   ds = load_dataset("zh-plus/tiny-imagenet")
   train_ds = ds["train"].with_transform(apply_transforms_train)
   val_ds   = ds["valid"].with_transform(apply_transforms_val)

   train_loader = DataLoader(train_ds, batch_size=64,  shuffle=True, num_workers=2, pin_memory=True)
   val_loader = DataLoader(val_ds, batch_size=64, num_workers=2, pin_memory=True)

   model = VIT_Model(
      img_size=(3, 64, 64),
      patch_size=4,
      token_len=384,
      preds=200,
      num_heads=6,
      Encoding_hidden_chan_mul=4.,
      depth=8
   )
   device = "cuda" if torch.cuda.is_available() else "cpu"
   model = model.to(device)

   print(f"Device: {device}")
   print(f"CUDA available: {torch.cuda.is_available()}")

   # Loss and Optimizer
   criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.2)
   optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

   checkpoint_dir = "/kaggle/working/checkpoints"
   os.makedirs(checkpoint_dir, exist_ok=True)
   best_f1 = 0.0

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

         optimizer.zero_grad()
         preds = model(imgs)
         loss = criterion(preds, labels)

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

            #top 1
            predicted = preds.argmax(dim=1)
            correct_top1 += (predicted == labels).sum().item()

            #top 5
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


      # Каждая 3-я эпоха
      if (epoch + 1) % 3 == 0:
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
         os.replace(
            tmp,
            f"{checkpoint_dir}/best_checkpoint.pt"
         )

         print("Best checkpoint updated")
