import torch
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

from Model.model import VIT_Model

ds = load_dataset("zh-plus/tiny-imagenet")

transform = transforms.Compose([
   transforms.Resize((512,512)),
   transforms.ToTensor(),
])

def apply_transforms(batch):
   batch["pixel_values"] = [transform(img) for img in batch["image"]]
   return batch

train_ds = ds["train"].with_transform(apply_transforms)
val_ds = ds["valid"].with_transform(apply_transforms)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

model = VIT_Model(
   img_size=(3, 512, 512),
   patch_size=64,
   token_len=768,
   preds=200,
   num_heads=8,
   Encoding_hidden_chan_mul=4.,
   depth=12
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Loss and Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

def top5_accuracy(preds, labels):
   top5 = preds.topk(5, dim=1).indices
   correct = top5.eq(labels.view(-1, 1)).sum().item()
   return correct

#-------------------------------------
# Цикл обучения
#-------------------------------------
for epoch in range(10):
   model.train()
   total_loss = 0

   for batch in train_loader:
      imgs = batch["pixel_values"].to(device)
      labels = batch["label"].to(device)

      preds = model(imgs)
      loss = criterion(preds, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss += loss.item()

   print(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_loader):.4f}")

   # Валидация
   model.eval()
   correct_top1 = 0
   correct_top5 = 0
   total = 0

   all_preds = []
   all_labels = []

   with torch.no_grad():
      for batch in val_loader:
         imgs = batch["pixel_values"].to(device)
         labels = batch["label"].to(device)

         preds = model(imgs)

         #top 1
         predicted = preds.argmax(dim=1)
         correct_top1 += (predicted == labels).sum().item()

         #top 5
         correct_top5 += top5_accuracy(preds, labels)

         all_preds.extend(predicted.cpu().numpy())
         all_preds.extend(labels.cpu().numpy())

         total += labels.size(0)

   acc_top1 = correct_top1 / total
   acc_top5 = correct_top5 / total


   precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
   recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
   f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

   print(f"Validation Top‑1 Accuracy: {acc_top1:.4f}")
   print(f"Validation Top‑5 Accuracy: {acc_top5:.4f}")
   print(f"Precision (macro): {precision:.4f}")
   print(f"Recall (macro): {recall:.4f}")
   print(f"F1‑score (macro): {f1:.4f}")

