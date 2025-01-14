import torch
from torchvision.transforms import v2
from PIL import Image
import requests 
import os
import matplotlib.pyplot as plt
from utils import transforms
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet50,ResNet50_Weights

from fastapi import FastAPI,File,UploadFile
from pydantic import BaseModel

label_map={0:"cat",1:"dog",2:"bat",3:"aloo",4:"posto",5:"ELephant",6:"Horse",7:"Goat",8:"insect",9:"butterfly"}
#defining the model
num_classes=10
class resnet_transfer(L.LightningModule):
  def __init__(self):
    super().__init__()

    backbone=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_filterss=backbone.fc.in_features
    layers=list(backbone.children())[:-1]
    self.feature_extractor=nn.Sequential(*layers)
    self.classifier=nn.Linear(num_filterss,num_classes)

  def forward(self,x):
    with torch.no_grad():
      x=self.feature_extractor(x)
    x=torch.flatten(x,1)
    x=self.classifier(x)
    return x

  def training_step(self,batch,batch_idx):
    x,y=batch
    y_pred=self(x)
    loss=F.cross_entropy(y_pred,y)
    self.log("train_loss",loss)
    return loss

  def configure_optimizers(self):
    optimizer=optim.Adam(self.parameters(),lr=0.001)
    return optimizer

  def validation_step(self,batch,batch_idx):
    x,y=batch
    y_pred=self(x)
    loss=F.cross_entropy(y_pred,y)
    self.log("val_loss",loss)
    return loss




app = FastAPI()
model=resnet_transfer.load_from_checkpoint("model/model.ckpt")
model.eval()

@app.get("/")
def home():
    return {"details":"tingtong"}

@app.post("/predict")
def predict(image: UploadFile = File(...)):
    image = Image.open(image.file)
    image = transforms(image).unsqueeze(0)
    y_pred=model(image)
    y_pred=torch.argmax(y_pred,1)

    return {"prediction": label_map[y_pred.item()]}
  