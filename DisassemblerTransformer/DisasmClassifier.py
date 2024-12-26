import json
import os
from transformers import AutoConfig, TFAutoModelForSequenceClassification, AutoTokenizer
from typing import Dict

class DisasmClassifier():
    def __init__(self, modelType: str, dataDir: str) -> None:
        self.dataDirectory: str = dataDir
        self.model = TFAutoModelForSequenceClassification.from_pretrained(modelType)
        self.tokenizer = AutoTokenizer.from_pretrained(modelType)
        self.config = AutoConfig.from_pretrained(modelType)
    

    def classifyInput(self) -> None:
        for file in os.listdir(self.dataDirectory):
            fullPathName = os.path.join(self.dataDirectory, file)
            if os.path.isdir(fullPathName):
                continue
            else:
                with open(fullPathName, 'r') as fileBuffer:
                    text = fileBuffer.read()
                tokenizedInput = self.tokenizer(text, padding=True, truncation=True, return_tensors="tf")
                logits = self.model.predict(tokenizedInput)[0]
                prediction = logits.argmax(axis=-1)
                print(self.config.id2label[prediction[0]])
