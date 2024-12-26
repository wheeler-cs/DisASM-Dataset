import os
from transformers import AutoConfig, TFAutoModelForSequenceClassification, AutoTokenizer

class DisasmClassifier():
    def __init__(self, modelType: str, dataDir: str) -> None:
        self.dataDirectory: str = dataDir
        self.model              = TFAutoModelForSequenceClassification.from_pretrained(modelType)
        self.tokenizer          = AutoTokenizer.from_pretrained(modelType)
        self.config             = AutoConfig.from_pretrained(modelType)
    

    def classifyInput(self) -> None:
        for file in os.listdir(self.dataDirectory):
            fullPathName = os.path.join(self.dataDirectory, file)
            if os.path.isdir(fullPathName):
                continue
            else:
                with open(fullPathName, 'r') as fileBuffer:
                    text = fileBuffer.read()
                text = text.split(sep='\n')
                tokenizedInput = self.tokenizer(text, padding=True, truncation=True, return_tensors="tf")
                print(f"File: {file}")
                logits = self.model.predict(tokenizedInput)[0]
                prediction = logits.argmax(axis=-1)
                print(self.config.id2label[prediction[0]])


    def outputEncoder(self) -> None:
        for file in os.listdir(self.dataDirectory):
            fullPathName = os.path.join(self.dataDirectory, file)
            if os.path.isdir(fullPathName):
                continue
            else:
                with open(fullPathName, 'r') as fileBuffer:
                    text = fileBuffer.read()
                text = text.split(sep='\n')
                tokenizedInput = self.tokenizer(text, padding=True, truncation=True, return_tensors="tf")
                output = self.model(**tokenizedInput, output_hidden_states=True)
                hidden_state = output.hidden_states[-1]
                del output
                print(f"\nFile: {file}")
                print(hidden_state)
