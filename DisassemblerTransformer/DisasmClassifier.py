import os
from transformers import AutoConfig, TFAutoModelForSequenceClassification, AutoTokenizer

class DisasmClassifier():
    def __init__(self, modelType: str) -> None:
        self.model              = TFAutoModelForSequenceClassification.from_pretrained(modelType)
        self.tokenizer          = AutoTokenizer.from_pretrained(modelType)
        self.config             = AutoConfig.from_pretrained(modelType)
    

    def classifyInput(self, filePath: str) -> None | str:
        # Make sure what was passed in is actually a file
        if os.path.isdir(filePath):
            return None
        else:
            # Load and prepare text from file
            with open(filePath, 'r') as fileBuffer: text = fileBuffer.read()
            text = text.split(sep='\n')
            # Tokenize input and make prediction
            tokenizedInput = self.tokenizer(text, padding=True, truncation=True, return_tensors="tf")
            logits = self.model.predict(tokenizedInput)[0]
            prediction = logits.argmax(axis=-1)
            # Return string representation of classification
            return self.config.id2label[prediction[0]]


    def outputEncoder(self, filePath: str) -> None:
        # Make sure what was passed in is actually a file
        if os.path.isdir(filePath):
            return None
        else:
            # Load and prepare text from file
            with open(filePath, 'r') as fileBuffer: text = fileBuffer.read()
            text = text.split(sep='\n')
            # Tokenize and encode text data
            tokenizedText = self.tokenizer(text, padding=True, truncation=True, return_tensors="tf")
            tensors = self.model(**tokenizedText, output_hidden_states=True)
            finalTensor = tensors.hidden_states[-1]
            del tensors # Need this or run OOM a lot easier
            return finalTensor
