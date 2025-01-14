from DisassemblerTransformer.DisasmDataLoader import DisasmDataLoader

import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from typing import Dict


# --- Global Variables -------------------------------------------------------------------------------------------------
gTokenizer    = None
gDataCollator = None


# --- Functions --------------------------------------------------------------------------------------------------------
def createTokenization(data):
        return gTokenizer(data["sequence"], padding=True, truncation=True, return_tensors="pt")


def computeMetrics(evalPrediction) -> None:
    accuracy = evaluate.load("accuracy")
    predictions, labels = evalPrediction
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# --- DisasmTransformer Class ------------------------------------------------------------------------------------------
class DisasmTransformer():
    def __init__(self, modelType: str, dataDir: str, savePath: str = "disasmTformer", batchSize: int = 8, epochs: int = 5):
        # Model parameters
        self.modelType = modelType
        self.dataDir   = dataDir
        self.savePath  = savePath
        self.batchSize = batchSize
        self.epochs    = epochs
        self.setGlobals()
        # Load dataset
        self.id2label:    Dict[int, str] = dict()
        self.label2id:    Dict[str, int] = dict()
        self.tokenizedData = None
        self.callDataLoader()
        # Init model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.modelType,
                                                                        num_labels=len(self.id2label),
                                                                        id2label=self.id2label,
                                                                        label2id=self.label2id)
        self.trainer = None
        self.initTrainer()


    def setGlobals(self) -> None:
         global gTokenizer
         gTokenizer = AutoTokenizer.from_pretrained(self.modelType)
         global gDataCollator
         gDataCollator = DataCollatorWithPadding(tokenizer=gTokenizer)
    

    def callDataLoader(self) -> None:
        dataLoader = DisasmDataLoader(self.dataDir)
        setDict = dataLoader.getDatasetDict()
        self.tokenizedData = setDict.map(createTokenization, batched=True)
        self.label2id, self.id2label = dataLoader.createLabelIdMappings()
    

    def initTrainer(self) -> None:
        trainArgs = TrainingArguments(output_dir=self.savePath,
                                       learning_rate=2e-5,
                                       per_device_train_batch_size=self.batchSize,
                                       per_device_eval_batch_size=self.batchSize,
                                       num_train_epochs=self.epochs,
                                       weight_decay=0.01,
                                       eval_strategy="epoch",
                                       save_strategy="epoch")
        self.trainer = Trainer(model=self.model,
                                args=trainArgs,
                                train_dataset=self.tokenizedData["train"],
                                eval_dataset=self.tokenizedData["test"],
                                processing_class=gTokenizer,
                                data_collator=gDataCollator,
                                compute_metrics=computeMetrics)

    
    def trainModel(self) -> None:
         self.trainer.train()
         self.model.save_pretrained(self.savePath)
         global gTokenizer
         gTokenizer.save_pretrained(self.savePath)
