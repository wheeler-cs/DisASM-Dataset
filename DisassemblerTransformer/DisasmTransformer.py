from DisassemblerTransformer.DisasmDataLoader import DisasmDataLoader

import evaluate
import numpy as np
from os import path
import pandas as pd
import tensorflow as tf
from transformers import create_optimizer, DataCollatorWithPadding, RobertaTokenizer, TFAutoModelForSequenceClassification
from transformers.keras_callbacks import KerasMetricCallback
from typing import Dict

# Tokenizer and collator to be used later
gTokenizer = None
gDataCollator = None


def createTokenization(data):
        return gTokenizer(data["sequence"], truncation=True)


def computeMetrics(evalPrediction) -> None:
    accuracy = evaluate.load("accuracy")
    predictions, labels = evalPrediction
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


class DisasmTransformer():
    def __init__(self, modelType: str, dataDir: str, modelSavePath: str = "", batchSize: int = 8, epochs: int = 5, saveTraining: bool = False):
        # Data mapping and tokenization
        self.dataDirectory:          str = dataDir
        self.dataLoader: DisasmDataLoader = DisasmDataLoader(dataDir)
        self.id2label:    Dict[int, str] = dict()
        self.label2id:    Dict[str, int] = dict()
        self.tokenizedData = None
        global gTokenizer
        gTokenizer = RobertaTokenizer.from_pretrained(modelType)
        global gDataCollator
        gDataCollator = DataCollatorWithPadding(tokenizer=gTokenizer, return_tensors="tf")
        self.callDataLoader()
        # Model parameters
        self.batchSize:       int = batchSize
        self.epochs:          int = epochs
        self.batchesPerEpoch: int = len(self.tokenizedData["train"]) // batchSize
        self.trainingSteps:   int = int(self.batchesPerEpoch * self.epochs)
        self.optimizer, self.schedule = create_optimizer(init_lr=2e-5,
                                                         num_warmup_steps=0,
                                                         num_train_steps=self.trainingSteps)
        # Transformer model
        self.model = TFAutoModelForSequenceClassification.from_pretrained(modelType,
                                                                          num_labels=2,
                                                                          id2label=self.id2label,
                                                                          label2id=self.label2id)
        self.trainingSet = None
        self.testingSet  = None
        # Training evaluation
        self.modelPath: str = modelSavePath
        self.saveTraining = saveTraining
        self.trainingHistory = None
    

    def callDataLoader(self):
        print("    Mapping datasets for dictionary...")
        dsDict = self.dataLoader.getDatasetDict()
        self.tokenizedData = dsDict.map(createTokenization, batched=True)
        self.label2id, self.id2label = self.dataLoader.createLabelIdMappings()
        self.batchesPerEpoch = len(self.tokenizedData)
    

    def prepareDatasets(self) -> None:
        print("    Finalizing dataset preparation...")
        self.trainingSet = self.model.prepare_tf_dataset(self.tokenizedData["train"],
                                                         shuffle=True,
                                                         batch_size=self.batchSize,
                                                         collate_fn=gDataCollator)
        self.testingSet  = self.model.prepare_tf_dataset(self.tokenizedData["test"],
                                                         shuffle=False,
                                                         batch_size=self.batchSize,
                                                         collate_fn=gDataCollator)


    def prepareModel(self) -> None:
        print("    Compiling Model for Training...")
        self.model.compile(optimizer=self.optimizer)
    

    def trainModel(self) -> None:
        metricCallback = KerasMetricCallback(metric_fn=computeMetrics, eval_dataset=self.testingSet)
        self.trainingHistory = self.model.fit(x=self.trainingSet,
                                              validation_data=self.testingSet,
                                              epochs=self.epochs,
                                              callbacks = [metricCallback])
        if(self.saveTraining):
            self.saveTrainingResults()
        if(self.modelPath != ""):
             self.model.save_pretrained(self.modelPath)
    

    def vectorizeInput(self, filePath: str) -> None:
        # Load file into array
        if(path.isfile(filePath)):
            txtArray = []
            with open(filePath) as txtFile:
                for line in txtFile:
                    txtArray.append(line[:-1])
            npArray = np.array(txtArray)
        else:
             raise Exception("Unable to load input for vectorization")
        # Prepare data for encoding
        tokenizedInput = gTokenizer(npArray, truncation=True)
        output = self.model(tokenizedInput)
        print(output)


    def saveTrainingResults(self, outFile: str = "results.csv") -> None:
        historyFrame = pd.DataFrame(self.trainingHistory.history)
        with open(outFile, 'w') as resultsFile:
                historyFrame.to_csv(resultsFile)
    

    def loadTrainingResults(self, inFile: str) -> None:
         self.model = TFAutoModelForSequenceClassification.from_pretrained(inFile,
                                                                           num_labels=2,
                                                                           id2label=self.id2label,
                                                                           label2id=self.label2id)



if __name__ == "__main__":
     print("This module cannot be ran as a stand-alone script")
