from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from math import floor
import numpy as np
from os import listdir, path
import pyarrow as pa
from tqdm import tqdm
from typing import Dict, List, Tuple


class DisasmDataLoader():
    def __init__(self, inputDir: str = "./data"):
        self.dataDirectory = inputDir
        self.datasets: DisasmDataset = list()
        self.loadDataset()


    def loadDataset(self):
        if (len(self.dataDirectory) < 1):
            raise ValueError("Invalid data directory provided")
        # Query target directory for classes based on directory names
        self.getClassesFromDirectories()
    

    def getClassesFromDirectories(self):
        print("    Indexing files for training...")
        # Assume the name of the directory is the classification of the data
        for inode in listdir(self.dataDirectory):
            fullPath = path.join(self.dataDirectory, inode)
            if(path.isdir(fullPath)):
                data = self.loadNumpyArrays(fullPath)
                self.datasets.append(DisasmDataset(inode, len(self.datasets), data))

    
    def loadNumpyArrays(self, dataDirectory: str) -> List[np.array]:
        arrayList = list()
        # Iterate through files in the directory and load data from them
        for inode in tqdm(listdir(dataDirectory)):
            fullPath = path.join(dataDirectory, inode)
            if(path.isfile(fullPath)):
                try:
                    # Load the array and store it
                    txtArray = []
                    # NumPy won't let me use a newline as a delimiter...
                    # this is what I'm forced to do instead
                    with open(fullPath) as txtFile:
                        for line in txtFile:
                            txtArray.append(line[:-1])
                    npArray = np.array(txtArray)
                    arrayList.append(npArray)
                except Exception as e: # Just ignore invalid files, it's probably okay
                    print(f"Unable to load {inode} ({e})")
        return arrayList
    

    def getDatasetDict(self) -> DatasetDict:
        compiledTraining = []
        compiledTesting = []
        for database in self.datasets:
            newDataset = []
            for element in database.data:
                # The np.array2string below is awful... Here's how it works:
                # The np.array is converted into a string in the format "['a' 'b' 'c' ... 'z']" for tokenization
                # The two braces at the ends of the string are removed and all ' are removed.
                # I did this to just get a really long string of function calls.
                newDataset.append({"label": database.encodedClass, "sequence": np.array2string(element, max_line_width=1_000_000_000)[1:-1].replace('\'', '')})
            train, test = splitDataset(newDataset, 0.8)
            compiledTraining += train
            compiledTesting += test
        trainDataset = Dataset.from_list(compiledTraining)
        testDataset = Dataset.from_list(compiledTesting)
        return DatasetDict({"train": trainDataset, "test": testDataset})


    def createLabelIdMappings(self) -> Tuple[Dict[int, str], Dict[int, str]]:
        enumList = enumerate(self.datasets)
        id2label = dict()
        label2id = dict()
        for enumeration in enumList:
            id2label[enumeration[1].classification] = enumeration[0]
            label2id[enumeration[0]] = enumeration[1].classification
        return id2label, label2id


class DisasmDataset(object):
    def __init__(self, classification: str, encodedClass: int, data: List[np.array]):
        self.classification: str = classification
        self.encodedClass: int = encodedClass
        self.data: List[np.array] = data


def splitDataset(dataset: List, percentage: float = 0.5) -> Tuple[List, List]:
    if (0 <= percentage <= 1):
        splitIdx = floor(len(dataset) * percentage)
        return dataset[:splitIdx], dataset[splitIdx:]
    else:
        raise ValueError("Percentage must be a decimal between 0 and 1")


if __name__ == "__main__":
    loader = DisasmDataLoader()
    datasetDictionary = loader.getDatasetDict()
    print(datasetDictionary)
