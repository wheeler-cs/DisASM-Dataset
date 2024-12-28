'''
@package DisasmClassifier

Module providing facilities for leveraging a pre-trained model in the task of assembly classification.
'''


import argparse
import logging
import os
import tensorflow as tf
from transformers import AutoConfig, TFAutoModelForSequenceClassification, AutoTokenizer


# == DisasmClassifier ==================================================================================================
class DisasmClassifier(object):
    '''
    Classifier for determining the associated classification of a given disassembly file. This class utilizes a
    sequence-based classifier derived from the Facebook AI RoBERTa model available on Hugging Face.
    '''

    def __init__(self, modelType: str) -> None:
        '''
        Class initializer.

        @param self Pointer to calling class instance.
        @param modelType String defining the model from Hugging Face to use for classification. It is recommended that
               this is a model that has been pre-trained on the associated classification task.
        '''
        self.model     = TFAutoModelForSequenceClassification.from_pretrained(modelType)
        self.tokenizer = AutoTokenizer.from_pretrained(modelType)
        self.config    = AutoConfig.from_pretrained(modelType)
    

    def classifyInput(self, filePath: str) -> None | str:
        '''
        Utilize a pre-trained model to obtain a classification for a given disassembly file.

        @param self Pointer to calling class instance.
        @param filePath Path to disassembly file that will be classified.

        @returns String representation of classification determined by pre-trained model.
        @retval None Path provided is a directory.

        @pre All member variables of calling class instance have been properly initialized.
        @see self.__init__()
        '''
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


    def outputEncoder(self, filePath: str) -> None | tf.Tensor:
        '''
        Utilize a pre-trained model in encoder-only mode to generate a tensor representation of an assembly file.

        @param self Pointer to calling class instance.
        @param filePath Path to disassembly file to be converted into a tensor.

        @returns Encoded tensor of provided assembly file.
        @retval None Path provided is a directory.

        @pre All member variables of calling class instance have been properly initialized.
        @see self.__init__()
        '''
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
        

# == main() ============================================================================================================
if __name__ == "__main__":
    # Provide command-line arguments so script can run by itself when needed.
    parser = argparse.ArgumentParser(prog="Disassembly Classifier",
                                     description="Standalone mode of operation for the DisasmClassifier module.")
    parser.add_argument("--model",
                        help="Name of Hugging Face or local model to be used with classifier.",
                        type=str,
                        required=True)
    parser.add_argument("--path",
                        help="Path of file to be used for classification or encoding.",
                        type=str,
                        required=True)
    parser.add_argument("--mode",
                        help="The operation the script should perform during execution.",
                        type=str,
                        choices=["classifier", "encoder"],
                        required=True)
    parser = parser.parse_args()

    # Create class instance
    disasm = DisasmClassifier(parser.model)

    # Perform desired task
    match(parser.mode):
        case "classifier":
            classification = disasm.classifyInput(parser.path)
            print(classification)
        case "encoder":
            encoderTensor = disasm.outputEncoder(parser.path)
            print(encoderTensor)
