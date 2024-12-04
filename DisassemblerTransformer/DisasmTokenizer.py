from keras_nlp import tokenizers
import numpy as np
import tensorflow as tf
from typing import List


class DisasmTokenizer(tokenizers.Tokenizer):
    def tokenize(self, input: List[np.array]) -> List[List[str]]:
        tf.experimental.numpy.experimental_enable_numpy_behavior()
        tokenizedList = list()
        for element in input:
            tokenizedList.append(element.tolist())
        return tokenizedList


    def detokenize(self, input):
        # TODO... maybe
        raise NotImplemented("Class currently does not support detokenization")
