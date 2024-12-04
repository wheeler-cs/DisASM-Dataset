import sys
sys.path += ["DisassemblerPipeline", "DisassemblerTransformer"]

from DisassemblerPipeline.Disassembler import Disassembler
from DisassemblerTransformer.DisasmTransformer import DisasmTransformer

import argparse
import os
from typing import List


# === General Functions ================================================================================================
def parseArgv() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="Disassembly Runner",
                                     description="Perform dataset generation or transformer training")
    # Valid command-line arguments
    parser.add_argument("-b", "--batchsize",
                        help="The batch size used during training",
                        type=int,
                        required=False,
                        default=32)
    parser.add_argument("-e", "--epochs",
                        help="The number of epochs the transformer will be trained over",
                        type=int,
                        required=False,
                        default=5)
    parser.add_argument("-m", "--mode",
                        help="The mode of operation the program should run in",
                        choices=["generator", "transformer"],
                        type=str,
                        required=True)
    parser.add_argument("-i", "--input",
                        help="Target directory for input files",
                        type=str,
                        required=True)
    parser.add_argument("-l", "--limit",
                        help="Limits the number of instructions saved from disassembly",
                        type=int,
                        required=False,
                        default=10_000)
    parser.add_argument("-x", "--extension",
                        help="File extension to be targeted by disassembler",
                        type=str,
                        required=False,
                        default=".exe")
    return parser.parse_args()


def createFileList(directory: str, extension: str) -> List[str]:
    if(extension == ""):
        raise ValueError("Target file extension cannot be empty")
    retList = list()
    for inode in os.listdir(directory):
        if(os.path.isfile(os.path.join(directory, inode))):
            # Wildcard gets all file extensions
            # Otherwise, make sure the extension specified is chosen
            if(extension == "*") or (inode[len(extension) * -1:] == extension):
                retList.append(os.path.join(directory, inode))
    return retList


def createOutDirectory(originalDir: str, subdir: str) -> None:
    os.makedirs(os.path.join(originalDir, subdir), exist_ok=True)


# === Generator Mode Functions =========================================================================================
def callDisassembler(argv: argparse.Namespace):
    fileList = createFileList(argv.input, argv.extension)
    createOutDirectory(argv.input, "disasm")
    print("[RUNNING DISASSEMBLER]")
    disasm: Disassembler = Disassembler(instructionLimit=argv.limit)
    disasm.processList(fileList, argv.input)


# === Transformer Mode Functions =======================================================================================
def callTransformer(argv: argparse.Namespace):
    print("[RUNNING TRANSFORMER]")
    dt = DisasmTransformer(argv.input, argv.batchsize, argv.epochs)
    dt.prepareDatasets()
    dt.prepareModel()
    dt.trainModel()




# === main =============================================================================================================
if __name__ == "__main__":
    argv: argparse.Namespace = parseArgv()

    if(argv.mode == "generator"):
        callDisassembler(argv)
    elif(argv.mode == "transformer"):
        callTransformer(argv)
    else:
        raise ValueError("Bad mode of operation passed into program")
