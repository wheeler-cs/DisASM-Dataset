import sys
sys.path += ["DisassemblerPipeline", "DisassemblerTransformer"]

from DisassemblerPipeline.Disassembler import Disassembler
from DisassemblerPipeline.ProcessManager import ProcessManager

import argparse
from numpy import array_split, ndarray
import os
from tqdm import tqdm
from typing import List


# === General Functions ================================================================================================
def parseArgv() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="Disassembly Runner",
                                     description="Entrypoint for the disassembly-based transformer framework",
                                     formatter_class=argparse.RawTextHelpFormatter)
    # Shared arguments
    parser.add_argument("-m", "--mode",
                        help="The mode of operation the program should run in",
                        choices=["evaluator", "generator", "transformer"],
                        type=str,
                        required=True)
    parser.add_argument("-i", "--input",
                        help="Target directory for input files",
                        type=str,
                        required=True)
    parser.add_argument("-o", "--output",
                        help="[Transformer] Location to write model after training",
                        type=str,
                        required=False,
                        default="")
    parser.add_argument("-l", "--limit",
                        help="[Evaluator] Limits the number of different instructions seen before stopping\n" + 
                             "[Generator] Limits the number of instructions disassembled",
                        type=int,
                        required=False,
                        default=10_000)
    parser.add_argument("-s", "--summary",
                        help="[Evaluator] Write a summary of found commands to a file\n" +
                             "[Transformer] Write the results of training to a file",
                        required=False,
                        default=True)
    # Generator arguments
    parser.add_argument("-t", "--threads",
                        help="Specifies the number of threads to use for disassembly",
                        type=int,
                        required=False,
                        default=1)
    parser.add_argument("-x", "--extension",
                        help="File extension to be targeted by disassembler",
                        type=str,
                        required=False,
                        default=".exe")
    # Transformer arguments
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
    parser.add_argument("-fc", "--forcecpu",
                        help="Force transformer training to only use CPU",
                        required=False)
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


def generateSublists(inputList: List, divisions: int = 2) -> List[List]:
    # Make sure the list length and number of desired divisions are appropriate
    if(len(inputList) < divisions):
        raise ValueError(f"Number of desired divisions ({divisions}) is not appropriate for list size ({len(inputList)})")
    # Divide the list into $n$ sublists using numpy
    sublists = array_split(inputList, divisions)
    # Convert np arrays to lists
    for i in enumerate(sublists):
         sublists[i[0]] = ndarray.tolist(i[1])
    return sublists


def createOutDirectory(originalDir: str, subdir: str) -> None:
    os.makedirs(os.path.join(originalDir, subdir), exist_ok=True)


# === Evaluator Mode Functions =========================================================================================
def callEvaluator(argv: argparse.Namespace) -> None:
    fileList = createFileList(argv.input, argv.extension)
    instrList = []
    stopRead = False
    for file in tqdm(fileList):
        if(stopRead):
            print(f"Reached limit of {argv.limit} unique commands")
            break
        with open(file, "r") as disasm:
            for line in disasm:
                if(line[:-1] not in instrList):
                    instrList.append(line[:-1])
                    if(len(instrList) >= argv.limit) and (argv.limit != 0):
                        stopRead = True
                        break
    with open("DictSize.log", "+a") as logFile:
        if(stopRead):
            logFile.write('*')
        logFile.write(str(len(instrList)) + '\n')
    if(argv.summary is not ""):
        instrList.sort()
        with open("Summary.log", "w") as summaryFile:
            for instruction in instrList:
                summaryFile.write(instruction + '\n')
    if not(stopRead):
        print(f"Number of unique instructions: {len(instrList)}")


# === Generator Mode Functions =========================================================================================
def callDisassembler(argv: argparse.Namespace):
    fileList = createFileList(argv.input, argv.extension)
    createOutDirectory(argv.input, "disasm")
    print("[RUNNING DISASSEMBLER]")
    if argv.threads < 1:
        raise ValueError("The number of threads cannot be less than 1")
    elif argv.threads == 1:
        disasm: Disassembler = Disassembler(instructionLimit=argv.limit)
        disasm.processList(fileList, argv.input)
    else:
        fileSublists = generateSublists(fileList, argv.threads)
        procManager = ProcessManager(argv.threads)
        for i in range(0, argv.threads):
            disasm = Disassembler(instructionLimit=argv.limit)
            procManager.addProcess(disasm.processList, [fileSublists[i], argv.input])
        procManager.startBatch()
        procManager.awaitBatch()


# === Transformer Mode Functions =======================================================================================
def callTransformer(argv: argparse.Namespace):
    print("[RUNNING TRANSFORMER]")
    # Importing tensorflow slows down the disassembler, even though it's not needed for that operation
    from DisassemblerTransformer.DisasmTransformer import DisasmTransformer
    if argv.forcecpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    dt = DisasmTransformer(dataDir=argv.input,
                           modelPath=argv.output,
                           batchSize=argv.batchsize,
                           epochs=argv.epochs,
                           saveTraining=argv.summary)
    dt.prepareDatasets()
    dt.prepareModel()
    dt.trainModel()


# === main =============================================================================================================
if __name__ == "__main__":
    argv: argparse.Namespace = parseArgv()

    match(argv.mode):
        case "evaluator":
            callEvaluator(argv)
        case "generator":
            callDisassembler(argv)
        case "transformer":
            callTransformer(argv)
        case _: 
            raise ValueError("Bad mode of operation passed into program")
