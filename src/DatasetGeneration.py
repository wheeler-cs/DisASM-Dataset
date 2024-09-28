##
#
#

# == Import ============================================================================================================
from re import split
from Disassembler import Disassembler
from ProcessManager import ProcessManager, ProcessEnqueueException

import argparse
import os
from typing import List


# == Functions =========================================================================================================
def createArgParser():
    programArgs = argparse.ArgumentParser(prog="Dataset Generation",
                                          description="Generates a dataset using input executable files.")
    programArgs.add_argument("-p", "--pCeiling",
                             default=1,
                             help="Maximum number of processes script can spawn.",
                             required=False,
                             type=int)
    programArgs.add_argument("-e", "--extension",
                             default="",
                             help="Exclude all file extensions except for this one.",
                             required=False,
                             type=str)
    programArgs.add_argument("-i", "--inputDir",
                             help="Directory containing input executable files.",
                             required=True,
                             type=str)
    programArgs.add_argument("-o", "--outputDir",
                             help="Directory output dataset should be stored in.",
                             required=True,
                             type=str)
    parsedArgs = programArgs.parse_args()
    return parsedArgs


def createFileList(directory: str, fileExtension: str = "") -> List[str]:
    fileList = list()
    for file in os.listdir(directory):
        if(fileExtension != ""):
            if(os.path.isfile(directory + '/' + file) and (file[len(fileExtension) * -1:] == fileExtension)):
                fileList.append(directory + '/' + file)
        else:
            if(os.path.isfile(directory + '/' + file)):
                fileList.append(directory + '/' + file)
    return fileList


# == Main ==============================================================================================================
if __name__ == "__main__":
    # Setup program
    programArgs = createArgParser()
    fileList = createFileList(programArgs.inputDir, programArgs.extension)
    processesCeiling = programArgs.pCeiling
    procMan = ProcessManager(processesCeiling)

    # Create a list of Disassembler objects
    disasmList: List[Disassembler] = list()
    for i in range(0, processesCeiling):
        disasmList.append(Disassembler())

    # Split list of files a number of sublists equal to the process cap
    splitList = []
    for i in range(0, processesCeiling):
        splitList.append([])
    for i in range(0, len(fileList)):
        splitList[i % processesCeiling].append(fileList[i])

    # Parallelize list processing
    for i in range(0, processesCeiling):
        procMan.addProcess(disasmList[i].processList(splitList[i], ""))
    procMan.startBatch()
    procMan.awaitBatch()
