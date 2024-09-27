##
#
#

# == Import ============================================================================================================
from Disassembler import Disassembler
from ProcessManager import ProcessManager, ProcessEnqueueException

import argparse
import os
from typing import List


# == Functions =========================================================================================================
def createArgParser():
    programArgs = argparse.ArgumentParser(prog="Dataset Generation", description="Generates a dataset using input executable files.")
    programArgs.add_argument("-p", "--pCeiling", default=1, help="Maximum number of processes script can spawn.", required=False, type=int)
    programArgs.add_argument("-e", "--extension", default="", help="Exclude all file extensions except for this one.", required=False, type=str)
    programArgs.add_argument("-i", "--inputDir", help="Directory containing input executable files.", required=True, type=str)
    programArgs.add_argument("-o", "--outputDir", help="Directory output dataset should be stored in.", required=True, type=str)
    parsedArgs = programArgs.parse_args()
    return parsedArgs


def createFileList(directory: str, fileExtension: str = "") -> List[str]:
    fileList = list()
    for file in os.listdir(directory):
        if(fileExtension != ""):
            if(os.path.isfile(directory + '/' + file) and (file[len(fileExtension) * -1:] == fileExtension)):
                fileList.append(file)
        else:
            if(os.path.isfile(directory + '/' + file)):
                fileList.append(file)
    return fileList


# == Main ==============================================================================================================
if __name__ == "__main__":
    programArgs = createArgParser()
    fileList = createFileList(programArgs.inputDir, programArgs.extension)
    procMan = ProcessManager(programArgs.pCeiling)
