##
# @package Disassembler
# Package containing definitions for the Disassembler class.
#
# To properly build a dataset from executable files, they must first be disassembled into their mnemonic form and their
# operands extracted as features.
#

# == Imports ===========================================================================================================
## @see https://www.capstone-engine.org/lang_python.html
from capstone import Cs, CS_ARCH_X86, CS_MODE_32, CS_MODE_64
import os
## @see https://github.com/erocarrera/pefile/blob/wiki/UsageExamples.md#introduction
from pefile import PE, PEFormatError
from sys import argv
from tqdm import tqdm
from typing import List



# == Disassembler Class ================================================================================================
##
# Function definitions and member variable declarations for the Disassembler class.
#
# Converting executable code back into its assembly representation requires the implementation of a disassembler. The
# Disassembler class not only provides an interface to pull readable instructions from machine code, but it also allows
# for some very basic manipulation of the instructions.
#
class Disassembler(object):
    ##
    # Class initializer called when a new Disassembler is instantiated.
    #
    # @param self Pointer to the new class instance.
    # @param inputFile The target executable file to be associated with the class instance.
    #
    def __init__(self, inputFile: str = "", instructionLimit: int = 10_000) -> None:
        ##
        # @var _exeName
        # File name of target executable for Disassembler instance.
        #
        self._exeName: str = inputFile

        ##
        # @var _executable
        # PE abstracted from the binary file using the `pefile` package.
        #
        self._executable: PE = None # type: ignore

        ##
        # @var _instructionLimit
        # Maximum number of instruction to extract from the provided executable.
        #
        self._instructionLimit: int = instructionLimit

        ##
        # @var _disassembler
        # Capstone disassembling object for converting machine code to assembly.
        #
        self._disassembler: Cs = Cs(CS_ARCH_X86, CS_MODE_32)

        ##
        # @var _textSecStart
        # Virtual address for the starting position of the .text section.
        #
        self._textSecStart: int = 0

        ##
        # @var _textSecEnd
        # Virtual address for the ending position of the .text section.
        #
        self._textSecEnd: int = 0

        ##
        # @var _disasmData
        # Sequence of disassembled machine code stored in a linear list.
        #
        self._disasmData: List[str] = list()

        # Code to setup class
        self._disassembler.skipdata = True # <-- Needed to keep going, even at `nop` instructions

        # Prepare class if an input file was provided
        if self._exeName != "":
            self._executable = PE(self._exeName, fast_load=True)
            self.disassemble()


    ##
    # Locates the .text section of an executable file and calculates its virtual starting and ending positions.
    #
    # @param self Pointer to calling class instance.
    #
    # In Windows PE files the .text section is typically used to store executable code. The virtual address and size of
    # each section within a file must be known by Windows beforehand, and, therefore, those values are stored within the
    # file itself. It is necessary to obtain these values to known where code that can be meaningfully disassembled
    # starts and ends within the binary stream of data.
    #
    def getTextSection(self) -> None:
        for section in self._executable.sections:
            # Linearly search for the 8-byte name of the .text section
            if(section.Name == b".text\0\0\0"):
                self._textSecStart = section.VirtualAddress
                self._textSecEnd = section.VirtualAddress + section.Misc_VirtualSize


    ##
    # Converts executable machine code to human-readable assembly.
    #
    # @param self Pointer to calling class instance.
    # @param includeAddress Flag used to determine if the virtual address of the instruction should be included with the
    #        mnemonic and operands.
    #
    # @pre Requires that the virtual address and size of the .text section of the executable has been calculated using
    #      getTextSection
    # @post Disassembled code is stored in the disasmData list as a set of strings.
    #
    # Converting machine code to assembly requires the usage of a disassembler. Instructions must be read in and the
    # appropriate number of operand bytes obtained before the conversion of binary data to text can occur. The Capstone
    # engine is implemented to facilitate this process and making handling different architectures and word sizes easier
    # to do.
    #
    def disassemble(self) -> None:
        self.getTextSection()
        # Pull the executable code from the PE file
        exeCode = self._executable.get_memory_mapped_image()[self._textSecStart:self._textSecEnd]
        # Get the entry point virtual address to ensure the correct address offset appears
        epVirtualAddress = self._executable.OPTIONAL_HEADER.ImageBase + self._textSecStart
        # Iterate over machine code and convert to assembly
        limit = self._instructionLimit
        for instruction in self._disassembler.disasm_lite(exeCode, epVirtualAddress):
            instruction = ' '.join(instruction[2:])
            limit -= 1
            self._disasmData.append(instruction)
            if limit <= 0:
                break


    ##
    # Dumps assembly code into a text file.
    #
    # @param self Pointer to calling class instance.
    # @param outputFile Destination file to store the text data for future use.
    #
    # Assembly code can be stored on the disk as a text file for future access or usage by another program. Each
    # instruction and its operands are given a single line in the output file.
    #
    def dumpAssembly(self, outputFile: str = "out.asm", delimiter: str = '\n') -> None:
        with open(outputFile, "w+") as disasmWrite:
            for instruction in self._disasmData:
                disasmWrite.write(f"{instruction}{delimiter}")



    ##
    #
    #
    def changeTarget(self, inputFile: str) -> None:
        self._exeName = inputFile
        # Some ".exe" files may not actually be in the PE format
        try:
            self._disasmData[:] = []
            self._executable = PE(self._exeName, fast_load=True)
            self.disassemble()
        except PEFormatError:
            print(f"Couldn't parse {self._exeName} due to formatting")
        except Exception as e:
            print(f"Couldn't parse {self._exeName} for {e}")

    ##
    #
    #
    def processList(self, inputList: List[str], outputDir: str) -> None:
        for element in tqdm(inputList):
            self.changeTarget(element)
            parsedName = os.path.split(element)
            higherDir = '/'.join(parsedName[:-1])
            newName = os.path.join(higherDir, "disasm", (parsedName[-1] + ".disasm"))
            self.dumpAssembly(newName, '\n')


    ##
    # Prints the information presented by `pefile.PE.dump_info()` to stdout.
    #
    # @param self Pointer to calling class instance.
    #
    def printReport(self) -> None:
        print(self._executable.dump_info())



# == Main ==============================================================================================================
if __name__ == "__main__":
    disasm = Disassembler(argv[1])
    disasm.dumpAssembly("data/out.disasm", ' ')
