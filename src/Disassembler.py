'''
@package Disassembler

'''

from capstone import Cs, CS_ARCH_X86, CS_MODE_32, CS_MODE_64
from pefile import PE
from sys import argv



# == Disassembler Class ================================================================================================
# See https://github.com/erocarrera/pefile/blob/wiki/UsageExamples.md#introduction
# See https://www.capstone-engine.org/lang_python.html
class Disassembler(object):
    def __init__(self, inputFile: str) -> None:
        self.exeName      = inputFile
        self.executable   = PE(argv[1], fast_load=True)
        self.disassembler = Cs(CS_ARCH_X86, CS_MODE_32)
        self.textSecStart = 0
        self.textSecEnd   = 0
        self.disasmData   = list()

        self.getTextSection()



    def getTextSection(self) -> None:
        for section in self.executable.sections:
            if(section.Name == b".text\0\0\0"):
                self.textSecStart = section.VirtualAddress
                self.textSecEnd = section.VirtualAddress + section.Misc_VirtualSize



    def disassemble(self, includeAddress: bool = False) -> None:
        exeCode = self.executable.get_memory_mapped_image()[self.textSecStart:self.textSecEnd]
        epVirtualAddress = self.executable.OPTIONAL_HEADER.ImageBase + self.textSecStart
        for instruction in self.disassembler.disasm(exeCode, epVirtualAddress):
            if includeAddress:
                self.disasmData.append(f"{hex(instruction.address)}: {instruction.mnemonic} {instruction.op_str.replace(' ', '')}")
            else:
                self.disasmData.append(f"{instruction.mnemonic} {instruction.op_str.replace(' ', '')}")
    


    def dumpAssembly(self, outputFile: str = "out.asm") -> None:
        with open(outputFile, "w+") as disasmWrite:
            for instruction in self.disasmData:
                disasmWrite.write(f"{instruction}\n")



    def printReport(self) -> None:
        print(self.executable.dump_info())



# == Main ==============================================================================================================
if __name__ == "__main__":
    disasm = Disassembler(argv[1])
    disasm.disassemble()
    disasm.dumpAssembly("data/out.disasm")
