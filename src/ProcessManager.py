##
#
#

# == Imports ===========================================================================================================
from multiprocessing import Process
from sys import argv
from time import sleep



# == Exceptions ========================================================================================================
class ProcessEnqueueException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)



# == Process Manager Class =============================================================================================
class ProcessManager(object):
    def __init__(self, processCeiling: int = 1) -> None:
        self._processCeiling: int = processCeiling
        self._processList = list()


    def addProcess(self, function, args = ()) -> None:
        if(len(self._processList) >= self._processCeiling):
            raise(ProcessEnqueueException(message="Cannot instantiate new process: Manager queue is full."))
        else:
            try:
                self._processList.append(Process(target=function, args=args))
            except Exception as e:
                raise(ProcessEnqueueException(message="Cannot instantiate new process: Process spawning failed."))
            

    def startBatch(self) -> None:
        for p in self._processList:
            p.start()

    
    def awaitBatch(self) -> None:
        for p in self._processList:
            p.join()
            p.close()


def testPrint() -> None:
    sleep(10)

        



# == Main ==============================================================================================================
if __name__ == "__main__":
    pm = ProcessManager(int(argv[1]))
    for i in range(0, int(argv[1])):
        pm.addProcess(testPrint, ())
    pm.startBatch()
    pm.awaitBatch()
