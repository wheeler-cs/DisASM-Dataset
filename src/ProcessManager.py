##
# @package ProcessManager
# Package containing definitions for ProcessManager class.
#

# == Imports ===========================================================================================================
from threading import Thread
from typing import List



# == Exceptions ========================================================================================================
##
# Exception intended to be used in the event adding a new process to ProcessManager's queue fails.
#
class ProcessEnqueueException(Exception):
    ##
    # Exception class initializer called when an exception of this type is raised.
    #
    # @param message Informational message to be associated with exception.
    #
    def __init__(self, message: str) -> None:
        super().__init__(message)



# == Process Manager Class =============================================================================================
class ProcessManager(object):
    def __init__(self, processCeiling: int = 1) -> None:
        self._processCeiling: int = processCeiling
        self._processList: List[Thread] = list()


    def addProcess(self, function, args = ()) -> None:
        if(len(self._processList) >= self._processCeiling):
            raise(ProcessEnqueueException(message="Cannot instantiate new process: Manager queue is full."))
        else:
            try:
                self._processList.append(Thread(target=function, args=args))
            except Exception as e:
                raise(ProcessEnqueueException(message="Cannot instantiate new process: Process spawning failed."))
            

    def startBatch(self) -> None:
        for p in self._processList:
            p.start()

    
    def awaitBatch(self) -> None:
        for p in self._processList:
            p.join()



# == Main ==============================================================================================================
if __name__ == "__main__":
    print("The ProcessManager class is intended to be used as part of a module.")
