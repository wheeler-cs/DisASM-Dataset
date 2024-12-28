# Disassembly Machine Learning Pipeline
This is a complete, all-in-one pipeline for generating a dataset of disassembled code, training a transformer on that
dataset, and utilizing the pre-trained model for the task of classification. This pipeline is primarily geared towards
the detection of malicious code and classifying Windows Portable Executable files(PE) as "benign" or "malicious."

| Table of Contents |
|-------------------|
| [Preparation](#preparation) |
| [Installation](#installation) |
| [Usage](#usage) |



## <a id=preparation>Preparation</a>
Before downloading the repository, you may need to install some install some packages to ensure the repo can be properly
setup and ran.

For Debian-like Linux distros, the packages `git`, `python3`, and `python3-pip` will need to be acquired from your
package manager. Other Linux distros will likely have the same packages under a similar name, but it is best to check
with your upstream repo to find out the exact name.

Windows users should install [git](https://git-scm.com) and [Python 3](https://python.org) from their respective
websites before proceeding.



## <a id=installation>Installation</a>
The repo comes with a **requirements.txt** file that can be used to install all of the dependencies needed by the
scripts. The steps for cloning the repo and setting up the Python virtual environment on Linux are as follows:

```
git clone https://github.com/wheeler-cs/DisASM-Dataset
cd DisASM-Dataset
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Windows users should be able to follow the same set of steps, which the exception of replacing
`source venv/bin/activate` with `./venv/bin/Activate.ps1` when activating the virtual environment.


## <a id=Usage>Usage</a>
Most of the functionalities of the various components can be accessed through the **runner.py** file, which can take a
number of arguments to dynamically change how the script behaves. There are four "modes" the scripts can run in, with
each fulfilling a specific function of the pipeline. A complete list of arguments the program can take can be obtained
using the command `python3 runner.py -h`.

### Generator
The generator mode is responsible for generating the dataset of assembly files from a provided dataset of executable
files.

### Transformer

### Classifier

### Evaluator
