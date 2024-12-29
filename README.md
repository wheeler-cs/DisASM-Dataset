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
files. The capstone framework is used to disassemble the executable files. During the disassembly process various
assembly commands are also genericized to reduce the size of the dictionary used for training the transformer.

The arguments available to the generator are:

```
-i, --input         Specifies the directory containing the files to be disassembled.
                    Type: String, REQUIRED
-t, --threads       Indicates the number of threads to use during disassembly.
                    Type: Integer
                    Default: 1
-x, --extension     Whitelist for files targeted by the disassembler based on extension.
                    Type: String
                    Default: ".exe"
-l, --limit         The maximum number of instructions that will be disassembled for each file.
                    Type: Integer
                    Default: 10,000
```

All executable files of a given classification should be placed in a subfolder of the project's root. During generation
a directory with the name **disasm** will be created inside this directory to store the disassembled code.

Sample directory structure:

```
|-data/
  |-benign/
    |-*.exe
    |-disasm/
      |-*.exe.disasm
```

Using the above directory structure, the **data/benign** directory can be passed as a command line argument to the
script to indicate it is the directory containing the executable files. The **disasm** directory is created
automatically if it does not exist.

```
python3 runner.py --mode generator -i ./data/benign
```

As some executable files are slow to disassemble, large datasets can take a quite a while to process. To help with this
issue, the generator process is capable of using multiple threads to speed up the process. This number should not
exceed the number of logical threads available on your machine. Additionally, memory considerations should be made as
capstone can consume a large amount of memory while running.

```
python3 runner.py --mode generator -i ./data/benign -t 4
```

Some files may not necessarily have an appropriate extension to indicate they are an executable. Alternative file
extensions can be specified at runtime. Functionality has also been included to take any file as input, regardless of
what type of extension it has, through the use of the wildcard **"*"** (note the use of quotes to bypass the console 
passing in all file names).

```
python3 runner.py --mode generator -i ./data/benign -x .bin

python3 runner.py --mode generator -i ./data/benign -x "*"
```

To reduce the disk space consumed by the dataset produced by the generator, the number of instructions disassembled for
each file can be limited. This is also reduces the amount of time required for generation. Any files smaller than the
value specified should be processed in their entirety. If every program _should_ be completely disassembled, a value of
0 can passed as the value for this argument.

```
python3 runner.py --mode generator -i ./data/benign -l 100000

python3 runner.py --mode generator -i ./data/benign -l 0
```

### Transformer

### Classifier

### Evaluator
