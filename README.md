# Sentence Splitter

Script to split documents into sentences.

## Setup

```bash
conda env create
conda activate sentence-splitter
spacy download en_core_web_sm
```

## Usage

Pass an input file with one document per line:

```bash
./sentence_splitter.py INPUT_FILE > OUTPUT_FILE
```

The output will be one sentence per line, and documents will be separated by an empty line. You can alternatively pass 
the input file in stdin.

To see more information about the script and its available options, run:

```bash
./sentence_splitter.py --help
```
