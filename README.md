# BertEval
Compare BERT-based models for document-level sentiment analysis using the [SemEval 2017](https://alt.qcri.org/semeval2017/task4/?id=download-the-full-training-data-for-semeval-2017-task-4) Twitter dataset.


## Installation:
Run the following command to install other dependencies

```bash
pip install -r requirements.txt
```

Download the SemEval twitter data from [here](https://alt.qcri.org/semeval2017/task4/?id=download-the-full-training-data-for-semeval-2017-task-4) and place the data from the Subtask A from the `GOLD` folder into the `data/tweets` folder.

## Running
The main program can be run by the following command:

```bash
python main.py train
```

Compare different preprocessing models using the `--preprocess_model` flag. Choose between `tfidf`, `bert-base-uncased`, or other models from [Huggingface](https://huggingface.co/). The program compares different 'head' neural models using the generated embeddings.

More options can be seen by using the `-h` tag.