# Sentences, Entities, and Keyphrases Extraction from Consumer Health Forums using Multi-task Learning

This repository contains dataset and code for three tasks related to question processing in the domain of Indonesian consumer health forums: sentence recognition, medical entity recognition, and keyphrase extraction.

## Tasks
1. **Sentence Recognition**. Given an input in the form of a question from consumer health forums, the expected output is the sentences identified in that question along with their respective types. We define three types of sentences in this work: `Background`, `Question`, and `Ignore`.
2. **Medical Entity Recognition**. Given an input in the form of a questions from consumer health forums, the expected output is a list of medical entities present in that question. We define four types of medical entities in this work: `Disease`, `Symptom`, `Drug`, and `Treatment`.
3. **Keyphrases Extraction**. Given an input in the form of a questions from consumer health forums, the expected output is a list of keyphrases representing the essence of that question.

## Dataset
The dataset for all three tasks can be accessed at `dataset/` folder, consisting of 1173 data points which splitted into 773 training instances, 200 validation instances, and 200 testing instances.

## Running the Code
For single-task learning, run the following command. Replace <task_id> with  `sentence_recognition`, `medical_entity_recognition`, or `keyphrase_extraction` to switch between tasks. Note that you also can change the encoder by setting the  `--pretrained_model` parameter.
```console
$ python single-task/main.py --task <task_id> --pretrained_model indobenchmark/indobert-large-p2
```

For multi-task learning, run the following command. Replace <task_ids> with two or more of the following options separated by comma (e.g. `mer,ke`): `sr` for sentence recognition, `mer` for medical entity recognition, `ke` for keyphrase extraction. The value of <mtl_structure> can be either `parallel` or `hierarchical`.
```console
$ python multi-task/main.py --tasks <task_ids> --structure <mtl_structure> --pretrained_model indobenchmark/indobert-large-p2
```
