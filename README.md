# Predicting Thai NER Tags Using CRFs Model

This repository provides an implementation of Named Entity Recognition (NER) using a **Conditional Random Fields (CRF)** model. NER is a task in natural language processing (NLP) where the goal is to identify and classify named entities (e.g., persons, organizations, locations) in text. This repo will guide you how to use CRF model to predict NER tags step by steps if you have any questions about this please create an issue.

### Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Data Processing](#data-processing)
  - [Load Data](#load-data)
  - [Read Data](#read-data)
  - [Create Label List](#create-a-label-list-of-tags)
  - [Replacing Unknow Tags](#Replacting-weird-tags-you-encountered)
  - [Merge Data](#Merge-train-data-and-evaluation-data-for-better-training-(Optional))
  - [Feature Extraction](#feature-extraction)
  - [Split Data](#Split-data-to-train-and-val-(Optional))
- [Modeling](#modeling)
  - [Config Model](#Config-model)
  - [Fit Model](#Fit-model-with-X,-y)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
---

## Overview

This project uses a CRF model to predict NER tags on text data. The CRF model is particularly suited for sequence labeling tasks such as NER because it models the dependencies between neighboring labels, making it effective in identifying entities in a sequence.

Key features:
- CRF-based approach for predicting NER tags.
- Trained on a standard NER dataset.
- Easy integration with any text-based input.

---

## Getting Started

To get started with this project, follow the instructions below.

### Prerequisites

- Python 3.6+
- pip (sklearn-crfsuite)
- Required libraries:
  - `sklearn`
  - `pandas`
  - `numpy`

### Installation

1. Clone the repository:

```bash
git clone https://github.com/ImtaeZ/Predicting-NER-tags-using-CRFs-Models
```
2. Install CRF
   
  ```bash
  !pip install sklearn-crfsuite
  ```
## Data processing
This notebool was made in Kaggle so you'll have to redirect the directoty to your own data paths.

#### 1. Load Datas
Since the data came in txt files we need to convert them to a dataframe first.
```bash
# Load Data
import os
import pandas as pd

# Define a funciton to read txt files as df
def txt_as_df(path):
    all_sentences = []

    for filename in sorted(os.listdir(path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(path, filename)

            sentences = []
            current_sentence = []

            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()

                    if not line:
                        if current_sentence:
                            sentences.append(current_sentence)
                            current_sentence = []
                    else:
                        word_data = line.split("\t")
                        if len(word_data) == 4:
                            current_sentence.append(word_data)

                if current_sentence:  # In case there's no empty line at the end
                    sentences.append(current_sentence)

            for sentence in sentences:
                df = pd.DataFrame(sentence, columns=["word", "pos", "ner", "cls"])
                all_sentences.append(df)

    return all_sentences
```
#### 2. Read Data
```bash
# Read Data
train_df = txt_as_df("---/train/train")
eval_df = txt_as_df("---/eval/eval")
```

This is the example of data in dataframe (1st 10 rows)
![image](https://github.com/user-attachments/assets/bb21cc89-38fc-4811-8458-7c69d15066a4)


#### 3. Create a Label List of Tags
```bash
# Create Label List
labels = ['O',
 'B_ORG',
 'B_PER',
 'B_LOC',
 'B_MEA',
 'I_DTM',
 'I_ORG',
 'E_ORG',
 'I_PER',
 'B_TTL',
 'E_PER',
 'B_DES',
 'E_LOC',
 'B_DTM',
 'B_NUM',
 'I_MEA',
 'E_DTM',
 'E_MEA',
 'I_LOC',
 'I_DES',
 'E_DES',
 'I_NUM',
 'E_NUM',
 'B_TRM',
 'B_BRN',
 'I_TRM',
 'E_TRM',
 'I_TTL',
 'I_BRN',
 'E_BRN',
 'E_TTL',
 'B_NAME']
```

#### 4. Replacing weird tags you encountered
```bash
def replace_weird_tag(dataframes, tags):
    
    for df in dataframes:
        
        df["ner"] = df["ner"].apply(lambda x: "B_ORG" if x in tags else x)
    
    return dataframes

train_df = replace_weird_tag(train_df, {'OBRN_B', 'MEA_BI', 'B_D`TM', 'ORG_I', 'I', '__', 'DDEM', 'B', 'PER_I'})
eval_df = replace_weird_tag(eval_df, {'LOC_I', 'ABB', 'B', '__', 'ORG_I'})
```

#### 5. Merge train data and evaluation data for better training (Optional)
In my case I need to predict the test data so I merged train_df and eval_df together but you don't need to do that because you will be evaluating your model based on evaluation data anyways.
```bash
# Merge train_df and eval_df
merge_df = train_df + eval_df
```

#### 6. Extract features to X,y
Extract feature that needed to predict in the X variable and the features that you want to predict in y variable this is very common practice for ML variables initializing
```bash
def extract_features(sentence_df):
    features = []
    for i in range(len(sentence_df)):
        word = sentence_df.iloc[i]["word"]
        pos_tag = sentence_df.iloc[i]["pos"]
        clause_boundary = sentence_df.iloc[i]["cls"]

        # Define Features for each Token
        token_features = {
            "word": word,
            "pos_tag": pos_tag,
            "clause_boundary": clause_boundary,
            "is_first_word": i == 0,
            "is_last_word": i == len(sentence_df) - 1,
            "prefix-1": word[0],
            "prefix-2": word[:2],
            "suffix-1": word[-1],
            "suffix-2": word[-2:],
            "prev_word": '' if i == 0 else sentence_df.iloc[i - 1]["word"],
            "next_word": '' if i == len(sentence_df) - 1 else sentence_df.iloc[i + 1]["word"],
            "prev_pos": '' if i == 0 else sentence_df.iloc[i - 1]["pos"],
            "next_pos": '' if i == len(sentence_df) - 1 else sentence_df.iloc[i + 1]["pos"],
        }
        
        features.append(token_features)
    return features

def preprocess_data(dataframes, has_labels=True):
    X = []
    y = []

    for df in dataframes:
        
        sentence_features = extract_features(df)
        X.append(sentence_features)

        if has_labels and "ner" in df.columns:
            sentence_labels = df["ner"].tolist()
            y.append(sentence_labels)
        else:
            y.append([])

    return X, y
```
Extract features to X and y
```bash
X, y = preprocess_data(merge_df)
```

#### 7. Split data to train and val (Optional)
If you already merge your eval and train data you can split them again in any portions using a sklearn library train_test_split
```bash
# Split data using train_test_split to train
import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.8 ,test_size = 0.2, random_state=42)
```

## Modeling

#### 1. Config model
```bash
# Config model
model = sklearn_crfsuite.CRF(
    algorithm="lbfgs",
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
    verbose=True
)
```
You can increase max_iterations for better training result if you have enough RAM and GPU for it.

#### 2. Fit model with X, y
```bash
model.fit(X_train, y_train)
```

## Evaluation
Evaluating your model using F1 macro score
```bash
from sklearn_crfsuite import metrics
y_pred = model.predict(X_val)
f1_score = metrics.flat_f1_score(y_val, y_pred, average="macro", labels=labels, zero_division=0)
f1_score
```
## Improve your model
You can use all the techniques to improve the performance by checking is it overfitting or not, visualize it using matplotlib to see the best leaf nodes or using cross validations techniques etc. After that, you can check F1 score to check the improvement. (But in this repo will keep it basic so there will be no techniques to improve the model.)

## Contributing

We welcome contributions to this project! Whether you're reporting a bug, suggesting an enhancement, or submitting a pull request, your feedback is important.

### How to Contribute:

1. **Fork the Repository:**
   - Navigate to the [GitHub repository](https://github.com/ImtaeZ/Predicting-NER-tags-using-CRFs-Models).
   - Click the **Fork** button to create a copy of the repository on your own GitHub account.

2. **Clone Your Fork:**
   - On your GitHub account, clone the repository to your local machine:
   
     ```bash
     git clone https://github.com/your-username/Predicting-NER-tags-using-CRFs-Models.git
     ```

3. **Create a New Branch:**
   - Create a new branch for your changes. This keeps your changes isolated from the main branch.
   
     ```bash
     git checkout -b feature-name
     ```

4. **Make Changes:**
   - Make your changes or additions to the code. Ensure that the code is properly formatted and adheres to the project’s conventions.
   - If applicable, add tests for your changes.

5. **Commit Your Changes:**
   - After making the changes, commit your modifications:
   
     ```bash
     git commit -am 'Add a descriptive commit message'
     ```

6. **Push Your Changes:**
   - Push your branch to your forked repository on GitHub:
   
     ```bash
     git push origin feature-name
     ```

7. **Submit a Pull Request:**
   - Go to the original repository on GitHub, click on the **Pull Request** button, and submit your pull request.
   - Be sure to explain the changes you made, and why they should be merged.

### Code of Conduct

Please be respectful and considerate when contributing. We aim to build a welcoming community for everyone. 

### Issues and Bugs

If you find any bugs or have suggestions for improvement, please feel free to open an issue or contribute to resolving the issues. You can report bugs or submit feature requests via GitHub issues.

---

Thank you for your contributions!

## License

This dataset is private do **NOT** use it for any other purporse than educating.
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### MIT License
