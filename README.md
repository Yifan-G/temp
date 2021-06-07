Project Description
** DocTalk performs summarization, keyphrase extraction, question answer. 
Doctalk downloads stanza_corenlp and nltk_data automatically , please make sure that there is internet access.

The Directories
DocTalk/doctalk/ # Our graph-based keyphrase extraction and summarization, , question answer model
DocTalk/eval_qa.py is used to test dataset
 - save* functions are used to save contents, questions, answers from dataset json file or csv file
 - answer* functions are used to ask doctalk to get answers
 - reason_with_bert function fills in all content to bert to test its performance
DocTalk/dataset/ # Contains sample documents from various datasets we used in the paper.
 - readme.txt: includes of dataset website and how to do evaluation 
 - evaluate.py is used to do evaluation (F1, exact_match,  ROUGE, bleu, meteor) and print stats
 - includes of dataset: Narrativeqa, NewsQA, SQuAD, textrank, HotpotQA

## Setup enviroment in linux:
- python 3.6 or newer, pip3, java 9.x or newer, SWI-Prolog 8.x or newer, graphviz
- also, having git installed is recommended for easy updates
The steps are as below with root:
$ yum install epel-release
$ yum install python3-pip
$ python3 -m pip install --upgrade pip setuptools wheel
$ pip3 install virtualenv
$ virtualenv -p python3.6 venv
$ . venv/bin/activate

install java
    $ yum -y install java

install   dependent packages, in  DocTalk folder
  $ pip install -r requirements.txt

if doctalk is used:
    copy doctalk requirements.txt under doctalk folder
    cd to  doctalk folder
    $ pip install -r requirements.txt

 if  StanzaGraphs is used, go to StanzaGraphs folder, run
    $ pip install -r requirements.txt
 
check packages version be command
    $ pip freeze 
run  every access
    $ . venv/bin/activate


## How to do test, for example Narrativeqa
python -i eval_qa.py
>>>saveNarrativeqa_QuestionContent()
>>>answerNarrativeqa()
>>>exit()

then go to dataset/Narrativeqa directory, read readme.txt, run
python evaluate.py

then you can get scores and stats

 





