dataset: https://github.com/ashkankzme/biased_textrank
biased_textrank_git directory is git clone code from https://github.com/ashkankzme/biased_textrank
dev directory is used to save articles and questions from biased_textrank_git/data/liar/clean_test.json
  - articles are retrieve from "statements" in clean_test.json
  - questions are retrieve from "claim" in clean_test.json

answer directory is used to save answer from biased_textrank_git/data/liar/clean_test.json
  - answers are retrieve from "new_justification" in clean_test.json


output directory is used to save doctalk's predictions and stats

at /root, run command:
. venv/bin/activate

then run the commands as below:
python evaluate.py

It will compare doctalk's predictions with the answers under answer directory
then the final stats and score will be saved into output, file name is score_textrank.txt
