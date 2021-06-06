""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys

  
def jload(infile) :
  ''' loads .json file, preprocessed from a .txt file'''
  with open(infile,'r',encoding='utf8') as f:
    res = json.load(f)
    return res

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    print('predictions length:', len(predictions))
    f1 = exact_match = total = 0
    for aindex, article in enumerate(dataset):
        for pindex, paragraph in enumerate(article['paragraphs']):
            for qindex, qa in enumerate(paragraph['qas']):
                total += 1
                if qa['id'] not in predictions:
                    message = article['title'] + ', Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)
                #if qindex == 1: break
            #if pindex ==0: break
        #if aindex == 3: break
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

##########################################################################
#Can be used for paragraph or whole article
#python evaluate.py paragraph
#python evaluate.py article
##########################################################################

if __name__ == '__main__':
  print('Number of arguments:', len(sys.argv), 'arguments.')
  print('Argument List:', str(sys.argv))
  if len(sys.argv) != 2 or  sys.argv[1] not in ['paragraph', 'article' ]:
    print('Run one of the commands as below:')
    print(' python evaluate.py paragraph')
    print(' python evaluate.py article ')
    sys.exit(0)
  type = sys.argv[1]
  dataset= jload( "dev-v1.1.json")
  outputDir = type + '/output/'

  predictions = jload( outputDir + 'predictions_talker.json')
  score = evaluate(dataset['data'], predictions)
  em_talker = round(score['exact_match'], 2)
  f1_talker = round(score['f1'], 2)
  content = 'talker_F1 = ' +  str(f1_talker) + ', talker_exact_match = ' + str(em_talker) + '\n'

  predictions = jload( outputDir + 'predictions_ripple.json')
  score = evaluate(dataset['data'], predictions)
  em_ripple = round(score['exact_match'], 2)
  f1_ripple = round(score['f1'], 2)
  content += 'ripple_F1 = ' +  str(f1_ripple) + ', ripple_exact_match = ' + str(em_ripple) + '\n'

  predictions = jload( outputDir + 'predictions_thinker.json')
  score = evaluate(dataset['data'], predictions)
  em_thinker = round(score['exact_match'], 2)
  f1_thinker = round(score['f1'], 2)
  content += 'thinker_F1 = ' +  str(f1_thinker) + ', thinker_exact_match = ' + str(em_thinker) + '\n'

  predictions = jload( outputDir + 'predictions_bert.json')
  score = evaluate(dataset['data'], predictions)
  em_bert = round(score['exact_match'], 2)
  f1_bert = round(score['f1'], 2)
  content += 'bert_F1 = ' +  str(f1_bert) + ', bert_exact_match = ' + str(em_bert) + '\n'

  totalSentsList = jload( outputDir + 'Total_Sents.json')
  avgSents = round(sum(totalSentsList)/len(totalSentsList), 2)
  totalWordsList = jload( outputDir + 'Total_Words.json')
  avgWords = round(sum(totalWordsList)/len(totalWordsList), 2)
  nlpParseDurList = jload( outputDir + 'nlpParse_duration.json')
  avgNlpParsrDur = round(sum(nlpParseDurList)/len(totalWordsList), 5)
  doctalkSummDurList = jload( outputDir + 'DoctalkSumm_duration.json')
  avgDoctalkSummDur = round(sum(doctalkSummDurList)/len(totalWordsList), 5) 

  talker_QA_self_list = jload( outputDir + 'QA_talk_self_duration.json')
  avgTlkQaSelf = round(sum(talker_QA_self_list)/len(predictions), 5)
  talker_QA_bert_list = jload( outputDir + 'QA_talk_bert_duration.json')
  avgTlkQaBert = round(sum(talker_QA_bert_list)/len(predictions), 5) 

  ripple_QA_self_list = jload( outputDir + 'QA_ripple_self_duration.json')
  avgRiQaSelf = round(sum(ripple_QA_self_list)/len(predictions), 5)
  ripple_QA_bert_list = jload( outputDir + 'QA_ripple_bert_duration.json')
  avgRiQaBert = round(sum(ripple_QA_bert_list)/len(predictions), 5) 

  thinker_QA_self_list = jload( outputDir + 'QA_thinker_self_duration.json')
  avgThQaSelf = round(sum(thinker_QA_self_list)/len(predictions), 5)
  thinker_QA_bert_list = jload( outputDir + 'QA_thinker_bert_duration.json')
  avgThQaBert = round(sum(thinker_QA_bert_list)/len(predictions), 5)

  bert_QA_bert_list = jload( outputDir + 'QA_bert_bert_duration.json')
  avgBertQaBert = round(sum(bert_QA_bert_list)/len(predictions), 5)
      

  stats = 'average Sentences: ' + str(avgSents) + '\n'
  stats += 'average words: ' + str(avgWords) + '\n'
  stats += 'Total articles: ' + str(len(totalWordsList)) + '\n'
  stats += 'average nlpParse duration per article (seconds): ' + str(avgNlpParsrDur) + '\n'
  stats += 'average Doctak summarization duration per article (seconds): ' + str(avgDoctalkSummDur) + '\n' 

  stats += 'Total questions: ' + str(len(predictions)) + '\n'
  stats += 'average talker self duration per question (seconds): ' + str(avgTlkQaSelf) + '\n' 
  stats += 'average talker bert duration per question (seconds): ' + str(avgTlkQaBert) + '\n' 
  stats += 'average ripple self duration per question (seconds): ' + str(avgRiQaSelf) + '\n' 
  stats += 'average ripple bert duration per question (seconds): ' + str(avgRiQaBert) + '\n' 
  stats += 'average thinker self duration per question (seconds): ' + str(avgThQaSelf) + '\n' 
  stats += 'average thinker bert duration per question (seconds): ' + str(avgThQaBert) + '\n' 
  stats += 'average Bert bert duration per question (seconds): ' + str(avgBertQaBert) + '\n' 

  print(stats )
  print("score:\n", content)

  toFile = outputDir + "SQuAD_1.1_score.txt"
  print('save score to file:', toFile)
  with open(toFile, 'w',encoding='utf8') as fscore:
    fscore.write(stats + "\n")
    fscore.write(content + "\n")
