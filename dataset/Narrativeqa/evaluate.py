import collections
import json
import os
import re
import string
import sys
import csv
import rouge_stats as rs
from tylib.pycocoevalcap.bleu.bleu import Bleu
from tylib.pycocoevalcap.rouge.rouge import Rouge
from tylib.pycocoevalcap.cider.cider import Cider
from tylib.pycocoevalcap.meteor.meteor import Meteor

def ropen(f) :
  return open(f,'r',encoding='utf8')

  
def jload(infile) :
  ''' loads .json file, preprocessed from a .txt file'''
  with ropen(infile) as f:
    res = json.load(f)
    return res
    

def normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace."""
  def remove_articles(text):
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    return re.sub(regex, ' ', text)
  def white_space_fix(text):
    return ' '.join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()
  return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
  if not s: return []
  return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
  return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
  gold_toks = get_tokens(a_gold)
  pred_toks = get_tokens(a_pred)
  common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
  num_same = sum(common.values())
  if len(gold_toks) == 0 or len(pred_toks) == 0:
    # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
    return int(gold_toks == pred_toks)
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(pred_toks)
  recall = 1.0 * num_same / len(gold_toks)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def get_rougeL_score( pred, gold):
  k=0
  for i, res in enumerate(rs.rstat(pred,gold)) :
    if i == 2:
      d=res[0]      
      fx=d['f'][0] 
      return fx     

def batch_meteor_score(ref, hyp):
	score, _ = Meteor().compute_score(ref, hyp)
	return score

def batch_bleu_score(ref, hyp, n=4):
	score, _ = Bleu(n=n).compute_score(ref, hyp)
	return score

def batch_rouge_score(ref, hyp):
	score, _ = Rouge().compute_score(ref, hyp)
	return score

def bleu_score4(prediction, ground_truth, progress=True):
	prediction = {'dummy':[prediction]}
	ground_truth = {'dummy':[ground_truth]}
	score, _ = Bleu(n=4).compute_score(prediction, ground_truth, progress=False)
	return score

def bleu_score1(prediction, ground_truth):
	prediction = {'dummy':[prediction]}
	ground_truth = {'dummy':[ground_truth]}
	score, _ = Bleu(n=1).compute_score(prediction, ground_truth)
	return score

def rouge_score(prediction, ground_truth):
	return Rouge().calc_score([prediction], [ground_truth])



def get_scores(alg):
  exact_scores = {}
  f1_scores = {}
  rougel_scores = {}
  i = 0
  with open('narrativeqa_github/qaps.csv', newline='') as csvfile:
    questionset = csv.DictReader(csvfile)
    for row in questionset:
      if row['set'] != 'test': continue
      document_id = row['document_id']
      question = row['question']
      gold_answer1 =  normalize_answer(row['answer1'])
      gold_answer2 =  normalize_answer(row['answer2'])
      #print('gold: [document_id]:', document_id, '[question]:', question, '[answer1]:', gold_answer1, ' [answer2]:', gold_answer2 )
      preds = jload("output/" + alg + '/' + document_id + ".txt")
      pred_answer = normalize_answer(preds[question])
      #for F1
      em_1 = compute_exact(gold_answer1, pred_answer)
      f1_1 = compute_f1(gold_answer1, pred_answer)
      em_2 = compute_exact(gold_answer2, pred_answer)
      f1_2 = compute_f1(gold_answer2, pred_answer)

      if f1_1 >= f1_2:
        exact_scores[document_id + '_' + question] = em_1
        f1_scores[document_id + '_' + question] = f1_1
      else:
        exact_scores[document_id + '_' + question] = em_2
        f1_scores[document_id + '_' + question] = f1_2

      #for rouge_L
      fm_1 = get_rougeL_score(pred_answer, gold_answer1)
      fm_2 = get_rougeL_score(pred_answer, gold_answer2)
      rougel_scores[document_id + '_' + question] = max(fm_1, fm_2)

      # thirt party Rouge_l
      r1 = batch_rouge_score(gold_answer1, pred_answer)
      print('r1:\n', r1)
      meteor = batch_meteor_score(gold_answer1, pred_answer)
      print('meteor:\n', meteor)     
      bleu_1 = bleu_score1(pred_answer, gold_answer1)
      print('bleu_1:\n', bleu_1)
      bleu_4 = bleu_score4(pred_answer, gold_answer1)
      print('bleu_4:\n', bleu_4)
      i += 1
      if i == 1: break

  score =dict()
  total = len(exact_scores)
  score['total'] = total
  score['f1'] = 100.0 * sum(f1_scores.values()) / total
  score['exact'] = 100.0 * sum(exact_scores.values()) / total
  score['rouge_l'] = 100.0 * sum(rougel_scores.values()) / total
  return score


def main():
  for alg in  ['talker', 'ripple', 'thinker', 'bert' ]:
    eval = get_scores(alg)
    content = alg + ': \n'
    content += ' F1 = ' + str(eval['f1']) + ', exact_match = ' + str(eval['exact']) + '\n'
    content += ' rouge_l=' + str(eval['rouge_l']) + '\n'  
  totalQ = eval['total']
  totalSentsList = jload( 'output/Total_Sents.json')
  avgSents = round(sum(totalSentsList)/len(totalSentsList), 2)
  totalWordsList = jload('output/Total_Words.json')
  avgWords = round(sum(totalWordsList)/len(totalWordsList), 2)
  nlpParseDurList = jload( 'output/nlpParse_duration.json')
  avgNlpParsrDur = round(sum(nlpParseDurList)/totalQ, 5)
  doctalkSummDurList = jload( 'output/DoctalkSumm_duration.json')
  avgDoctalkSummDur = round(sum(doctalkSummDurList)/totalQ, 5)  
  talker_QA_self_list = jload( 'output/QA_talk_self_duration.json')
  avgTlkQaSelf = round(sum(talker_QA_self_list)/totalQ, 5)
  talker_QA_bert_list = jload( 'output/QA_talk_bert_duration.json')
  avgTlkQaBert = round(sum(talker_QA_bert_list)/totalQ, 5) 

  ripple_QA_self_list = jload( 'output/QA_ripple_self_duration.json')
  avgRiQaSelf = round(sum(ripple_QA_self_list)/totalQ, 5)
  ripple_QA_bert_list = jload( 'output/QA_ripple_bert_duration.json')
  avgRiQaBert = round(sum(ripple_QA_bert_list)/totalQ, 5) 

  thinker_QA_self_list = jload( 'output/QA_thinker_self_duration.json')
  avgThQaSelf = round(sum(thinker_QA_self_list)/totalQ, 5)
  thinker_QA_bert_list = jload( 'output/QA_thinker_bert_duration.json')
  avgThQaBert = round(sum(thinker_QA_bert_list)/totalQ, 5)

  bert_QA_bert_list = jload( 'output/QA_bert_bert_duration.json')
  avgBertQaBert = round(sum(bert_QA_bert_list)/totalQ, 5)

  stats = 'average Sentences: ' + str(avgSents) + '\n'
  stats += 'average words: ' + str(avgWords) + '\n'
  stats += 'Total questions: ' + str(totalQ) + '\n'
  stats += 'average nlpParse duration per question (seconds): ' + str(avgNlpParsrDur) + '\n'
  stats += 'average Doctak summarization duration per question (seconds): ' + str(avgDoctalkSummDur) + '\n' 
  stats += 'average talker self duration per question (seconds): ' + str(avgTlkQaSelf) + '\n' 
  stats += 'average talker bert duration per question (seconds): ' + str(avgTlkQaBert) + '\n' 
  stats += 'average ripple self duration per question (seconds): ' + str(avgRiQaSelf) + '\n' 
  stats += 'average ripple bert duration per question (seconds): ' + str(avgRiQaBert) + '\n' 
  stats += 'average thinker self duration per question (seconds): ' + str(avgThQaSelf) + '\n' 
  stats += 'average thinker bert duration per question (seconds): ' + str(avgThQaBert) + '\n' 
  stats += 'average Bert bert duration per question (seconds): ' + str(avgBertQaBert) + '\n' 


  print(stats )
  print("score:\n", content)

  toFile = "output/score_NewsQA.txt"
  print('save score to file:', toFile)
  
  with open(toFile,'w',encoding='utf8') as fscore:
    fscore.write(stats + "\n")
    fscore.write(content + "\n")

if __name__ == '__main__':
  main()

