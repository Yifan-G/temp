import sys
import os
import json
import argparse
from doctalk.talk import Talker, nice_keys, exists_file, jload, jsave
from doctalk.params import talk_params, ropen, wopen
from doctalk.think import reason_with_File, reason_with_Text
from dataset.squad.one.evaluate import evaluate

qidAnswer_Talker = dict()
qidAnswer_Ripple = dict()
qidAnswer_Thinker = dict()
qidAnswer_Bert = dict()
totalSentsList = []
totalWordsList = []
nlpParseDurList = []
doctalkSummDurList = []  
talker_QA_self_list = []
talker_QA_bert_list = []
ripple_QA_self_list = []
ripple_QA_bert_list= []
thinker_QA_self_list = []
thinker_QA_bert_list = []
bert_QA_bert_list = []


def createSQuADQuestionIDMap(version):
  datadir = "dataset/squad/"  + version + "/"
  if version == "one":
    dataset= jload( datadir + "dev-v1.1.json")
  else : #version ="2.0"
    dataset= jload( datadir + "dev-v2.0.json")
  #data is []
  #print('data[0]:', dataset['data'][0])
  qidMap = dict()
  for article in dataset['data']:
      for i, paragraph in enumerate(article['paragraphs']):    
         questions = paragraph['qas']
         for question in questions:
             qid = question['id']
             q=question['question']
             qidMap[qid] = article['title']  + "_" + str(i) + "_" + q
  output = json.dumps(qidMap)
  fname = datadir + "qidMap.json"
  with wopen(fname) as f:
    f.write(output + "\n")
  f.close()

           
def saveSQuAD_QuestionContent(version):
  datadir = "dataset/squad/" + version + "/"
  outputDir = datadir + "/paragraph/"
  os.makedirs(outputDir,exist_ok=True)
  os.makedirs(outputDir + 'dev',exist_ok=True)
  os.makedirs(outputDir + 'output',exist_ok=True)  
  if version == "one":
    dataset= jload( datadir + "dev-v1.1.json")
  else : #version ="2.0"
    dataset= jload( datadir + "dev-v2.0.json")
  #data is []
  #print('data[0]:', dataset['data'][0])
  for article in dataset['data']: 
      for i, paragraph in enumerate(article['paragraphs']):
         fname = outputDir + "dev/" + article['title']  + "_" + str(i) + ".txt"
         context = paragraph['context']
         with wopen(fname) as fcontext :
            fcontext.write(context + "\n")
         fcontext.close()          
         questions = paragraph['qas']
         fqname = outputDir + "dev/" + article['title']  + "_" + str(i) + "_quest.txt" 
         with wopen(fqname) as fquest:
           for question in questions:
             q=question['question']
             fquest.write(q + "\n")
           fquest.close()

 

def answerSQuADFromFile(version):
  datadir = "dataset/squad/" + version + "/"
  if version == "one":
    dataset= jload( datadir + "dev-v1.1.json")
  else : #version ="2.0"
    dataset= jload( datadir + "dev-v2.0.json")
  #data is []
  outputDir = datadir + "/paragraph/output/"
  loadResult(outputDir)
  

  for count, article in enumerate(dataset['data']):
    for i, paragraph in enumerate(article['paragraphs']):
      #if i<3: continue
      fname = datadir +"paragraph/dev/" + article['title']  + "_" + str(i)
      Talker, Ripple, Thinker, Bert,totalSents, totalWords, nlpParseDur, doctalkSummDur, doctalkQaDur = reason_with_doctalk_FromFile(fname)
      '''
      print('\n\nanswerSQuAD, Talker:', Talker)
      print('answerSQuAD, Ripple:', Ripple)
      print('answerSQuAD, Thinker:', Thinker)
      print('answerSQuAD, Bert:', Bert)
      print('Total sentences:', totalSents)
      print('Total words:', totalWords)
      print("Stanza nlp parse duration(seconds): ", round(nlpParseDur, 5))
      print("doctalk summKeys duration(seconds): ", round(doctalkSummDur, 5))
      print("doctalk Q&A duration(seconds): ", doctalkQaDur)
      '''
      totalSentsList.append(totalSents)
      totalWordsList.append(totalWords)
      nlpParseDurList.append(nlpParseDur)
      doctalkSummDurList.append(doctalkSummDur)
      talker_QA_self_list.append(doctalkQaDur['talker']['self'])
      talker_QA_bert_list.append(doctalkQaDur['talker']['bert'])
      ripple_QA_self_list.append(doctalkQaDur['ripple']['self'])
      ripple_QA_bert_list.append(doctalkQaDur['ripple']['bert'])
      thinker_QA_self_list.append(doctalkQaDur['thinker']['self'])
      thinker_QA_bert_list.append(doctalkQaDur['thinker']['bert'])
      bert_QA_bert_list.append(doctalkQaDur['bert']['bert'])

      qids = []
      for qa in paragraph['qas']:
        qid = qa['id']
        qids.append(qid)
      for j, qid in enumerate(qids):
        qidAnswer_Talker[qid] = Talker[j]
        qidAnswer_Ripple[qid] = Ripple[j]
        qidAnswer_Thinker[qid] = Thinker[j]
        qidAnswer_Bert[qid] = Bert[j]
        #if j == 1: break
      '''
      print('qidAnswer_Talker:', qidAnswer_Talker)
      print('qidAnswer_Ripple:', qidAnswer_Ripple)
      print('qidAnswer_Thinker:', qidAnswer_Thinker)
      print('qidAnswer_Bert:', qidAnswer_Bert)
      print('totalSentsList:', totalSentsList)
      print('totalWordsList:', totalWordsList)
      print("nlpParseDurList: ", nlpParseDurList)
      print("doctalkSummDurList: ", doctalkSummDurList)
      print('talker_QA_self_list:', talker_QA_self_list)
      print('talker_QA_bert_list:', talker_QA_bert_list)
      print('ripple_QA_self_list:', ripple_QA_self_list)
      print('ripple_QA_bert_list:', ripple_QA_bert_list)
      print('thinker_QA_self_list:', thinker_QA_self_list)
      print('thinker_QA_bert_list:', thinker_QA_bert_list)
      print('bert_QA_bert_list:', bert_QA_bert_list)
      '''

      outputTalker = json.dumps(qidAnswer_Talker)
      with wopen(outputDir + 'predictions_talker.json' ) as ftalk:
        ftalk.write(outputTalker + "\n")

      outputRipple = json.dumps(qidAnswer_Ripple)
      with wopen(outputDir + 'predictions_ripple.json' ) as fRipple:
        fRipple.write(outputRipple + "\n")
      
      outputThinker = json.dumps(qidAnswer_Thinker)
      with wopen(outputDir + 'predictions_thinker.json' ) as fthink:
        fthink.write(outputThinker + "\n")

      outputBert = json.dumps(qidAnswer_Bert)
      with wopen(outputDir + 'predictions_bert.json' ) as fbert:
        fbert.write(outputBert + "\n")

      saveStats_WordsDuration(outputDir, totalSentsList, totalWordsList, 
        nlpParseDurList, doctalkSummDurList, 
        talker_QA_self_list, talker_QA_bert_list,
        ripple_QA_self_list, ripple_QA_bert_list,
        thinker_QA_self_list, thinker_QA_bert_list,
        bert_QA_bert_list )
      #if i == 5: break
    #if count ==0: break


def answerSQuADFromText(version):
  datadir = "dataset/squad/" + version + "/"
  if version == "one":
    dataset= jload( datadir + "dev-v1.1.json")
  else : #version ="2.0"
    dataset= jload( datadir + "dev-v2.0.json")
  #data is []
  qidAnswer_Talker =dict() 
  qidAnswer_Ripple =dict() 
  qidAnswer_Thinker =dict() 
  qidAnswer_Bert =dict() 
  for count, article in enumerate(dataset['data']):
    for i, paragraph in enumerate(article['paragraphs']):
      context = paragraph['context']
      print('\n\n^^^^^^^^^^^^^^^^^^^^^^^^^^\n')
      print(article['title'], ' paragraph[', i, ']')
      print('context:\n',context )
      print('\n^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n')

      qlist = []
      questions = paragraph['qas']
      for question in questions:
        q=question['question']
        qlist.append(q)
      Talker, Ripple, Thinker, Bert = reason_with_doctalk_FromText(context, qlist)
      print('answerSQuAD, Talker:', Talker )
      print('answerSQuAD, Ripple:', Ripple )
      print('answerSQuAD, Thinker:', Thinker )
      print('answerSQuAD, Bert:', Bert )
      qids = []
      for qa in paragraph['qas']:
        qid = qa['id']
        qids.append(qid)
      print('qids length:', len(qids), ', detail:', qids)
      for j, qid in enumerate(qids):
        qidAnswer_Talker[qid] = Talker[j]
        qidAnswer_Ripple[qid] = Ripple[j]
        qidAnswer_Thinker[qid] = Thinker[j]
        qidAnswer_Bert[qid] = Bert[j]
      if i == 0: break
    if count==0: break
  #print('\n qidAnswer_Talker:', qidAnswer_Talker)
  #print('\n qidAnswer_Ripple:', qidAnswer_Ripple)
  #print('\n qidAnswer_Thinker:', qidAnswer_Thinker)
  #print('\n qidAnswer_Bert:', qidAnswer_Bert)  
  
  outputTalker = json.dumps(qidAnswer_Talker)
  with wopen(datadir + 'predictions_talker.json' ) as ftalk:
    ftalk.write(outputTalk + "\n")

  outputRipple = json.dumps(qidAnswer_Ripple)
  with wopen(datadir + 'predictions_ripple.json' ) as fRipple:
    fRipple.write(outputRipple + "\n")
  
  outputThinker = json.dumps(qidAnswer_Thinker)
  with wopen(datadir + 'predictions_thinker.json' ) as fthink:
    fthink.write(outputThink + "\n")

  outputBert = json.dumps(qidAnswer_Bert)
  with wopen(datadir + 'predictions_bert.json' ) as fbert:
    fbert.write(outputBert + "\n")

#################################################################################
# above work for squad paragraph, it is same test in squad1.1 dev
# https://rajpurkar.github.io/SQuAD-explorer/explore/1.1/dev/
# evaluateSQuAD(type) can be used for wholeArticle
# below is for whole Article
#####################################################################################

def saveSQuAD_QuestionContent_wholeArticle(version):
  datadir = "dataset/squad/" + version + "/"
  outputdir = datadir + "/article/"
  os.makedirs(outputdir,exist_ok=True)
  os.makedirs(outputdir + 'dev',exist_ok=True)
  os.makedirs(outputdir + 'output',exist_ok=True)
  if version == "one":
    dataset= jload( datadir + "dev-v1.1.json")
  else : #version ="2.0"
    dataset= jload( datadir + "dev-v2.0.json")
  #data is []
  #print('data[0]:', dataset['data'][0])
  for article in dataset['data']:
      context = ''
      questions = '' 
      totalParagragh = len(article['paragraphs'])   
      for i, paragraph in enumerate(article['paragraphs']):
         context += paragraph['context'] + '\n'
         qas = paragraph['qas']
         for q in qas:
             questions += q['question'] + '\n'
         if i == (totalParagragh -1):
            fname = outputdir + "dev/" + article['title'] + ".txt"
            with wopen(fname) as fcontext :
              fcontext.write(context + "\n")
            fname = outputdir + "dev/" + article['title'] + "_quest.txt" 
            with wopen(fname) as f :
              f.write(questions + "\n")



def answerSQuADFromFile_wholeArticle():
  datadir = "dataset/squad/one/"
  outputDir = datadir + "article/output/"
  os.makedirs(outputDir,exist_ok=True)
  dataset= jload( datadir + "dev-v1.1.json")

  loadResult(outputDir)
  '''
  print('\n\nbefore start')
  print('qidAnswer_Talker:', qidAnswer_Talker)
  print('qidAnswer_Ripple:', qidAnswer_Ripple)
  print('qidAnswer_Thinker:', qidAnswer_Thinker)
  print('qidAnswer_Bert:', qidAnswer_Bert)
  print('totalSentsList:', totalSentsList)
  print('totalWordsList:', totalWordsList)
  print("nlpParseDurList: ", nlpParseDurList)
  print("doctalkSummDurList: ", doctalkSummDurList)
  print("talker_QA_self_list: ", talker_QA_self_list)
  print("talker_QA_bert_list: ", talker_QA_bert_list)
  print("ripple_QA_self_list: ", ripple_QA_self_list)
  print("ripple_QA_bert_list: ", ripple_QA_bert_list)
  print("thinker_QA_self_list: ", thinker_QA_self_list)
  print("thinker_QA_bert_list: ", thinker_QA_bert_list)
  print("bert_QA_bert_list: ", bert_QA_bert_list)
  '''
  #data is []
  for count, article in enumerate(dataset['data']):
    #if count < 2: continue
    qids = []
    fname = datadir + "article/" +"dev/" + article['title'] 
    Talker, Ripple, Thinker, Bert,totalSents, totalWords, nlpParseDur, doctalkSummDur, doctalkQaDur = reason_with_doctalk_FromFile(fname)
    '''  
    print('answerSQuAD, Talker:', Talker)
    print('answerSQuAD, Ripple:', Ripple)
    print('answerSQuAD, Thinker:', Thinker)
    print('answerSQuAD, Bert:', Bert)
    print('Total sentences:', totalSents)
    print('Total words:', totalWords)
    print("Stanza nlp parse duration(seconds): ", round(nlpParseDur, 5))
    print("doctalk Summarization duration(seconds): ", round(doctalkSummDur, 5))
    print("doctalk Q&A duration(seconds): ", doctalkQaDur)
    '''  
    totalSentsList.append(totalSents)
    totalWordsList.append(totalWords)
    nlpParseDurList.append(nlpParseDur)
    doctalkSummDurList.append(doctalkSummDur)
    talker_QA_self_list.append(doctalkQaDur['talker']['self'])
    talker_QA_bert_list.append(doctalkQaDur['talker']['bert'])
    ripple_QA_self_list.append(doctalkQaDur['ripple']['self'])
    ripple_QA_bert_list.append(doctalkQaDur['ripple']['bert'])
    thinker_QA_self_list.append(doctalkQaDur['thinker']['self'])
    thinker_QA_bert_list.append(doctalkQaDur['thinker']['bert'])
    bert_QA_bert_list.append(doctalkQaDur['bert']['bert'])


    for i, paragraph in enumerate(article['paragraphs']):      
      for qa in paragraph['qas']:
        qid = qa['id']
        qids.append(qid)
      
    for j, qid in enumerate(qids):
      qidAnswer_Talker[qid] = Talker[j]
      qidAnswer_Ripple[qid] = Ripple[j]
      qidAnswer_Thinker[qid] = Thinker[j]
      qidAnswer_Bert[qid] = Bert[j]
      #if j ==1: break
    '''
    print('\n\nDone, save to files')
    print('qidAnswer_Talker:', qidAnswer_Talker)
    print('qidAnswer_Ripple:', qidAnswer_Ripple)
    print('qidAnswer_Thinker:', qidAnswer_Thinker)
    print('qidAnswer_Bert:', qidAnswer_Bert)
    print('totalSentsList:', totalSentsList)
    print('totalWordsList:', totalWordsList)
    print("nlpParseDurList: ", nlpParseDurList)
    print("doctalkSummDurList: ", doctalkSummDurList)
    print('talker_QA_self_list:', talker_QA_self_list)
    print('talker_QA_bert_list:', talker_QA_bert_list)
    print('ripple_QA_self_list:', ripple_QA_self_list)
    print('ripple_QA_bert_list:', ripple_QA_bert_list)
    print('thinker_QA_self_list:', thinker_QA_self_list)
    print('thinker_QA_bert_list:', thinker_QA_bert_list)
    print('bert_QA_bert_list:', bert_QA_bert_list)
    '''
    
    outputTalker = json.dumps(qidAnswer_Talker)
    with wopen(outputDir + 'predictions_talker.json' ) as ftalk:
      ftalk.write(outputTalker + "\n")

    outputRipple = json.dumps(qidAnswer_Ripple)
    with wopen(outputDir + 'predictions_ripple.json' ) as fRipple:
      fRipple.write(outputRipple + "\n")
    
    outputThinker = json.dumps(qidAnswer_Thinker)
    with wopen(outputDir + 'predictions_thinker.json' ) as fthink:
      fthink.write(outputThinker + "\n")

    outputBert = json.dumps(qidAnswer_Bert)
    with wopen(outputDir + 'predictions_bert.json' ) as fbert:
      fbert.write(outputBert + "\n")

    saveStats_WordsDuration(outputDir, totalSentsList, totalWordsList, 
        nlpParseDurList, doctalkSummDurList, 
        talker_QA_self_list, talker_QA_bert_list,
        ripple_QA_self_list, ripple_QA_bert_list,
        thinker_QA_self_list, thinker_QA_bert_list,
        bert_QA_bert_list )
    #if count == 3: break
    

##########################################################################################
#for NewsQA
#########################################################################################

def saveNewQA_QuestionContent():
  datadir = "dataset/NewsQA/"
  os.makedirs(datadir + 'dev',exist_ok=True)
  os.makedirs(datadir + 'answer',exist_ok=True)

  dataset= jload( datadir + 'combined-newsqa-data-v1.json')
  #data is []
  print('how many storys:', len(dataset['data']))

  for i, article in enumerate(dataset['data']):
    storyId = article["storyId"]
    storyId  =  storyId [len("./cnn/stories/"):]   
    keeplen= len(storyId) - len(".story")
    storyId = storyId[:keeplen]    
    
    fname = datadir + "dev/" + storyId + ".txt"
    conext = article['text']
    with wopen(fname) as fcontext :
      fcontext.write(conext + "\n")
          
    questions = article['questions']
    qstring = ""
    astring = ""
    answerMap =dict()  
    
    for j, question in enumerate(questions):
      q=question['q']
      qstring += q + "\n"
      a=question['consensus']
      #print('a:', a)
      if 'badQuestion' in a.keys():
        #print("*****find bad question")
        answer = ""
      elif 'noAnswer' in a.keys():
        #print("******noAnswer")
        answer = ""
      else:
        start = a['s']
        end = a['e']
        #print('start:end, ', start, end )
        answer = conext[a['s']:a['e']]
      #print('answer:', answer)
      answerMap[storyId + "_" + str(j)] = answer
      

    fqname = datadir + "dev/" + storyId + "_quest.txt" 
    with wopen(fqname) as fquest:
      fquest.write(qstring + "\n")
    
    faname = datadir + "answer/" + storyId + ".txt"
    outputAnswer = json.dumps(answerMap)
    with wopen(faname) as fanswer:
      fanswer.write( outputAnswer + "\n" )


def answerNewsQA():
  datadir = "dataset/NewsQA/"
  outputDir = 'dataset/NewsQA/output/'
  os.makedirs(outputDir, exist_ok=True)
  os.makedirs(outputDir + 'talker',exist_ok=True)
  os.makedirs(outputDir + 'ripple',exist_ok=True)
  os.makedirs(outputDir + 'thinker',exist_ok=True)
  os.makedirs(outputDir + 'bert',exist_ok=True)
  dataset= jload( datadir + 'combined-newsqa-data-v1.json')

  loadResult(outputDir)

  for i, article in enumerate(dataset['data']):
    #if i < 1: continue
    storyId = article["storyId"]
    storyId  =  storyId [len("./cnn/stories/"):]
    keeplen= len(storyId) - len(".story")
    storyId = storyId[:keeplen]
    fname = datadir + "dev/" + storyId

  
    Talker, Ripple, Thinker, Bert,totalSents, totalWords, nlpParseDur, doctalkSummDur, doctalkQaDur = reason_with_doctalk_FromFile(fname)
    '''  
    print('answerNewsQA, Talker:', Talker)
    print('answerNewsQA, Ripple:', Ripple)
    print('answerNewsQA, Thinker:', Thinker)
    print('answerNewsQA, Bert:', Bert)
    print('Total sentences:', totalSents)
    print('Total words:', totalWords)
    print("Stanza nlp parse duration(seconds): ", round(nlpParseDur, 5))
    print("doctalk Summarization duration(seconds): ", round(doctalkSummDur, 5))
    print("doctalk Q&A duration(seconds): ", doctalkQaDur)
    '''  
    totalSentsList.append(totalSents)
    totalWordsList.append(totalWords)
    nlpParseDurList.append(nlpParseDur)
    doctalkSummDurList.append(doctalkSummDur)
    talker_QA_self_list.append(doctalkQaDur['talker']['self'])
    talker_QA_bert_list.append(doctalkQaDur['talker']['bert'])
    ripple_QA_self_list.append(doctalkQaDur['ripple']['self'])
    ripple_QA_bert_list.append(doctalkQaDur['ripple']['bert'])
    thinker_QA_self_list.append(doctalkQaDur['thinker']['self'])
    thinker_QA_bert_list.append(doctalkQaDur['thinker']['bert'])
    bert_QA_bert_list.append(doctalkQaDur['bert']['bert'])


    
    questions = article['questions']
    qidAnswer_Talker =dict()  
    qidAnswer_Ripple =dict()  
    qidAnswer_Thinker =dict()
    qidAnswer_Bert =dict()
  
    for j, question in enumerate(questions):
      qidAnswer_Talker[storyId + "_" + str(j)] = Talker[j]
      qidAnswer_Ripple[storyId + "_" + str(j)] = Ripple[j] 
      qidAnswer_Thinker[storyId + "_" + str(j)] = Thinker[j]
      qidAnswer_Bert[storyId + "_" + str(j)] = Bert[j]
    
    fname = outputDir + "talker/" + storyId + ".txt"
    outputTalker = json.dumps(qidAnswer_Talker)
    with wopen(fname) as ftalk:
      ftalk.write(outputTalker + "\n")

    fname = outputDir + "ripple/" + storyId + ".txt"
    outputRipple = json.dumps(qidAnswer_Ripple)
    with wopen(fname) as fRipple:
      fRipple.write(outputRipple + "\n")
  
    outputThinker = json.dumps(qidAnswer_Thinker)
    fname = outputDir + "thinker/" + storyId + ".txt"
    with wopen(fname) as fthink:
      fthink.write(outputThinker + "\n")

    outputBertAnswer = json.dumps(qidAnswer_Bert)
    fname = outputDir + "bert/" + storyId + ".txt"
    with wopen(fname) as fbert:
      fbert.write(outputBertAnswer + "\n")
    
    saveStats_WordsDuration(outputDir, totalSentsList, totalWordsList, 
        nlpParseDurList, doctalkSummDurList, 
        talker_QA_self_list, talker_QA_bert_list,
        ripple_QA_self_list, ripple_QA_bert_list,
        thinker_QA_self_list, thinker_QA_bert_list,
        bert_QA_bert_list )
    #if i==1:break

###########################################################################################################################
### below if for narrativeqa
##############################################################################################################################
def saveNarrativeqa_QuestionContent():
  baseDir = 'dataset/Narrativeqa/'
  os.makedirs(baseDir + 'dev/', exist_ok=True)
  os.makedirs(baseDir + 'output/', exist_ok=True)
  gitDir = baseDir + '/narrativeqa_github/'
  import csv
  with open(gitDir + '/third_party/wikipedia/summaries.csv', newline='') as csvfile:
    dataset = csv.DictReader(csvfile)
    for row in dataset:
      if row['set'] == 'test':
        fname = baseDir + "dev/" + row['document_id'] + ".txt"
        with wopen(fname) as fcontext :
          fcontext.write(row['summary'] + '\n')

  from collections import defaultdict
  dqs = defaultdict(list)
  with open(gitDir + 'qaps.csv', newline='') as csvfile:
    questionset = csv.DictReader(csvfile)
    for row in questionset:
      if row['set'] == 'test':
        dqs[row['document_id']].append(row['question'])

  for id in dqs:
    questions = "\n".join(dqs[id])
    fname = baseDir + "dev/" + id + "_quest.txt"
    with wopen(fname) as f :
      f.write(questions + '\n')
 

def answerNarrativeqa():
  baseDir = "dataset/Narrativeqa/"
  outputDir = baseDir + 'output/'
  gitDir = baseDir + 'narrativeqa_github/'
  os.makedirs(outputDir, exist_ok=True)
  os.makedirs(outputDir + 'talker',exist_ok=True)
  os.makedirs(outputDir + 'ripple',exist_ok=True)
  os.makedirs(outputDir + 'thinker',exist_ok=True)
  os.makedirs(outputDir + 'bert',exist_ok=True)

  loadResult(outputDir)
  import csv
  from collections import defaultdict
  dataIds = []
  with open(gitDir + '/third_party/wikipedia/summaries.csv', newline='') as csvfile:
    dataset = csv.DictReader(csvfile)
    for row in dataset:
      if row['set'] != 'test': continue
      dataIds.append(row['document_id'])
  dqs = defaultdict(list)
  with open(gitDir + 'qaps.csv', newline='') as csvfile:
    questionset = csv.DictReader(csvfile)
    for row in questionset:
      if row['set'] != 'test': continue
      dqs[row['document_id']].append(row['question']) 
  #i = 0
  for document_id in dataIds:
    fname = baseDir + "dev/" + document_id
    if os.path.exists(fname + '.txt') == False:
      continue
    Talker, Ripple, Thinker, Bert,totalSents, totalWords, nlpParseDur, doctalkSummDur, doctalkQaDur = reason_with_doctalk_FromFile(fname)
    '''
    print('answerNewsQA, Talker:', Talker)
    print('answerNewsQA, Ripple:', Ripple)
    print('answerNewsQA, Thinker:', Thinker)
    print('answerNewsQA, Bert:', Bert)
    print('Total sentences:', totalSents)
    print('Total words:', totalWords)
    print("Stanza nlp parse duration(seconds): ", round(nlpParseDur, 5))
    print("doctalk Summarization duration(seconds): ", round(doctalkSummDur, 5))
    print("doctalk Q&A duration(seconds): ", doctalkQaDur)
    '''
    totalSentsList.append(totalSents)
    totalWordsList.append(totalWords)
    nlpParseDurList.append(nlpParseDur)
    doctalkSummDurList.append(doctalkSummDur)
    talker_QA_self_list.append(doctalkQaDur['talker']['self'])
    talker_QA_bert_list.append(doctalkQaDur['talker']['bert'])
    ripple_QA_self_list.append(doctalkQaDur['ripple']['self'])
    ripple_QA_bert_list.append(doctalkQaDur['ripple']['bert'])
    thinker_QA_self_list.append(doctalkQaDur['thinker']['self'])
    thinker_QA_bert_list.append(doctalkQaDur['thinker']['bert'])
    bert_QA_bert_list.append(doctalkQaDur['bert']['bert'])
    

    for j, question in enumerate(dqs[document_id]):
      qidAnswer_Talker[question] = Talker[j]
      qidAnswer_Ripple[question] = Ripple[j] 
      qidAnswer_Thinker[question] = Thinker[j]
      qidAnswer_Bert[question] = Bert[j]
    
    '''
    print('\n\nDone, save to files')
    print('qidAnswer_Talker:', qidAnswer_Talker)
    print('qidAnswer_Ripple:', qidAnswer_Ripple)
    print('qidAnswer_Thinker:', qidAnswer_Thinker)
    print('qidAnswer_Bert:', qidAnswer_Bert)
    print('totalSentsList:', totalSentsList)
    print('totalWordsList:', totalWordsList)
    print("nlpParseDurList: ", nlpParseDurList)
    print("doctalkSummDurList: ", doctalkSummDurList)
    print('talker_QA_self_list:', talker_QA_self_list)
    print('talker_QA_bert_list:', talker_QA_bert_list)
    print('ripple_QA_self_list:', ripple_QA_self_list)
    print('ripple_QA_bert_list:', ripple_QA_bert_list)
    print('thinker_QA_self_list:', thinker_QA_self_list)
    print('thinker_QA_bert_list:', thinker_QA_bert_list)
    print('bert_QA_bert_list:', bert_QA_bert_list)
    '''

    fname = outputDir + "talker/" + document_id + ".txt"
    outputTalker = json.dumps(qidAnswer_Talker)
    with wopen(fname) as ftalk:
      ftalk.write(outputTalker + "\n")

    fname = outputDir + "ripple/" + document_id + ".txt"
    outputRipple = json.dumps(qidAnswer_Ripple)
    with wopen(fname) as fRipple:
      fRipple.write(outputRipple + "\n")
  
    outputThinker = json.dumps(qidAnswer_Thinker)
    fname = outputDir + "thinker/" + document_id + ".txt"
    with wopen(fname) as fthink:
      fthink.write(outputThinker + "\n")

    outputBertAnswer = json.dumps(qidAnswer_Bert)
    fname = outputDir + "bert/" + document_id + ".txt"
    with wopen(fname) as fbert:
      fbert.write(outputBertAnswer + "\n")
    
    saveStats_WordsDuration(outputDir, totalSentsList, totalWordsList, 
        nlpParseDurList, doctalkSummDurList, 
        talker_QA_self_list, talker_QA_bert_list,
        ripple_QA_self_list, ripple_QA_bert_list,
        thinker_QA_self_list, thinker_QA_bert_list,
        bert_QA_bert_list )
    #i = i+1
    #if i==1:break


###########################################################################################################################
### below if for HotpotQA
##############################################################################################################################
def saveHotpotQA_QuestionContent():
  os.makedirs('dataset/HotpotQA/dev/', exist_ok=True)
  os.makedirs('dataset/HotpotQA/answer/', exist_ok=True)
  dataset= jload('dataset/HotpotQA/hotpot_dev_distractor_v1.json')  
  #data is []
  print('dataset length:', len(dataset))
  for i, article in enumerate(dataset):    
    quest_id = article["_id"]
    #print('quest_id:', quest_id)
    question = article["question"]
    #print('question ', i, ':', question)
    fqname = "dataset/HotpotQA/dev/" + quest_id + "_quest.txt" 
    with wopen(fqname) as fquest:
        fquest.write(question + "\n")
    
    answer = article["answer"]
    fqname = "dataset/HotpotQA/answer/" + quest_id + ".txt" 
    with wopen(fqname) as fanswer:
      fanswer.write(answer + "\n")

    text = ''
    paralist = article["context"]
    #print('paralist type:', type(paralist))
    for para in paralist:
      #title = para[0]
      #print('\n\ntitle:', title)
      text += '\n'
      sentences = para[1]
      #print('sentences:', sentences)
      for sent in sentences:
        text += sent
    
    fname = "dataset/HotpotQA/dev/" + quest_id + ".txt"
    with wopen(fname) as fcontext :
      fcontext.write(text + '\n')
    #if i > 10 : break  


def answerHotpotQA():
  outputDir = 'dataset/HotpotQA/output/'
  os.makedirs(outputDir, exist_ok=True)
  dataset= jload('dataset/HotpotQA/hotpot_dev_distractor_v1.json')
  #data is []
  loadResult(outputDir)

  for i, article in enumerate(dataset): 
    #if i<12: continue   
    quest_id = article["_id"]
    fname = "dataset/HotpotQA/dev/" + quest_id
    Talker, Ripple, Thinker, Bert,totalSents, totalWords, nlpParseDur, doctalkSummDur, doctalkQaDur = reason_with_doctalk_FromFile(fname)
    '''
    print('answerNewsQA, Talker:', Talker)
    print('answerNewsQA, Ripple:', Ripple)
    print('answerNewsQA, Thinker:', Thinker)
    print('answerNewsQA, Bert:', Bert)
    print('Total sentences:', totalSents)
    print('Total words:', totalWords)
    print("Stanza nlp parse duration(seconds): ", round(nlpParseDur, 5))
    print("doctalk Summarization duration(seconds): ", round(doctalkSummDur, 5))
    print("doctalk Q&A duration(seconds): ", doctalkQaDur)
    '''
    qidAnswer_Talker[quest_id] = Talker[0]
    qidAnswer_Ripple[quest_id] = Ripple[0]
    qidAnswer_Thinker[quest_id] = Thinker[0]
    qidAnswer_Bert[quest_id] = Bert[0]
    totalSentsList.append(totalSents)
    totalWordsList.append(totalWords)
    nlpParseDurList.append(nlpParseDur)
    doctalkSummDurList.append(doctalkSummDur)
    talker_QA_self_list.append(doctalkQaDur['talker']['self'])
    talker_QA_bert_list.append(doctalkQaDur['talker']['bert'])
    ripple_QA_self_list.append(doctalkQaDur['ripple']['self'])
    ripple_QA_bert_list.append(doctalkQaDur['ripple']['bert'])
    thinker_QA_self_list.append(doctalkQaDur['thinker']['self'])
    thinker_QA_bert_list.append(doctalkQaDur['thinker']['bert'])
    bert_QA_bert_list.append(doctalkQaDur['bert']['bert'])

    '''
    print('\n\nDone, save to files')
    print('qidAnswer_Talker:', qidAnswer_Talker)
    print('qidAnswer_Ripple:', qidAnswer_Ripple)
    print('qidAnswer_Thinker:', qidAnswer_Thinker)
    print('qidAnswer_Bert:', qidAnswer_Bert)
    print('totalSentsList:', totalSentsList)
    print('totalWordsList:', totalWordsList)
    print("nlpParseDurList: ", nlpParseDurList)
    print("doctalkSummDurList: ", doctalkSummDurList)
    print('talker_QA_self_list:', talker_QA_self_list)
    print('talker_QA_bert_list:', talker_QA_bert_list)
    print('ripple_QA_self_list:', ripple_QA_self_list)
    print('ripple_QA_bert_list:', ripple_QA_bert_list)
    print('thinker_QA_self_list:', thinker_QA_self_list)
    print('thinker_QA_bert_list:', thinker_QA_bert_list)
    print('bert_QA_bert_list:', bert_QA_bert_list)
    '''
    
    outputTalker = json.dumps(qidAnswer_Talker)
    with wopen(outputDir + 'predictions_talker.json' ) as ftalk:
      ftalk.write(outputTalker + "\n")

    outputRipple = json.dumps(qidAnswer_Ripple)
    with wopen(outputDir + 'predictions_ripple.json' ) as fRipple:
      fRipple.write(outputRipple + "\n")
    
    outputThinker = json.dumps(qidAnswer_Thinker)
    with wopen(outputDir + 'predictions_thinker.json' ) as fthink:
      fthink.write(outputThinker + "\n")

    outputBert = json.dumps(qidAnswer_Bert)
    with wopen(outputDir + 'predictions_bert.json' ) as fbert:
      fbert.write(outputBert + "\n")

    saveStats_WordsDuration(outputDir, totalSentsList, totalWordsList, 
        nlpParseDurList, doctalkSummDurList, 
        talker_QA_self_list, talker_QA_bert_list,
        ripple_QA_self_list, ripple_QA_bert_list,
        thinker_QA_self_list, thinker_QA_bert_list,
        bert_QA_bert_list )
    #if i ==500 : break


###########################################################################################################################
### below if for biased_textrank
##############################################################################################################################
def saveTextrank_QuestionContent():
  os.makedirs('dataset/textrank/dev/', exist_ok=True)
  os.makedirs('dataset/textrank/answer/', exist_ok=True)
  dataset= jload('dataset/textrank/biased_textrank_git/data/liar/clean_test.json')  
  #data is []
  print('dataset length:', len(dataset))
  for article in dataset:    
    document_id = article["id"]
    question = article["claim"].replace('&nbsp;', '') 
    fqname = "dataset/textrank/dev/" + str(document_id) + "_quest.txt" 
    with wopen(fqname) as f:
        f.write(question + "\n")    

    text = article["statements"].replace('&nbsp;', '') 
    fname = "dataset/textrank/dev/" + str(document_id) + ".txt"
    with wopen(fname) as f :
      f.write(text + '\n')

    answer = article["new_justification"].replace('&nbsp;', '')      
    fqname = "dataset/textrank/answer/" + str(document_id) + ".txt" 
    with wopen(fqname) as f:
      f.write(answer + "\n")


def answerTextrank():
  outputDir = 'dataset/textrank/output/'
  os.makedirs(outputDir, exist_ok=True)
  os.makedirs(outputDir + 'talker',exist_ok=True)
  os.makedirs(outputDir + 'ripple',exist_ok=True)
  os.makedirs(outputDir + 'thinker',exist_ok=True)
  loadResult(outputDir)

  '''
  print('\n\nbefore start')
  print('totalSentsList:', totalSentsList)
  print('totalWordsList:', totalWordsList)
  print("nlpParseDurList: ", nlpParseDurList)
  print("doctalkSummDurList: ", doctalkSummDurList)
  print("talker_QA_self_list: ", talker_QA_self_list)
  print("ripple_QA_self_list: ", ripple_QA_self_list)
  print("thinker_QA_self_list: ", thinker_QA_self_list)
  '''

  for id in range(0, 1283):
    fname = "dataset/textrank/dev/" + str(id)
    print(fname)
    if os.path.exists(fname + '.txt') == False:
      continue
    Talker, Ripple, Thinker, _ ,totalSents, totalWords, nlpParseDur, doctalkSummDur, doctalkQaDur = reason_with_doctalk_FromFile(fname)
    '''  
    print('Talker:', Talker)
    print('Ripple:', Ripple)
    print('Thinker:', Thinker)
    print('Total sentences:', totalSents)
    print('Total words:', totalWords)
    print("Stanza nlp parse duration(seconds): ", round(nlpParseDur, 5))
    print("doctalk Summarization duration(seconds): ", round(doctalkSummDur, 5))
    print("doctalk Q&A duration(seconds): ", doctalkQaDur)
    '''  
    totalSentsList.append(totalSents)
    totalWordsList.append(totalWords)
    nlpParseDurList.append(nlpParseDur)
    doctalkSummDurList.append(doctalkSummDur)
    talker_QA_self_list.append(doctalkQaDur['talker']['self'])
    ripple_QA_self_list.append(doctalkQaDur['ripple']['self'])
    thinker_QA_self_list.append(doctalkQaDur['thinker']['self'])

    fname = outputDir + "talker/" + str(id) + ".txt"
    outputTalker = json.dumps(qidAnswer_Talker)
    with wopen(fname) as ftalk:
      ftalk.write(Talker[0] + "\n")

    fname = outputDir + "ripple/" + str(id) + ".txt"
    outputRipple = json.dumps(qidAnswer_Ripple)
    with wopen(fname) as fRipple:
      fRipple.write(Ripple[0] + "\n")
  
    outputThinker = json.dumps(qidAnswer_Thinker)
    fname = outputDir + "thinker/" + str(id) + ".txt"
    with wopen(fname) as fthink:
      fthink.write(Thinker[0] + "\n")


    '''
    print('\n\nDone, save to files')
    print('totalSentsList:', totalSentsList)
    print('totalWordsList:', totalWordsList)
    print("nlpParseDurList: ", nlpParseDurList)
    print("doctalkSummDurList: ", doctalkSummDurList)
    print('talker_QA_self_list:', talker_QA_self_list)
    print('ripple_QA_self_list:', ripple_QA_self_list)
    print('thinker_QA_self_list:', thinker_QA_self_list)
    '''
    saveStats_WordsDuration(outputDir, totalSentsList, totalWordsList, 
        nlpParseDurList, doctalkSummDurList, 
        talker_QA_self_list, talker_QA_bert_list,
        ripple_QA_self_list, ripple_QA_bert_list,
        thinker_QA_self_list, thinker_QA_bert_list,
        bert_QA_bert_list )
    




def saveCoQA_QuestionContent():
  dataset= jload('dataset/CoQA/coqa-dev-v1.0.json')  
  print('data length:', len(dataset['data']))  
  for article in dataset['data']:
    quest_id = article["id"]
    fname = "dataset/CoQA/dev/" + quest_id + ".txt"
    conext = article['story']
    with wopen(fname) as fcontext :
      fcontext.write(conext + "\n")
    fcontext.close()          
    questions = article['questions']
    fqname = "dataset/CoQA/dev/" + quest_id + "_quest.txt" 
    with wopen(fqname) as fquest:
      for question in questions:
        q=question['input_text']
        fquest.write(q + "\n")
      fquest.close()
      questions = article['questions']
    answers = article['answers']
    faname = "dataset/CoQA/answer/" + quest_id + ".txt" 
    with wopen(faname) as fanswer:
      for answer in answers:
        span_text=answer['span_text']
        input_text = answer['input_text']
        fanswer.write(span_text + "\n" + input_text + "\n")
      fanswer.close()


def answerCoQA():
  dataset= jload('dataset/CoQA/coqa-dev-v1.0.json')
  #data is []
  qidTalkAnswerMap =dict()
  qidThinkAnswerMap =dict() 
  qidDiffAnswerMap =dict()  
  for i, article in enumerate(dataset['data']):    
    quest_id = article["id"]
    fname = "dataset/CoQA/dev/" + quest_id
    talkans, thinkans = reason_with_doctalk_FromFile(fname)
    print('answerCoQA:', talkans, ',', thinkans )
    '''
    if not talkans[0]: talkans[0] = ''
    qidTalkAnswerMap[quest_id] = talkans[0]
    if not thinkans[0]: thinkans[0] = ''
    qidThinkAnswerMap[quest_id] = thinkans[0]
    if talkans != thinkans:
      diff = 'talk:' + talkans[0] + '; think:' + thinkans[0]
      qidDiffAnswerMap[quest_id] = diff
    '''
    if i > 1 : break
    '''
  print('\nqidTalkAnswerMap:', qidTalkAnswerMap)
  print('\nqidThinkAnswerMap:', qidThinkAnswerMap)
  print('\nqidDiffAnswerMap', qidDiffAnswerMap)

  answerTalkMap = dict()
  answerTalkMap["answer"] = qidTalkAnswerMap      
  outputTalk = json.dumps(answerTalkMap)
  fname = "dataset/HotpotQA/pred_talk.json"
  with wopen(fname) as ftalk:
    ftalk.write(outputTalk + "\n")
  ftalk.close()

  answerThinkMap = dict()
  answerThinkMap["answer"] = qidThinkAnswerMap 
  outputThink = json.dumps(answerThinkMap)
  fname = "dataset/HotpotQA/pred_think.json"
  with wopen(fname) as fthink:
    fthink.write(outputThink + "\n")
  fthink.close()
  outputDiff = json.dumps(qidDiffAnswerMap)
  fname = "dataset/HotpotQA/pred_diff.json"
  with wopen(fname) as fdiff:
    fdiff.write(outputDiff + "\n")
  fdiff.close()
  '''



def reason_with_doctalk_FromFile(fname) :  
  params = talk_params()
  params.with_answerer=True
  params.top_answers = 4
  params.answers_by_rank = True
  Talker, Ripple, Thinker, Bert, totalSents, totalWords, nlpParseDur, doctalkSumDur, QADur= reason_with_File(fname, params)
  return Talker, Ripple, Thinker, Bert, totalSents, totalWords, nlpParseDur, doctalkSumDur, QADur


def reason_with_doctalk_FromText(text, qlist) :  
  params = talk_params()
  params.with_answerer=True
  params.answers_by_rank = True
  Talker, Ripple, Thinker, Bert = reason_with_Text(text, qlist, params)
  return Talker, Ripple, Thinker, Bert
  

def saveStats_WordsDuration(outputDir, totalSentsList, totalWordsList, 
                  nlpParseDurList, doctalkSummDurList,
                  tlkqss, tlkqbs, rqss, rqbs, thkqss, thkqbs, bbs ):
  outputTotalSents = json.dumps(totalSentsList)
  with wopen(outputDir + 'Total_Sents.json' ) as f:
    f.write(outputTotalSents + "\n")

  outputTotalWords = json.dumps(totalWordsList)
  with wopen(outputDir + 'Total_Words.json' ) as f:
    f.write(outputTotalWords + "\n")

  outputNlpParseDur = json.dumps(nlpParseDurList)
  with wopen(outputDir + 'nlpParse_duration.json' ) as f:
    f.write(outputNlpParseDur + "\n")

  outputDoctalkSummDur = json.dumps(doctalkSummDurList)
  with wopen(outputDir + 'DoctalkSumm_duration.json' ) as f:
    f.write(outputDoctalkSummDur + "\n")

  with wopen(outputDir + 'QA_talk_self_duration.json' ) as f:
    f.write(json.dumps(tlkqss) + "\n")
  with wopen(outputDir + 'QA_talk_bert_duration.json' ) as f:
    f.write(json.dumps(tlkqbs) + "\n")
  with wopen(outputDir + 'QA_ripple_self_duration.json' ) as f:
    f.write(json.dumps(rqss) + "\n")
  with wopen(outputDir + 'QA_ripple_bert_duration.json' ) as f:
    f.write(json.dumps(rqbs) + "\n")
  with wopen(outputDir + 'QA_thinker_self_duration.json' ) as f:
    f.write(json.dumps(thkqss) + "\n")
  with wopen(outputDir + 'QA_thinker_bert_duration.json' ) as f:
    f.write(json.dumps(thkqbs) + "\n")
  with wopen(outputDir + 'QA_bert_bert_duration.json' ) as f:
    f.write(json.dumps(bbs) + "\n")
 

def loadResult(outputdir):
  global qidAnswer_Talker, qidAnswer_Ripple, qidAnswer_Thinker, qidAnswer_Bert
  global totalSentsList, totalWordsList
  global nlpParseDurList, doctalkSummDurList
  global talker_QA_self_list, talker_QA_bert_list, ripple_QA_self_list, ripple_QA_bert_list
  global thinker_QA_self_list, thinker_QA_bert_list, bert_QA_bert_list

  if os.path.exists(outputdir +  'predictions_talker.json'):
    qidAnswer_Talker = jload( outputdir +  'predictions_talker.json')
  if os.path.exists(outputdir +  'predictions_ripple.json'):
    qidAnswer_Ripple = jload( outputdir +  'predictions_ripple.json')
  if os.path.exists(outputdir +  'predictions_thinker.json'):
    qidAnswer_Thinker = jload( outputdir +  'predictions_thinker.json')
  if os.path.exists(outputdir +  'predictions_bert.json'):
    qidAnswer_Bert = jload( outputdir +  'predictions_bert.json')
  if os.path.exists(outputdir +  'Total_Sents.json'):
    totalSentsList = jload( outputdir +  'Total_Sents.json')
  if os.path.exists(outputdir +  'Total_Words.json'):
    totalWordsList = jload( outputdir +  'Total_Words.json')
  if os.path.exists(outputdir +  'nlpParse_duration.json'):
    nlpParseDurList = jload( outputdir +  'nlpParse_duration.json')
  if os.path.exists(outputdir +  'DoctalkSumm_duration.json'):
    doctalkSummDurList = jload( outputdir +  'DoctalkSumm_duration.json') 
  if os.path.exists(outputdir +  'QA_talk_self_duration.json'):
    talker_QA_self_list =  jload( outputdir +  'QA_talk_self_duration.json') 
  if os.path.exists(outputdir +  'QA_talk_bert_duration.json'):
    talker_QA_bert_list =  jload( outputdir +  'QA_talk_bert_duration.json') 
  if os.path.exists(outputdir +  'QA_ripple_self_duration.json'):
    ripple_QA_self_list =  jload( outputdir +  'QA_ripple_self_duration.json') 
  if os.path.exists(outputdir +  'QA_ripple_bert_duration.json'):
    ripple_QA_bert_list =  jload( outputdir +  'QA_ripple_bert_duration.json') 
  if os.path.exists(outputdir +  'QA_thinker_self_duration.json'):
    thinker_QA_self_list =  jload( outputdir +  'QA_thinker_self_duration.json') 
  if os.path.exists(outputdir +  'QA_thinker_bert_duration.json'):
    thinker_QA_bert_list =  jload( outputdir +  'QA_thinker_bert_duration.json') 
  if os.path.exists(outputdir +  'QA_bert_bert_duration.json'):
    bert_QA_bert_list =  jload( outputdir +  'QA_bert_bert_duration.json') 
  print('loadSQuADResult:')
  print('qidAnswer_Talker:', qidAnswer_Talker)
  print('qidAnswer_Ripple:', qidAnswer_Ripple)
  print('qidAnswer_Thinker:', qidAnswer_Thinker)
  print('qidAnswer_Bert:', qidAnswer_Bert)
  print('totalSentsList:', totalSentsList)
  print('totalWordsList:', totalWordsList)
  print("nlpParseDurList: ", nlpParseDurList)
  print("doctalkSummDurList: ", doctalkSummDurList)
  print("talker_QA_self_list: ", talker_QA_self_list)
  print("talker_QA_bert_list: ", talker_QA_bert_list)
  print("ripple_QA_self_list: ", ripple_QA_self_list)
  print("ripple_QA_bert_list: ", ripple_QA_bert_list)
  print("thinker_QA_self_list: ", thinker_QA_self_list)
  print("thinker_QA_bert_list: ", thinker_QA_bert_list)
  print("bert_QA_bert_list: ", bert_QA_bert_list)


if __name__ == '__main__' :
  pass



