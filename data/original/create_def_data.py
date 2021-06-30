import pandas as pd 
import random
random.seed(100)
all_emotions = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise','neutral']

label2id = {
    "admiration":0,
    "amusement":1,
    "disapproval":10,
    "disgust":11,
    "embarrassment":12,
    "excitement":13,
    "fear":14,
    "gratitude":15,
    "grief":16,
    "joy":17,
    "love":18,
    "nervousness":19,
    "anger":2,
    "optimism":20,
    "pride":21,
    "realization":22,
    "relief":23,
    "remorse":24,
    "sadness":25,
    "surprise":26,
    "neutral":27,
    "annoyance":3,
    "approval":4,
    "caring":5,
    "confusion":6,
    "curiosity":7,
    "desire":8,
    "disappointment":9
  }
id2label = {
    "0": "admiration",
    "1": "amusement",
    "10": "disapproval",
    "11": "disgust",
    "12": "embarrassment",
    "13": "excitement",
    "14": "fear",
    "15": "gratitude",
    "16": "grief",
    "17": "joy",
    "18": "love",
    "19": "nervousness",
    "2": "anger",
    "20": "optimism",
    "21": "pride",
    "22": "realization",
    "23": "relief",
    "24": "remorse",
    "25": "sadness",
    "26": "surprise",
    "27": "neutral",
    "3": "annoyance",
    "4": "approval",
    "5": "caring",
    "6": "confusion",
    "7": "curiosity",
    "8": "desire",
    "9": "disappointment"
  }
definition_dict = {"0":"Finding something impressive or worthy of respect",
"1":"Finding something funny or being entertained",
"2":"A strong feeling of displeasure or antagonism",
"3":"Mild anger,irritation",
"4":"Having or expressing a favourable opinion",
"5":"Displaying kindness and concern for others",
"6":"Lack of understanding, uncertainty",
"7":"A strong desire to know or learn something",
"8":"A strong feeling of wanting something or wishing for something to happen",
"9":"Sadness or displeasure caused by the nonfulfillment of one’s hopes or expectations",
"10":"Having or expressing an unfavorable opinion",
"11":"Revulsion or strong disapproval aroused by something unpleasant or offensive",
"12":"Self-consciousness, shame, or awkwardness",
"13":"Feeling of great enthusiasm and eagerness",
"14":"Being afraid or worried",
"15":"A feeling of thankfulness and appreciation",
"16":"Intense sorrow, especially caused by someone’s death",
"17":"A feeling of pleasure and happiness",
"18":"A strong positive emotion of regard and affection",
"19":"Apprehension, worry, anxiety",
"20":"Hopefulness and confidence about the future or the success of something",
"21":"Pleasure or satisfaction due to ones own achievements or the achievements of those with whom one is closely associated",
"22":"Becoming aware of something",
"23":"Reassurance and relaxation following release from anxiety or distress",
"24":"Regret or guilty feeling",
"25":"Emotional pain, sorrow",
"26":"Feeling astonished, startled by something unexpected",
"27":"No emotions"
}
for mode in ["train","test","dev"]:
	file = mode+".tsv"
	df = pd.read_csv(file,sep="\t",names=["text","emo","id"])
	seq_labels = []
	defns = []
	text = []
	emo = []
	text_id = []
	for index, row in df.iterrows():
		emotions = list(map(int,row["emo"].split(",")))
		not_labels = list(set(range(28)) - set(emotions))
		random.shuffle(not_labels)
		for num, e in enumerate(emotions):
			defns.append(definition_dict[str(e)])
			seq_labels.append(0)
			text.append(row["text"])
			emo.append(row["emo"])
			text_id.append(row["id"])
			defns.append(definition_dict[str(not_labels[num])])
			seq_labels.append(1)
			text.append(row["text"])
			emo.append(row["emo"])
			text_id.append(row["id"])
	data = {"text":text,"defn":defns,"seq_label":seq_labels,"emo":emo,"id":text_id}
	df1 = pd.DataFrame(data)
	df1.to_csv(mode+"_def100.tsv",sep="\t",index=False,header=False)





