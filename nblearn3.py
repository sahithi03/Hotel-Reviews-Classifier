import sys
import os
from pathlib import Path
import re
import string
from collections import defaultdict
from collections import Counter

# preprocess each line to remove stop words and punctuation
def preprocess(line):

    stop_words = {'the','and','to','a','i','was','in','of','we','hotel','for','room','it','my','myself','at','that','this','then','than','is','were',
                  'not','with','on','had','have','chicago','they','but','our','very','stay','there','you','as','from','be','would','when','all'
                  ,'so','staff','me','are','service','one','rooms','out','stayed','no','an','up','us','if','like','get','desk','night','just','again',
                  'will','time','about','after','location','bed','even','did','do','by','could','would','or','which','only','back','front','here','also','got',
                  'what','more','some','their','been','hotels','experience','bathroom','first','place','really','other','day','next','because','two','its',
                  'go','your','made','lobby','has','he','she','city','while','food','breakfast','any','am','pm','too','another','right','downtown','asked','restaurant','people',
                  'weekend','away','how','where','pool','went','told','beds','door','michigan','take','internet','wife','husband','trip','minutes','who','shower','before','going',
                  'them','did','water','was','wasnt','off','price','booked','took','being','every', "he", "him", "his", "himself", "she", "her", "hers","upon",'both','said','left','around','above'
                   'below','know','knew','each','having',"yourself", "yourselves", "ourselves","whom","herself", "it", "itself",'should','money','make','into','nights','days','felt','feel','bit','th',
                  'hour','east','west','north','south','guests','guest','business','area','floor','see','saw','think','thought','suite','use','gave','toilet','sheets','building','youre','four','coming',
                  'making','id','leave','either','car','non','put'}
    token_list = []
    line = line.lower().replace(".", ' ').replace(",", ' ').replace("&", " ").replace('!', "").replace('-',' ').replace('/',' ')
    line = line.split(" ")
    for word in line:
        if word not in stop_words:
            word = word.translate(str.maketrans('','',string.punctuation))
            word = word.rstrip().lstrip().strip("\n")
            word = re.sub(r'\d','',word)
            if len(word) >= 1 and word not in stop_words :
                token_list.append(word)
    return token_list

# create histogram
def get_conditional_probability(train_files):
    truthful_reviews = 0
    positive_reviews = 0
    deceptive_reviews = 0
    negative_reviews = 0
    total_reviews = 0
    conditional_probability = {}
    all_words = defaultdict(int)
    for file in train_files:
        total_reviews += 1
        if "positive" in file.lower():
            positive_reviews += 1
        if "negative" in file.lower():
            negative_reviews += 1
        if "truthful" in file.lower():
            truthful_reviews += 1
        if "deceptive" in file.lower():
            deceptive_reviews += 1
        label1 = "positive" if "positive" in file.lower() else "negative"
        label2 = "truthful" if "truthful" in file.lower() else "deceptive"
        text = open(file, "r")
        for line in text:
            token_list = preprocess(line)
            for token in token_list:
                all_words[token] += 1
                if token in conditional_probability:
                    if label1 in conditional_probability[token]:
                        conditional_probability[token][label1] += 1
                    else:
                        conditional_probability[token][label1] = 1
                    if label2 in conditional_probability[token]:
                        conditional_probability[token][label2] += 1
                    else:
                        conditional_probability[token][label2] = 1
                else:
                    conditional_probability[token] = {label1: 1, label2: 1}
        text.close()
    labels = ["truthful", "deceptive", "positive", "negative"]
    for key in conditional_probability.keys():
        for label in labels:
            if label not in conditional_probability[key]:
                conditional_probability[key][label] = 0
    result = [conditional_probability, all_words, positive_reviews, negative_reviews, truthful_reviews, deceptive_reviews, total_reviews]
    return result

def remove_low_frequency_words(intermediate_result):
    cond_prob = intermediate_result[0]
    all_words = Counter(intermediate_result[1])
    print(all_words.most_common(300))
    for key, value in all_words.items():
        if value < 2:
            #print(key)
            del cond_prob[key]
    intermediate_result[0] = cond_prob
    return intermediate_result

#calculate probabilities and create a model
def calculate_probabilities(intermediate_result):
    cond_prob  = intermediate_result[0]
    all_words = intermediate_result[1]
    positive_reviews = intermediate_result[2]
    negative_reviews = intermediate_result[3]
    truthful_reviews = intermediate_result[4]
    deceptive_reviews = intermediate_result[5]
    total_reviews = intermediate_result[6]

    write_rows = []
    labels = ["truthful", "deceptive", "positive", "negative"]
    class_counts = {}
    prior_prob = {}

    for key,val in cond_prob.items():
        for label in labels:
            if label in class_counts:
                class_counts[label] += cond_prob[key][label]
            else:
                class_counts[label] = cond_prob[key][label]

    prior_prob['positive'] = positive_reviews / total_reviews
    prior_prob['negative'] = negative_reviews / total_reviews
    prior_prob['truthful'] = truthful_reviews / total_reviews
    prior_prob['deceptive'] = deceptive_reviews / total_reviews
    write_rows.append(["Prior Probabilities", prior_prob['positive'], prior_prob['negative'], prior_prob['truthful'],
                       prior_prob['deceptive']])
    for token in cond_prob.keys():
        for label in labels:
            cond_prob[token][label] = (cond_prob[token][label] + 1) / (len(cond_prob) + class_counts[label])

        write_rows.append([token, cond_prob[token]['positive'], cond_prob[token]['negative'], cond_prob[token]['truthful'],
                           cond_prob[token]['deceptive']])
    with open("nbmodel.txt", "w") as f:
        for row in write_rows:
            write_str = ""
            for word in row:
                write_str += str(word) + ","
            write_str += "\n"
            f.write(write_str)

# Get Training files with the path attached
def get_train_files(root_directory):
    train_files = []
    for root, sub_directory, files in os.walk(root_directory):
        for file in files:
            file_path = root + "/" + file
            if not re.search(ignore_files_pattern, file_path):
                train_files.append(file_path)
    return train_files

def main():
    root_directory = sys.argv[1]
    train_files = get_train_files(root_directory)
    intermediate_result = get_conditional_probability(train_files)
    cleaned_result = remove_low_frequency_words(intermediate_result)
    calculate_probabilities(cleaned_result)


if __name__ == "__main__":

    ignore_files_pattern = r"(README|DS_Store|LICENSE)"
    main()