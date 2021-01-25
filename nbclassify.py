import sys
import os
import re
import math
import string

def get_model():
    nb_model = {}
    with open("nbmodel.txt","r") as f:
        for line in f:
            line = line.rstrip()
            line = line.rstrip(",")
            line = line.split(",")
            line[1:] = [float(i) for i in line[1:]]

            nb_model[line[0]] = line[1:]
    return nb_model

def get_test_files(test_directory):
    test_files = []
    for root,sub_directory,files in os.walk(test_directory):
        for file in files:
            file_path = root + "/" + file
            if not re.search(ignore_files_pattern, file_path):
                test_files.append(file_path)

    return test_files

def preprocess(line):
    stop_words = {'the', 'and', 'to', 'a', 'i', 'was', 'in', 'of', 'we', 'hotel', 'for', 'room', 'it', 'my', 'myself',
                  'at', 'that', 'this', 'then', 'than', 'is', 'were',
                  'not', 'with', 'on', 'had', 'have', 'chicago', 'they', 'but', 'our', 'very', 'stay', 'there', 'you',
                  'as', 'from', 'be', 'would', 'when', 'all'
        , 'so', 'staff', 'me', 'are', 'service', 'one', 'rooms', 'out', 'stayed', 'no', 'an', 'up', 'us', 'if', 'like',
                  'get', 'desk', 'night', 'just', 'again',
                  'will', 'time', 'about', 'after', 'location', 'bed', 'even', 'did', 'do', 'by', 'could', 'would',
                  'or', 'which', 'only', 'back', 'front', 'here', 'also', 'got',
                  'what', 'more', 'some', 'their', 'been', 'hotels', 'experience', 'bathroom', 'first', 'place',
                  'really', 'other', 'day', 'next', 'because', 'two', 'its',
                  'go', 'your', 'made', 'lobby', 'has', 'he', 'she', 'city', 'while', 'food', 'breakfast', 'any', 'am',
                  'pm', 'too', 'another', 'right', 'downtown', 'asked', 'restaurant', 'people',
                  'weekend', 'away', 'how', 'where', 'pool', 'went', 'told', 'beds', 'door', 'michigan', 'take',
                  'internet', 'wife', 'husband', 'trip', 'minutes', 'who', 'shower', 'before', 'going',
                  'them', 'did', 'water', 'was', 'wasnt', 'off', 'price', 'booked', 'took', 'being', 'every', "he",
                  "him", "his", "himself", "she", "her", "hers", "upon", 'both', 'said', 'left', 'around', 'above','below',
                  'know', 'knew', 'each', 'having', "yourself", "yourselves", "ourselves", "whom", "herself", "it",
                  "itself", 'should', 'money', 'make', 'into', 'nights', 'days', 'felt', 'feel', 'bit', 'th',
                  'hour', 'east', 'west', 'north', 'south', 'guests', 'guest', 'business', 'area', 'floor', 'see',
                  'saw', 'think', 'thought', 'suite', 'use', 'gave', 'toilet', 'sheets', 'building', 'youre', 'four',
                  'coming',
                  'making', 'id', 'leave', 'either', 'car', 'non', 'put'}
    token_list = []
    line = line.lower().replace(".",' ').replace(",",' ').replace("&", " ").replace('!',"").replace('-',' ').replace('/',' ')
    line = line.split(" ")

    for word in line:
        if word not in stop_words:
            word = word.translate(str.maketrans('','',string.punctuation))
            word = word.rstrip().lstrip().strip("\n")
            word = re.sub(r'\d','',word)
            if len(word) >= 1 and word not in stop_words :
                token_list.append(word)
    return token_list

def get_test_tokens_output(test_files, nb_model):

    output = []
    for test_file in test_files:
        results = {'positive': math.log(nb_model['Prior Probabilities'][0]),
                   'negative': math.log(nb_model['Prior Probabilities'][1]),
                   'truthful': math.log(nb_model['Prior Probabilities'][2]),
                   'deceptive': math.log(nb_model['Prior Probabilities'][3])}
        f = open(test_file,"r")
        for line in f:
            token_list = preprocess(line)
            for token in token_list:
                if token in nb_model:
                    results['positive'] += math.log(nb_model[token][0])
                    results['negative'] += math.log(nb_model[token][1])
                    results['truthful'] += math.log(nb_model[token][2])
                    results['deceptive'] += math.log(nb_model[token][3])
        label1 = "truthful" if results['truthful'] > results['deceptive'] else "deceptive"
        label2 = "positive" if results['positive'] > results['negative'] else "negative"
        row = [label1, label2, test_file]
        output.append(row)
    return output

def main():
    nb_model = get_model()
    test_directory = sys.argv[1]
    test_files = get_test_files(test_directory)
    output_data = get_test_tokens_output(test_files, nb_model)
    with open("nboutput.txt","w") as output_file:
        for row in output_data:
            output_str = ""
            for col in row:
                output_str += col + " "
            output_str += "\n"
            output_file.writelines(output_str)

if __name__ == "__main__":
    ignore_files_pattern = r"(README|DS_Store|LICENSE)"
    main()
