from sklearn.metrics import f1_score
import re
import os
import sys
import string


def get_test_files(directory):
    test_files = []
    for root, sub_directory, files in os.walk(directory):
        for file in files:
            file_path = root + "/" + file
            if not re.search(ignore_files_pattern, file_path):
                test_files.append(file_path)
    return test_files

def get_model(model_file):
    nb_model = {}
    with open(model_file, "r") as f:
        for line in f:
            line = line.strip('\n')
            line = line.rstrip(',')
            nb_model[line.split(',')[0]] = line.split(',')[1:]
    return nb_model

def preprocess(line):
    stop_words = {'the', 'and', 'to', 'a', 'i', 'was', 'in', 'of', 'we', 'hotel', 'for', 'it', 'my', 'at', 'that', 'is',
                  'were', 'this', 'with', 'on', 'had', 'they', 'our', 'by', 'or', 'stayed', 'are', 'there', 'as', 'be',
                  'from', 'would', 'when', 'all', 'an', 'up', 'us', 'after', 'which', 'stay', 'have', 'you', 'so', 'me',
                  'front', 'their', 'rooms',
                  'been', 'did', 'room', 'place', 'day', 'its', 'here', 'first', 'she', 'hotels', 'your', 'made', 'am',
                  'any', 'people', 'staying', 'other', 'make',
                  'city', 'he'}
    token_list = []
    line = line.lower().replace(".", ' ').replace(",", ' ').replace("&", " ").replace('!', "").replace('-',' ').replace('/',' ')

    line = line.split(" ")
    for word in line:
        if word not in stop_words:
            word = word.translate(str.maketrans('', '', string.punctuation))
            word = word.rstrip().lstrip().strip("\n")
            word = re.sub(r'\d', '', word)
            if len(word) >= 1 and word not in stop_words:
                token_list.append(word)
    return token_list

def build_features(test_files):
    review_count = 0
    all_feature_vectors = []
    for file in test_files:
        review_count += 1
        text = open(file, "r")
        feature_vector = {}
        for line in text:
            token_list = preprocess(line)
            for word in token_list:
                if word in feature_vector:
                    feature_vector[word] += 1
                else:
                    feature_vector[word] = 1
        all_feature_vectors.append(feature_vector)
    return all_feature_vectors, review_count

def write_to_output_file(score1, score2, file_name, output_data):
    output_str = ""
    if (score1 > 0 and score2 > 0):
        output_str += "Truthful" + " " + "Positive" + str(file_name)
    elif (score1 < 0 and score2 < 0):
        output_str += "Deceptive" + " " + "Negative" + str(file_name)
    elif (score1 < 0 and score2 > 0):
        output_str += "Truthful" + " " + "Negative" + str(file_name)
    elif (score1 > 0 and score2 < 0):
        output_str += "Deceptive" + " " + "Positive" + str(file_name)
    output_data.append(output_str)
    return output_data

def predict(test_files, percep_model):
    all_features, review_count = build_features(test_files)
    weights1_list = percep_model['weights_positive_negative']
    weights2_list = percep_model['weights_true_deceptive']
    bias1 = float(percep_model['bias_positive_negative'][0])
    bias2 = float(percep_model['bias_true_deceptive'][0])

    weights1 = {}
    for i in range(len(weights1_list)):
        temp = weights1_list[i].split(":")
        weights1[temp[0]] = float(temp[1])
    weights2 = {}
    for j in range(len(weights2_list)):
        temp = weights2_list[j].split(":")
        weights2[temp[0]] = float(temp[1])
    output_data = []
    for test_file in test_files:
        f = open(test_file, "r")
        feature_vector = {}
        for line in f:
            token_list = preprocess(line)
            for word in token_list:
                if word in feature_vector:
                    feature_vector[word] += 1
                else:
                    feature_vector[word] = 1
        score1 = bias1
        for word in feature_vector.keys():
            if word in weights1:
                score1 += weights1[word] * feature_vector[word]
        score2 = bias2
        for word in feature_vector.keys():
            if word in weights2:
                score2 += weights2[word] * feature_vector[word]
        output_str = ""
        if score2 > 0:
            output_str += "truthful" + " "
        else:
            output_str += "deceptive" + " "
        if score1 > 0:
            output_str += "positive" + " "
        else:
            output_str += "negative" + " "
        output_str += test_file
        output_data.append(output_str)
    output_dict = {}
    for row in output_data:
        row = row.split(" ")
        output_dict[row[2]] = (row[0], row[1])
    expected = {"True":[], "Fake":[], "Pos":[], "Neg":[]}
    observed = {"True":[], "Fake":[], "Pos":[], "Neg":[]}
    for test_file in test_files:
        observed_val = output_dict[test_file]
        observed["True"].append(observed_val[0])
        observed['Pos'].append(observed_val[1])

        polarity = test_file.split('/')[6].split('_')[0]
        sentiment = test_file.split('/')[7].split('_')[0]
        expected['True'].append(sentiment)
        expected['Pos'].append(polarity)
    avg = 0
    for label in observed:
        if label == "Neg":
            label = "Pos"
        if label ==  "Fake":
            label = "True"
        print(f1_score(expected[label], observed[label], average='macro'))
        avg += f1_score(expected[label], observed[label], average='macro')
    print(avg/4)


    with open("percepoutput.txt", "w") as f:
        for index, row in enumerate(output_data):
            f.write(row)
            if index != len(output_data) -1 :
                f.write("\n")




def main():
    test_directory = sys.argv[2]
    test_files = get_test_files(test_directory)
    model_file = sys.argv[1]
    percep_model = get_model(model_file)
    predict(test_files, percep_model)




if __name__ == "__main__":
    ignore_files_pattern = r"(README|DS_Store|LICENSE)"
    main()