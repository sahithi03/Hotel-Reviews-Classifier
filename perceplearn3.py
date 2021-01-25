import sys
import string
import re
import os
from collections import defaultdict
from collections import Counter
from random import shuffle

def get_train_files(root_directory):
    train_files = []
    for root, sub_directory, files in os.walk(root_directory):
        for file in files:
            file_path = root + "/" + file
            if not re.search(ignore_files_pattern, file_path):
                train_files.append(file_path)
    return train_files

def preprocess(line):
    stop_words = {'the', 'and', 'to', 'a', 'i', 'was', 'in', 'of', 'we', 'hotel', 'for', 'it', 'my', 'at', 'that', 'is',
                  'were', 'this', 'with', 'on', 'had', 'they', 'our', 'by', 'or', 'stayed', 'are', 'there', 'as', 'be',
                  'from', 'would', 'when', 'all', 'an', 'up', 'us', 'after', 'which', 'stay', 'have', 'you', 'so', 'me',
                  'front', 'their', 'rooms',
                  'been', 'did', 'room', 'place', 'day','its','here','first','she','hotels','your','made','am','any','people','another','staying','other','make',
                  'city','he','could'}
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

def fit(train_files):
    all_feature_vectors = {}
    pos_neg_label = {}
    tru_dec_label = {}
    review_count = 0
    all_words = defaultdict(int)
    for file in train_files:
        label1 = 1 if "positive" in file.lower() else -1
        label2 = 1 if "truthful" in file.lower() else -1
        review_count += 1
        text = open(file, "r")
        feature_vector = {}
        for line in text:
            token_list = preprocess(line)
            for word in token_list:
                all_words[word] += 1
                if word in feature_vector:
                    feature_vector[word] += 1
                else:
                    feature_vector[word] = 1

        pos_neg_label[file] = label1
        tru_dec_label[file] = label2
        all_feature_vectors[file] = feature_vector
    remove_words = set()
    for word in all_words.keys():
        if all_words[word] < 2:
            remove_words.add(word)
    #print(remove_words)
    print(Counter(all_words).most_common(100))
    return all_feature_vectors, pos_neg_label, tru_dec_label, remove_words, review_count

def vanilla_train_perceptron(all_feature_vectors, pos_neg_label_dict, tru_decep_label_dict, remove_words, review_count,  train_files):
    bias1 = 0
    bias2 = 0
    weight1 = defaultdict(int)
    weight2 = defaultdict(int)
    for i in range(75):
        shuffle(train_files)
        for file_name in train_files:
            score1 = bias1
            pos_neg_label = pos_neg_label_dict[file_name]
            feature_vector = all_feature_vectors[file_name]
            for feature in feature_vector.keys():
                if feature not in remove_words:
                    score1 += weight1[feature] * feature_vector[feature]
            if score1 * pos_neg_label <= 0:
                for feature in feature_vector.keys():
                    if feature not in remove_words:
                        weight1[feature] += pos_neg_label * feature_vector[feature]
                bias1 += pos_neg_label

            score2 = bias2
            tru_decep_label = tru_decep_label_dict[file_name]
            #print(feature_vector)
            for feature in feature_vector.keys():
                if feature not in remove_words:
                    score2 += weight2[feature] * feature_vector[feature]
            if score2 * tru_decep_label <= 0:
                for feature in feature_vector.keys():
                    if feature not in remove_words:
                        weight2[feature] += tru_decep_label * feature_vector[feature]
                bias2 += tru_decep_label
    return weight1, bias1, weight2, bias2

def average_train_perceptron(all_feature_vectors, pos_neg_label_dict, tru_decep_label_dict, remove_words, review_count, train_files):
    bias1 = 0
    bias2 = 0
    avg_weight1 = defaultdict(int)
    avg_weight2 = defaultdict(int)
    cached_weights1 = defaultdict(int)
    cached_weights2 = defaultdict(int)
    cached_bias1 = 0
    cached_bias2 = 0
    counter1 = 1
    counter2 = 1
    for i in range(75):
        shuffle(train_files)
        for file_name in  train_files:
            score1 = bias1
            feature_vector = all_feature_vectors[file_name]
            pos_neg_label = pos_neg_label_dict[file_name]
            for feature in feature_vector.keys():
                if feature not in remove_words:
                    score1 += avg_weight1[feature] * feature_vector[feature]
            if score1 * pos_neg_label <= 0:
                for feature in feature_vector.keys():
                    if feature not in remove_words:
                        avg_weight1[feature] += pos_neg_label * feature_vector[feature]
                for feature in feature_vector.keys():
                    if feature not in remove_words:
                        cached_weights1[feature] += pos_neg_label * feature_vector[feature] * counter1
                bias1 += pos_neg_label
                cached_bias1 += pos_neg_label * counter1
            counter1 += 1

            score2 = bias2
            tru_decep_label = tru_decep_label_dict[file_name]
            for feature in feature_vector.keys():
                if feature not in remove_words:
                    score2 += avg_weight2[feature] * feature_vector[feature]
            if score2 * tru_decep_label <= 0:
                for feature in feature_vector.keys():
                    if feature not in remove_words:
                        avg_weight2[feature] += tru_decep_label * feature_vector[feature]
                for feature in feature_vector.keys():
                    if feature not in remove_words:
                        cached_weights2[feature] += tru_decep_label * feature_vector[feature] * counter2
                bias2 += tru_decep_label
                cached_bias2 += tru_decep_label * counter2
            counter2 += 1
    for feature in avg_weight1:
        avg_weight1[feature] = avg_weight1[feature] - (cached_weights1[feature] / counter1)
    bias1 = bias1 - (cached_bias1 / counter1)

    for feature in avg_weight2:
        avg_weight2[feature] = avg_weight2[feature] - (cached_weights2[feature] / counter2)
    bias2 = bias2 - (cached_bias2 / counter2)

    return avg_weight1, bias1, avg_weight2, bias2

def main():
    root_directory = sys.argv[1]
    train_files = get_train_files(root_directory)
    all_feature_vectors, pos_neg_label_dict, tru_decep_label_dict,remove_words, review_count = fit(train_files)
    vanilla_weight1, vanilla_bias1, vanilla_weight2, vanilla_bias2 = vanilla_train_perceptron(all_feature_vectors, pos_neg_label_dict, tru_decep_label_dict, remove_words, review_count, train_files)
    avg_weight1, avg_bias1, avg_weight2, avg_bias2 = average_train_perceptron(all_feature_vectors, pos_neg_label_dict, tru_decep_label_dict, remove_words, review_count, train_files)
    with open("vanillamodel.txt","w") as f:
        f.write("weights_positive_negative,")
        for key,value in vanilla_weight1.items():
            f.write(str(key) + ":" + str(value) + ",")
        f.write("\n")
        f.write("bias_positive_negative,")
        f.write(str(vanilla_bias1))
        f.write("\n")
        f.write("weights_true_deceptive,")
        for key, value in vanilla_weight2.items():
            f.write(str(key) + ":" + str(value) + ",")
        f.write("\n")
        f.write("bias_true_deceptive,")
        f.write(str(vanilla_bias2))

    with open("averagedmodel.txt","w") as f:
        f.write("weights_positive_negative,")
        for key,value in avg_weight1.items():
            f.write(str(key) + ":" + str(value) + ",")
        f.write("\n")
        f.write("bias_positive_negative,")
        f.write(str(avg_bias1))
        f.write("\n")
        f.write("weights_true_deceptive,")
        for key, value in avg_weight2.items():
            f.write(str(key) + ":" + str(value) + ",")
        f.write("\n")
        f.write("bias_true_deceptive,")
        f.write(str(avg_bias2))

if __name__ == "__main__":
    ignore_files_pattern = r"(README|DS_Store|LICENSE)"
    main()