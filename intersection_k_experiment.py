from gensim.models import Word2Vec
import nltk
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from string import punctuation
from nltk.corpus import stopwords
import docx
import itertools
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
import random
import pickle


# nltk.download('stopwords')
# nltk.download('punkt')


def normalize_text(text, stop_words):
    # remove special characters\whitespaces
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)

    # lower case & tokenize text
    tokens = re.split(r'\s+', text.lower().strip())

    # filter stopwords out of text &
    # re-create text from filtered tokens
    cleaned_text = ' '.join(token for token in tokens if token not in stop_words)
    return cleaned_text


def words_counter(sentences, words, window):
    # sentences is a list of lists, words is a list of wanted words and window size of Word2Vec
    text = list(itertools.chain.from_iterable(sentences))
    pairs = list(itertools.combinations(words, 2))
    pairs_dict = dict.fromkeys(pairs, 0)
    limit = len(text) - 1
    for i,word in enumerate(text):
        if word not in words:
            continue
        for j in range(-window, window+1):
            if i+j < 0 or j == 0:
                continue
            if i+j > limit:
                break
            checked_word = text[i+j]
            if checked_word in words and checked_word != word:
                if (word, checked_word) in pairs_dict.keys():
                    pairs_dict[(word, checked_word)] += 1
                elif (checked_word, word) in pairs_dict.keys():
                    pairs_dict[(checked_word, word)] += 1
                else:
                    raise ValueError("Something in logic does'nt make sence")
    pairs_dict = dict(reversed(sorted(pairs_dict.items(), key=lambda x: x[1])))
    return pairs_dict


def book_preprocessing(book, nicknames):
    '''doc = docx.Document(book)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    fullText = ' '.join(word for word in fullText)
    '''
    file = open(book, 'r', encoding='utf-8')
    fullText = file.read()
    file.close()
    STOPWORDS = set(stopwords.words('english')) | set(punctuation) | set(ENGLISH_STOP_WORDS)
    text = normalize_text(fullText,STOPWORDS)
    for key in nicknames.keys():
        text = text.replace(key, nicknames[key])
    snt = nltk.tokenize.sent_tokenize(text)
    sentences = [re.findall(r"[\w']+", l.lower()) for l in snt]
    for sentence in sentences:
        if len(sentence) > 0:
            current_word = sentence[0]
            for i in range(1, len(sentence)):
                if sentence[i] == current_word:
                    sentence[i] = 0
                else:
                    current_word = sentence[i]
            while 0 in sentence:
                sentence.remove(0)
    return sentences


def intersection_of_lists(model_list, counter_list, k=-1):
    intersection_num = 0
    if k != -1:
        model_list = model_list[:k]
        counter_list = counter_list[:k]
    union = counter_list.copy()
    for elem in model_list:
        if elem in counter_list or (elem[1], elem[0]) in counter_list:
            intersection_num += 1
        else:
            union.append(elem)
    return intersection_num/len(union)*100


def intersection(k, model_dict, counter_dict):
    # returns percentage number of intersections out of k pairs (k must be smaller than number of pairs)
    dict_len = len(model_dict)
    assert k <= dict_len
    if k == dict_len:
        return k
    model_list = list(model_dict.keys())
    last_value = model_dict[model_list[k - 1]]
    k_tag = k - 1
    for i in range(k, dict_len):
        if last_value == model_dict[model_list[i]]:
            k_tag = i
        else:
            break
    model_list = model_list[:k_tag + 1]
    counter_list = list(counter_dict.keys())
    last_value = counter_dict[counter_list[k-1]]
    k_tag = k-1
    for i in range(k, dict_len):
        if last_value == counter_dict[counter_list[i]]:
            k_tag = i
        else:
            break
    counter_list = counter_list[:k_tag+1]
    return intersection_of_lists(model_list, counter_list)


def baseline_intersection(k, pairs, models_per_window_num):
    baseline = 0
    for i in range(models_per_window_num):
        perm1 = random.sample(pairs, len(pairs))
        perm2 = random.sample(pairs, len(pairs))
        baseline += intersection_of_lists(perm1, perm2, k=k)
    return baseline/models_per_window_num


def intersection_graph_experiment1(samples_per_window, list_of_k, list_of_windows, list_of_baselines, list_of_colors, title):
    for j, k in enumerate(list_of_k):
        plt.plot(list_of_windows, samples_per_window[j], '-ok', label=f"k = {k} {title}", color=list_of_colors[j])
        baseline_yaxis = np.full(len(list_of_windows), list_of_baselines[j])
        plt.plot(list_of_windows, baseline_yaxis, '*', label=f"baseline for k = {k} {title}", color=list_of_colors[j])
    plt.xlabel("window size")
    plt.ylabel("intersection (%)")
    plt.legend()
    plt.title("intersection as function of window size with different values of k")
#   plt.savefig("intersection_graph_experiment1.png")
#   plt.show()
#   plt.plot()


def intersection_graph_experiment2(samples_per_book, windows, books):
    x = range(1, len(books)+1)
    for j, window in enumerate(windows):
        plt.xticks(x, books)
        plt.plot(x, samples_per_book[j], label=f"window = {window}")
    plt.xlabel("book name")
    plt.ylabel("intersection (%)")
    plt.legend()
    plt.title("intersection in different parts with k = 10")
#   plt.savefig("intersection_graph_experiment2.png")
    plt.show()
    plt.plot()


def extract_manual(manual_names):
    # gets excel name and sheet name and returns sorted dictionary
    wb = pd.read_excel(manual_names[0], sheet_name=manual_names[1], engine="openpyxl")
    manual_dict = dict()
    number_of_lines = len(wb)
    for i in range(number_of_lines):
        line = wb.iloc[i]
        key = (line[0], line[1])
        value = line[2]
        manual_dict[key] = value
    manual_dict = dict(reversed(sorted(manual_dict.items(), key=lambda x: x[1])))
    return manual_dict


def experiment_one_book(list_of_k, windows, models_per_window_num, list_of_colors, title, manual_names=False):
    time_point = time.time()
    book = "my_michael.txt"
    nicknames = {"gonen": "michael", "michael gonen": "michael", "kamnitzer": "yoram",
                 "yair gonen": "yair", "greenbaum": "hannah", "hannah greenbaum - gonen": "hannah",
                 "hannah gonen": "hannah", "duba": "glick", "yehezkel gonen": "yehezkel"}
    sentences = book_preprocessing(book, nicknames)
    words = ["michael", "yair", "hannah", "yoram", "glick", "kadishman", "yehezkel",
             "hadassah", "emanuel", "yardena"]
    pairs = list(itertools.combinations(words, 2))
    mincount = 5
    samples_per_window = np.zeros((len(list_of_k), len(windows))).tolist()
    if manual_names:
        checked_dict = extract_manual(manual_names)
        # returns sorted dict
    for w_index, window in enumerate(windows):
        if not manual_names:
            checked_dict = words_counter(sentences, words, window)
        intersection_sum_per_k = np.zeros(len(list_of_k))
        for i in range(models_per_window_num):
            model = Word2Vec(sentences, min_count=mincount, window=window, iter=30, size=20, sg=1, hs=1)
            model_dict = dict.fromkeys(pairs, 0)
            for pair in pairs:
                model_dict[pair] = model.wv.similarity(pair[0], pair[1])
            model_dict = dict(reversed(sorted(model_dict.items(), key=lambda x: x[1])))
            for j, k in enumerate(list_of_k):
                intersection_sum_per_k[j] += intersection(k, model_dict, checked_dict)
            print(f"window {window} iter {i}")
        intersection_avg_per_k = intersection_sum_per_k/models_per_window_num
        for j in range(len(list_of_k)):
            samples_per_window[j][w_index] = intersection_avg_per_k[j]
    list_of_baselines = []
    for k in list_of_k:
        list_of_baselines.append(baseline_intersection(k, pairs, models_per_window_num))
    time_point = (time.time() - time_point) / 60
    print(f"This simulation took {time_point} minutes")
    intersection_graph_experiment1(samples_per_window, list_of_k, windows, list_of_baselines, list_of_colors, title)


def experiment_one_book_counter_vs_manual(list_of_k, windows, list_of_colors, title, manual_names):
    time_point = time.time()
    words = ["yonatan", "rimona", "yolek", "hava", "azariah",
             "srulik", "anat", "benya", "bolognesi", "eshkol"]
    names_counters = ["pairs_dict3.dict", "pairs_dict5.dict", "pairs_dict10.dict", "pairs_dict15.dict",
                      "pairs_dict20.dict", "pairs_dict25.dict", "pairs_dict30.dict", "pairs_dict40.dict"]
    pairs = list(itertools.combinations(words, 2))
    samples_per_window = np.zeros((len(list_of_k), len(windows))).tolist()
    checked_dict = extract_manual(manual_names)
    # returns sorted dict
    for w_index, window in enumerate(windows):
        with open(names_counters[w_index], "rb") as file:
            counter_dict = pickle.load(file)
        for j, k in enumerate(list_of_k):
            samples_per_window[j][w_index] += intersection(k, counter_dict, checked_dict)
        print(f"window {window}")
    list_of_baselines = []
    for k in list_of_k:
        list_of_baselines.append(baseline_intersection(k, pairs, 20))
    time_point = (time.time() - time_point) / 60
    print(f"This simulation took {time_point} minutes")
    intersection_graph_experiment1(samples_per_window, list_of_k, windows, list_of_baselines, list_of_colors, title)


def experiment_parts(k, windows, models_per_window_num, manual_names=False):
    time_point = time.time()
    books = ["A Perfect Peace - complete,Clean1.docx",
             "A Perfect Peace - complete,Clean2.docx",
             "A Perfect Peace - complete,Clean.docx"]
    names = ["Part 1", "Part 2", "Full"]
    nicknames = {"yoni": "yonatan", "gitlin": "azariah", "zaro": "azariah", "trotsky": "benya", "bini": "benya",
                 "yisra": "yolek"}
    words = ["yonatan", "rimona", "yolek", "hava", "azariah",
             "srulik", "anat", "benya", "bolognesi", "eshkol"]
    pairs = list(itertools.combinations(words, 2))
    mincount = 5
    samples_per_book = np.zeros((len(windows), len(books))).tolist()
    for book_index, book in enumerate(books):
        sentences = book_preprocessing(book, nicknames)
        sample_for_one_book = np.zeros(len(windows))
        if manual_names:
            checked_dict = extract_manual([manual_names[0], manual_names[book_index+1]])
        for w_index, window in enumerate(windows):
            if not manual_names:
                checked_dict = words_counter(sentences, words, window)
            for i in range(models_per_window_num):
                model = Word2Vec(sentences, min_count=mincount, window=window, iter=30, size=20, sg=1, hs=1)
                model_dict = dict.fromkeys(pairs, 0)
                for pair in pairs:
                    model_dict[pair] = model.wv.similarity(pair[0], pair[1])
                model_dict = dict(reversed(sorted(model_dict.items(), key=lambda x: x[1])))
                sample_for_one_book[w_index] += intersection(k, model_dict, checked_dict)
                print(f"book {names[book_index]} window {window} iter {i}")
            sample_for_one_book[w_index] /= models_per_window_num
            samples_per_book[w_index][book_index] = sample_for_one_book[w_index]
    time_point = (time.time() - time_point) / 60
    print(f"This simulation took {time_point} minutes")
    intersection_graph_experiment2(samples_per_book, windows, names)


windows = [3, 5, 10, 15, 20, 25, 30, 40]
list_of_k = [5, 10, 15]
manual_names = ["sorted_edges_manual.xlsx", "part 1", "part 2", "all novel"]

colors1 = ['red', 'green', 'blue']
colors2 = ['pink', 'black', 'yellow']
colors3 = ['orange', 'purple', 'brown']

titles = ['model vs counting', 'model vs manual', 'counting vs manual']

experiment_one_book(list_of_k, windows, 40, colors1, titles[0])
# experiment_one_book(list_of_k, windows, 20, colors2, titles[1], manual_names=manual_names[::3])
# experiment_one_book_counter_vs_manual(list_of_k, windows, colors3, titles[2], manual_names[::3])
plt.show()

