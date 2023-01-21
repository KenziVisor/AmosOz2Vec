from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import nltk
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from string import punctuation
from nltk.corpus import stopwords
import itertools
import pickle
import os
import subprocess
import shutil
import codecs
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
import networkx as nx
import statistics
from xlwt import Workbook
import random
import ntpath
from pathlib import Path
import kneed


def normalize_text(text, stop_words):
    # remove special characters\whitespaces
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I | re.A)

    # lower case & tokenize text
    tokens = re.split(r'\s+', text.lower().strip())

    # filter stopwords out of text &
    # re-create text from filtered tokens
    cleaned_text = ' '.join(token for token in tokens if token not in stop_words)
    return cleaned_text


def words_counter(sentences, words, window, filename):
    '''
    :param sentences: Text as list of lists
    :param words: Relevant words to count in text
    :param window: The window size used to count an appearance of a pair
    :param filename: Name of the output file
    :return: Prints to file the dictionary of pair appearances in a window
    '''
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
    with open(filename, "wb") as file:
        pickle.dump(pairs_dict, file)


def find_bookNLP(searched_file):
    '''
    :param searched_file: Name of a file in directory
    :return: The path for the file's directory
    '''
    home = str(Path.home())
    for root, dirs, files in os.walk(home):
        for name in files:
            if name == searched_file:
                dir = os.path.abspath(os.path.join(root, name)).replace(searched_file, '')
                return dir


def number_of_characters_decision(text):
    y = []
    for seq in text:
        if seq.isnumeric():
            y.append(int(seq))
    x = range(len(y))
    knee = kneed.KneeLocator(x, y, S=3.0, curve="convex", direction="decreasing").knee + 1
    while y[knee] < 5 and knee >= 0:
        knee -= 1
    return knee


def bookNLP_preprocessing(book_path_origin, number_of_characters, is_gui):
    '''
    :param book_path_origin: Path for a book .txt file
    :param number_of_characters: Number of extracted characters from BookNLP output
    :param is_gui: Decision to pick the number of characters or to find it
    :return: Returns the characters list and a dictionary of nicknames as key and name as value
    '''
    regen = re.compile('[^a-zA-z0-9_]')
    book = regen.sub('', ntpath.basename(book_path_origin).replace('.txt', '').replace(' ', '_'))
    cd_path = find_bookNLP("book-nlp.jar")
    book_path_dest = cd_path + f'data/originalTexts/{book}.txt'
    try:
        shutil.copyfile(book_path_origin, book_path_dest)
    except shutil.SameFileError:
        os.remove(book_path_dest)
        shutil.copyfile(book_path_origin, book_path_dest)
    book_path_dest = f'data/originalTexts/{book}.txt'
    command = f'bash runjava novels/BookNLP -doc {book_path_dest} -printHTML -p data/output/{book} -tok data/tokens/{book}.tokens -f'
    subprocess.call(f'cd {cd_path} & {command}', shell=True)
    file = codecs.open(cd_path + f'data/output/{book}/book.id.html', 'r', 'utf-8')
    text = BeautifulSoup(file.readline(), features="lxml").get_text().split()
    text[0] = text[0].replace("Characters", "")
    file.close()
    words = []
    is_name = True
    word = ''
    current_name = ''
    nicknames = dict()
    characters_decision = number_of_characters
    if not is_gui:
        characters_decision = number_of_characters_decision(text)
    text = text[1:]
    for seq in text:
        if len(words) == characters_decision:
            break
        checked_letter = seq[0]
        if 'a' <= checked_letter <= 'z' or 'A' <= checked_letter <= 'Z':
            if word == '':
                word = seq.lower()
            else:
                word = word + ' ' + seq.lower()
            continue
        if checked_letter == '(':
            if is_name:
                is_name = False
                temp = word
                temp = temp.replace(' ', '_')
                temp = temp.replace('.', '')
                nicknames[word] = temp
                current_name = temp
            else:
                nicknames[word] = current_name
            word = ''
            continue
        if '1' <= checked_letter <= '9':
            is_name = True
            if current_name not in words:
                words.append(current_name)
    nicknames = dict(reversed(sorted(nicknames.items(), key=lambda x: len(x[0]))))
    os.remove(cd_path + f'data/originalTexts/{book}.txt')
    return words, nicknames


def book_preprocessing(book_path_origin, number_of_characters, is_gui):
    '''
    :param book_path_origin: Path for a book .txt file
    :param number_of_characters: Number of extracted characters from BookNLP output
    :param is_gui: Decision to pick the number of characters or to find it
    :return: Makes a pre-processing to the book, including BookNLP usage and nickname replacements and returns the
    parsed text as sentences and the characters names extracted from BookNLP.
    '''

    '''doc = docx.Document(book)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    fullText = ' '.join(word for word in fullText)
    '''

    words, nicknames = bookNLP_preprocessing(book_path_origin, number_of_characters, is_gui)
    file = open(book_path_origin, 'r', encoding='utf-8')
    fullText = file.read()
    file.close()
    STOPWORDS = set(stopwords.words('english')) | set(punctuation) | set(ENGLISH_STOP_WORDS)
    text = normalize_text(fullText, STOPWORDS)
    for sign in list(punctuation):
        text = text.replace(sign, ' ' + sign + ' ')
    for key in nicknames.keys():
        text = text.replace(' ' + key + ' ', ' ' + nicknames[key] + ' ')

    # start gutenberg fix
    if "project_gutenberg_literary_archive_foundation" in words:
        words.remove("project_gutenberg_literary_archive_foundation")
        text = text.replace("project_gutenberg_literary_archive_foundation", "")
    #end gutenberg fix
    # start yoni fix
    '''
    if "yoni" in words:
        words.remove("yoni")
        text = text.replace("yoni", "yonatan")
    '''
    # end yoni fix

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
    return sentences, words


def AmosOz2Vec(name, book_path_origin, number_of_characters, window, is_gui=True):
    '''
    :param name: The general name for the model, counter and characters saves
    :param book_path_origin: The path where the book is in
    :param number_of_characters: Number of extracted characters from BookNLP output
    :param window: Window size for Word2Vec
    :param is_gui: Decision to pick the number of characters or to find it
    :return: Saves in files a Word2Vec model, a dictionary counter and characters names
    '''
    sentences, words = book_preprocessing(book_path_origin, number_of_characters, is_gui)
    words_counter(sentences, words, window, f"{name}.dict")
    model = Word2Vec(sentences, min_count=3, window=window, iter=30, size=20, sg=1, hs=1)
    model.save(f"{name}.wv")
    with open(f'{name}.words', "wb") as file:
        pickle.dump(words, file)


def rank_dictionary(counter_dict):
    '''
    :param counter_dict: Sorted dictionary to rank
    :return: A dictionary with same keys as input, but the value is the relative rank of the key
    '''
    keys = list(counter_dict.keys())
    ranked_dict = dict.fromkeys(keys, 0)
    curr_rank = 0
    how_many_in_rank = 0
    curr_value = -1
    for key in keys:
        value = counter_dict[key]
        if curr_value != value:
            rank = curr_rank + 1 + how_many_in_rank
            how_many_in_rank = 0
            curr_value = value
            curr_rank = rank
        else:
            rank = curr_rank
            how_many_in_rank += 1
        ranked_dict[key] = rank
    return ranked_dict


def position_diff(dict_models, dict_counters, avg_models, avg_counters, sheet1, step, residual, with_manual_no_parts=False):
    '''
    :param dict_models: List of dictionaries of Word2vec models / counters
    :param dict_counters: List of dictionaries of counters / manuals
    :param avg_models: List of avg of dictionary values of Word2vec models / counters
    :param avg_counters: List of avg of dictionary values of counters / manuals
    :param sheet1: Representation of the excel file
    :param step: The prints of the excel jump in step per list value of dict_models
    :param residual: The prints of the excel adds residual to create space in excel
    :param with_manual_no_parts: To compare parts of a book, FOLLOW THE INSTRUCTIONS OF THE FLAGS.
    :return: Prints to excel the differential of keys in model/counter compared with counter/manual
    '''
    counter_dict = dict()
    ranked_counter = dict()
    if with_manual_no_parts:
        counter_dict = dict_counters
        ranked_counter = rank_dictionary(counter_dict)
    for i in range(len(dict_models)):
        model_dict = dict_models[i]
        ranked_model = rank_dictionary(model_dict)
        if not with_manual_no_parts:
            counter_dict = dict_counters[i]
            ranked_counter = rank_dictionary(counter_dict)
        keys = list(ranked_model.keys())
        location_col = step * i + residual + 3
        diff_sum = 0
        diff_list = []
        sheet1.write(0, location_col, "diff factor")
        for j, key in enumerate(keys):
            rank_of_key_model = ranked_model[key]
            if key in ranked_counter.keys():
                rank_of_key_counter = ranked_counter[key]
            elif (key[1], key[0]) in ranked_counter.keys():
                rank_of_key_counter = ranked_counter[(key[1], key[0])]
            else:
                raise ValueError(f"Something wrong with the keys on position diff {key}")
            diff = rank_of_key_model-rank_of_key_counter
            if counter_dict[key] >= avg_counters[i] or model_dict[key] >= avg_models[i]:
                diff_list.append(diff)
            diff_sum += diff
            sheet1.write(j+1, location_col, str(diff))


def position_diff_graph_theory(model_dict, counter_dict):
    '''
    :param model_dict: A dictionary of Word2Vec model / counter
    :param counter_dict: A dictionary of counter / manual
    :return: A list of differential of keys in model/counter compared to counter/manual
    '''
    ranked_model = rank_dictionary(model_dict)
    ranked_counter = rank_dictionary(counter_dict)
    keys = list(ranked_model.keys())
    diff_list = []
    for j, key in enumerate(keys):
        rank_of_key_model = ranked_model[key]
        if key in ranked_counter.keys():
            rank_of_key_counter = ranked_counter[key]
        elif (key[1], key[0]) in ranked_counter.keys():
            rank_of_key_counter = ranked_counter[(key[1], key[0])]
        else:
            raise ValueError(f"Something wrong with the keys on position diff {key}")
        diff = rank_of_key_model-rank_of_key_counter
        diff_list.append(diff)
    return diff_list


def draw_graph(G, title, counter_dict):
    '''
    :param G: Graph G
    :param title: Title for plot
    :param counter_dict: Compared dictionary for heatmap
    :return: Draw a heatmap of characters: Red color represents low appearence in counter/manual but
    high apperance in model/counter and blue the opposite.
    '''
    pos = nx.spring_layout(G)
    lables = {e: str(G.get_edge_data(*e)["weight"]) + f"/{counter_dict[e]}" for e in G.edges()}
    ax = plt.gca()
    ax.set_title(title)
    model_dict = nx.get_edge_attributes(G, 'weight')
    model_dict = dict(reversed(sorted(model_dict.items(), key=lambda x: x[1])))
    diff_list = position_diff_graph_theory(model_dict, counter_dict)
    for i, pair in enumerate(model_dict.keys()):
        G[pair[0]][pair[1]]['diff'] = -diff_list[i]
    nx.draw(G, pos, with_labels=True, ax=ax, node_color='yellow')
    edges_plot = nx.draw_networkx_edges(G, pos, edge_color=nx.get_edge_attributes(G, 'diff').values(),
                                        width=3, edge_cmap=plt.cm.get_cmap('rainbow'))
    nx.draw_networkx_edge_labels(G, pos, edge_labels=lables)
    cbar = plt.colorbar(edges_plot, shrink=0.5)
    cbar.ax.set_xlabel("\nRed - semantic relationship\nBlue - co-occurrence relationship")
    plt.show()
    plt.plot()


def create_graph(file_name, words_file, dict_name):
    with open(words_file, "rb") as file:
        words_k = pickle.load(file)
    model = Word2Vec.load(file_name)
    weights = []
    bad_words = []
    G = nx.Graph()
    for word in words_k:
        if word in model.wv.vocab:
            G.add_node(word)
        else:
            bad_words.append(word)
    if bad_words:
        for word in bad_words:
            words_k.remove(word)
        os.remove(words_file)
        with open(words_file, "wb") as file:
            pickle.dump(words_k, file)
    '''if len(words_k) <= 5:
        print(words_file)
        print("number of characters: ", len(words_k))'''
    pairs = list(itertools.combinations(words_k, 2))
    for pair in pairs:
        weights.append(round(model.wv.similarity(pair[0], pair[1]), 2))
    avg_wv = round(float(statistics.mean(weights)), 2)  # average weight between characters
    with open(dict_name, "rb") as file:
        counter_dict = pickle.load(file)
    avg_counter = round(float(statistics.mean(counter_dict.values())), 2)
    for i, pair in enumerate(pairs):
        count_pair = 0
        if pair in counter_dict.keys():
            count_pair = counter_dict[pair]
        else:
            count_pair = counter_dict[(pair[1], pair[0])]
        if weights[i] >= avg_wv or count_pair >= avg_counter:
            G.add_edge(pair[0], pair[1], weight=weights[i])
    return G, avg_wv, avg_counter


def graph_theory(names):
    '''
    :param names: Names for pre-saved models
    :return: Extracts the saved files, builds the graphs from Word2Vec and draws a heatmap of characters:
    Red color represents low appearence in counter/manual but high apperance in model/counter and blue the opposite.
    '''
    file_names = [name+'.wv' for name in names]
    dict_names = [name+'.dict' for name in names]
    words = [name+'.words' for name in names]
    for k, file_name in enumerate(file_names):
        G, avg_wv, avg_counter = create_graph(file_name, words[k], dict_names[k])
        with open(dict_names[k], "rb") as file:
            counter_dict = pickle.load(file)
        counter_dict = dict(reversed(sorted(counter_dict.items(), key=lambda x: x[1])))
        title = f"{file_name}, vector size = 20, model = skip-gram, loss function = softmax, avg w2v weight = {avg_wv}, avg counter = {avg_counter}"
        draw_graph(G, title, counter_dict)


def graph_theory_closest(names, connection):
    '''
    :param names: Names for pre-saved models
    :param connection: Number of closest words to be seen on screen
    :return: Draw a graph of n-closest words of a Word2Vec model
    '''
    file_names = [name + '.wv' for name in names]
    words = [name + '.words' for name in names]
    for k, file_name in enumerate(file_names):
        with open(words[k], "rb") as file:
            words_k = pickle.load(file)
        model = Word2Vec.load(file_name)
        title = f"{file_name} closest {connection} graph"
        G = nx.Graph()
        for word in words_k:
            G.add_node(word)
            # connection + number of characters not including 'word'
            closest = model.wv.most_similar(word, topn=connection + len(words_k) - 1)
            size = 0
            for close in closest:
                name = close[0]
                if name not in words_k:
                    if name not in G.nodes:
                        G.add_node(name)
                    weight = round(close[1], 2)
                    G.add_edge(word, name, weight=weight)
                    size += 1
                    if size == connection:
                        break
        color_map = []
        for node in G.nodes:
            if node in words_k:
                color_map.append("yellow")
            else:
                color_map.append("white")
        pos = nx.spring_layout(G)
        ax = plt.gca()
        ax.set_title(title)
        nx.draw(G, pos, with_labels=True, ax=ax, node_color=color_map)
        plt.show()


def write_to_excel(dictionary, k, title, sheet1):
    '''
    :param dictionary: A dictionary of pairs and weight
    :param k: The location of the column to start the print
    :param title: The title of the col
    :param sheet1: Representation of the excel file.
    :return: Prints to excel a dictionary and returns the avg and the sorted dictionary
    '''
    # gets a non sorted dictionary, a position and excel sheet and prints weights to excel
    weighted_dict = dict(reversed(sorted(dictionary.items(), key=lambda x: x[1])))
    avg = round(statistics.mean(weighted_dict.values()), 2)  # average weight between characters
    std = round(statistics.stdev(weighted_dict.values(), xbar=avg), 2)
    sheet1.write(0, k, title)
    for j,pair in enumerate(weighted_dict.keys()):
        weighted_dict[pair] = round(weighted_dict[pair], 2)
        sheet1.write(j+1, k, pair[0])
        sheet1.write(j+1, k+1, pair[1])
        sheet1.write(j+1, k+2, str(weighted_dict[pair]))
    dict_len = len(weighted_dict)
    sheet1.write(dict_len+1, k, "avg characters")
    sheet1.write(dict_len+1, k+1, str(avg))
    sheet1.write(dict_len+2, k, "std characters")
    sheet1.write(dict_len+2, k+1, str(std))
    return avg, weighted_dict


def statistics_from_model(model, num):
    '''
    :param model: A Word2Vec model
    :param num: Number of random iterations
    :return: Returns the avg and std of num-random pairs similarity in Word2Vec
    '''
    vocab = list(model.wv.vocab.keys())
    samples_array = np.zeros(num)
    for i in range(num):
        sample = random.sample(vocab, 2)
        samples_array[i] += model.wv.similarity(sample[0], sample[1])
    avg = round(np.mean(samples_array), 2)
    std = round(np.std(samples_array, ddof=1), 2)
    return avg, std


def unusual_behavior(dict_models, dict_counters, avg_models, avg_counters, words, sheet1, step, residual, with_manual_no_parts=False):
    '''
    :param dict_models: List of dictionaries of Word2vec models / counters
    :param dict_counters: List of dictionaries of counters / manuals
    :param avg_models: List of avg of dictionary values of Word2vec models / counters
    :param avg_counters: List of avg of dictionary values of counters / manuals
    :param words: List of lists. Each list contains characters names relevant to the proper model
    :param sheet1: Representation of the excel file
    :param step: The prints of the excel jump in step per list value of dict_models
    :param residual: The prints of the excel adds residual to create space in excel
    :param with_manual_no_parts: To compare parts of a book, FOLLOW THE INSTRUCTIONS OF THE FLAGS.
    :return: Prints to the excel pairs that thier model/counter score is low, but thier counter/manual score is high
    and the opposite
    '''
    # check for unusual behavior such that cases below average in model and over average in counter and the opposite.
    if with_manual_no_parts:
        counter_dict = dict_counters
        counter_avg = avg_counters
    for i in range(len(dict_models)):
        model_dict = dict_models[i]
        model_avg = avg_models[i]
        if not with_manual_no_parts:
            counter_dict = dict_counters[i]
            counter_avg = avg_counters[i]
        location_row = len(model_dict) + 5
        location_col = step * i + residual
        sheet1.write(location_row, location_col, "interesting cases")
        location_row += 1
        words_i = words[i]
        pairs = list(itertools.combinations(words_i, 2))
        for pair in pairs:
            pair_relate_model = model_dict[pair]
            if pair in counter_dict.keys():
                pair_relate_counter = counter_dict[pair]
            elif (pair[1], pair[0]) in counter_dict.keys():
                pair_relate_counter = counter_dict[(pair[1], pair[0])]
            else:
                raise ValueError(f"Something wrong with the keys on unusual behavior {pair}")
            if (pair_relate_model >= model_avg and pair_relate_counter < counter_avg) or \
                    (pair_relate_model < model_avg and pair_relate_counter >= counter_avg):
                sheet1.write(location_row, location_col, pair[0])
                sheet1.write(location_row, location_col + 1, pair[1])
                location_row += 1


def models2excel(file_names, words, sheet1, step, residual):
    '''
    :param file_names: names of Word2Vec models
    :param words: List of lists. Each list contains characters names relevant to the proper model
    :param sheet1: Representation of the excel file
    :param step: The prints of the excel jump in step per list value of dict_models
    :param residual: The prints of the excel adds residual to create space in excel
    :return: Load Word2Vec models, print them to excel and return the models list and avg lists of the models.
    '''
    dict_models = []
    avg_models = []
    for k, file_name in enumerate(file_names):
        model = Word2Vec.load(file_name)
        words_k = words[k]
        pairs = list(itertools.combinations(words_k, 2))
        dictionary = dict.fromkeys(pairs, 0)
        for pair in pairs:
            dictionary[pair] = model.wv.similarity(pair[0], pair[1])
        # UPDATE THE FILE NAME WHEN YOU DONT WORK WITH PARTS!!!
        avg_characters, dictionary = write_to_excel(dictionary, step * k + residual, file_name, sheet1)
        avg_models.append(avg_characters)
        dict_len = len(dictionary)
        avg, std = statistics_from_model(model, 100)
        sheet1.write(dict_len + 3, step * k + residual, "avg noise")
        sheet1.write(dict_len + 3, step * k + residual + 1, str(avg))
        sheet1.write(dict_len + 4, step * k + residual, "std noise")
        sheet1.write(dict_len + 4, step * k + residual + 1, str(std))
        dict_models.append(dictionary)
    return dict_models, avg_models


def counters2excel(names_counters, sheet1, step, residual):
    '''
    :param names_counters: Names of counters files
    :param sheet1: Representation of the excel file
    :param step: The prints of the excel jump in step per list value of dict_models
    :param residual: The prints of the excel adds residual to create space in excel
    :return: Load counters, print them to excel and return the models list and avg lists of the models.
    '''
    dict_counters = []
    avg_counters = []
    for i, counter in enumerate(names_counters):
        with open(counter, "rb") as file:
            dictionary = pickle.load(file)
        avg_characters, dictionary = write_to_excel(dictionary, step * i + residual, names_counters[i], sheet1)
        dict_counters.append(dictionary)
        avg_counters.append(avg_characters)
    return dict_counters, avg_counters


def extract_manual(manual_names):
    '''
    :param manual_names: A list of The file name and the sheet name of a manual excel file
    :return: A sorted dictionary with keys of pairs and thier manually configured values
    '''
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


def manual2excel(manual_names_list, sheet1, step, residual, parts=False):
    '''
    :param manual_names: A list of lists. Each list containes the file name and the sheet name of a manual excel file
    :param sheet1: Representation of the excel file
    :param step: The prints of the excel jump in step per list value of dict_models
    :param residual: The prints of the excel adds residual to create space in excel
    :param parts: To compare parts of a book, FOLLOW THE INSTRUCTIONS OF THE FLAGS.
    :return: Load manual excel files, print them to excel and return the models list and avg lists of the models.
    '''
    if not parts:
        for i, manual_names in enumerate(manual_names_list):
            manual_dict = extract_manual(manual_names)
            avg, manual_dict = write_to_excel(manual_dict, step * i + residual, manual_names[0], sheet1)
        return manual_dict, avg
    '''
    dict_manuals = []
    avg_manuals = []
    for i in range(1,3):
        manual_dict = extract_manual([manual_names[0], manual_names[i]])
        avg, manual_dict = write_to_excel(manual_dict, step * (i - 1) + residual, "zibi", sheet1)
        dict_manuals.append(manual_dict)
        avg_manuals.append(avg)
    return dict_manuals, avg_manuals
    '''
    raise ValueError("Parts is not working for now")


def flags_choice(flags_array):
    '''
    :param flags_array: An array of model, counter and manual flags
    :return: A decision of what comparison are we doing.
    1 means Word2Vec and counter. 2 means Word2vec and manual and 3 means counter and manual.
    -1 means that more or less than 2 flags were filled with true.
    '''
    decision_check = 0
    for flag in flags_array:
        if flag:
            decision_check+=1
    if decision_check != 2:
        return -1
    if flags_array[0] == flags_array[1]:
        return 1
    if flags_array[0] == flags_array[2]:
        return 2
    return 3


'''

IMPORTANT!!! INSTRUCTIONS OF THE FLAGS!!!

IF YOU USE MANUAL, PUT HIM ALWAYS AS SECOND VARIABLE!!!

TO MANUAL WITH PARTS:
SEND manual_names TO manual2excel AND SYMBOL PARTS=TRUE
THE TWO REMAINED FUNCTIONS NEED THE SYMBOL with_manual_no_parts=False

TO MANUAL WITHOUT PARTS:
SEND manual_names TO manual2excel AND SYMBOL PARTS=FALSE
THE TWO REMAINED FUNCTIONS NEED THE SYMBOL with_manual_no_parts=TRUE

TO RUN WITHOUT MANUAL:
THE TWO REMAINED FUNCTIONS NEED THE SYMBOL with_manual_no_parts=False

'''


def sorted_edges(names, output_excel_name, model_flag=False, counter_flag=False, manual_flag=False, manual_paths=[]):
    '''
    :param names: Names for pre-saved models
    :param output_excel_name: Name of the saved excel file
    :param model_flag: Marked when comparison to Word2Vec models is made
    :param counter_flag: Marked when comparison to counters is made
    :param manual_flag: Marked when comparison to manuals is made
    :param manual_path: if manual_flag is on, a path to the files should be inserted in a list
    :return: Compares between EXACTLY 2 flags marked with TRUE and prints results to excel
    '''
    decision = flags_choice([model_flag, counter_flag, manual_flag])
    if decision == -1:
        raise ValueError("Enter EXACTLY two options")
    if manual_flag and manual_paths == []:
        raise ValueError("Enter manual path")
    wb = Workbook()
    sheet1 = wb.add_sheet(output_excel_name)
    words_names = [name + '.words' for name in names]
    words = []
    for word_name in words_names:
        with open(word_name, "rb") as file:
            words.append(pickle.load(file))
    if decision == 1:
        file_names = [name + '.wv' for name in names]
        dict_names = [name + '.dict' for name in names]
        dict_models, avg_models = models2excel(file_names, words, sheet1, step=8, residual=0)
        dict_counters, avg_counters = counters2excel(dict_names, sheet1, step=8, residual=4)
        unusual_behavior(dict_models, dict_counters, avg_models, avg_counters, words, sheet1, step=8, residual=0)
        position_diff(dict_models, dict_counters, avg_models, avg_counters, sheet1, step=8, residual=0)
    elif decision == 2:
        pass
    else:
        pass
    wb.save(output_excel_name + '.xls')


'''
name = "A_Perfect_Peace_complete_Clean_25"
book_path_origin = "C:/Users/kobik/Desktop/הנדסת מערכות תקשורת/שנה ד/פרויקט/code_corpus/A_Perfect_Peace_complete_Clean.txt"
number_of_characters = 10
window = 25
# AmosOz2Vec(name, book_path_origin, number_of_characters, window)
# print("AmosOz2Vec completed")
names = ["A_Perfect_Peace_complete_Clean_25", "A_Perfect_Peace_complete_Clean_15"]
connection = 6
# graph_theory(names)
# graph_theory_closest(names, connection)
sorted_edges(names, "kenzi", model_flag=True, counter_flag=True)
print("sorted edges completed")
'''
