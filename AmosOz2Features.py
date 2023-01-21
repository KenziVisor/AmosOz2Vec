import networkx as nx
import pickle
import statistics
from matplotlib import pyplot as plt
from xlwt import Workbook
import os
import pandas as pd
from AmosOz2Function import create_graph
from AmosOz2Function import position_diff_graph_theory
from AmosOz2Function import AmosOz2Vec
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers


def from_graph_to_model_dict(G):
    model_dict = nx.get_edge_attributes(G, 'weight')
    model_dict = dict(reversed(sorted(model_dict.items(), key=lambda x: x[1])))
    return model_dict


def draw_graph(G, title):
    pos = nx.spring_layout(G)
    ax = plt.gca()
    ax.set_title(title)
    nx.draw(G, pos, with_labels=True, ax=ax)
    plt.show()
    plt.plot()


def decide_color(names):
    file_names = [name + '.wv' for name in names]
    dict_names = [name + '.dict' for name in names]
    words = [name + '.words' for name in names]
    colors_dict_list = []
    avg_diff_list = []
    std_diff_list = []
    max_diff_list = []
    min_diff_list = []
    for k, file_name in enumerate(file_names):
        G, avg_all_weights, avg_count = create_graph(file_name, words[k], dict_names[k])
        model_dict = from_graph_to_model_dict(G)
        with open(dict_names[k], "rb") as file:
            counter_dict = pickle.load(file)
        counter_dict = dict(reversed(sorted(counter_dict.items(), key=lambda x: x[1])))
        diff = position_diff_graph_theory(model_dict, counter_dict)
        diff = [-item for item in diff]
        avg_diff = statistics.mean(diff)
        avg_diff_list.append(avg_diff)
        std_diff = 0
        if len(diff) > 1:
            std_diff = statistics.stdev(diff, xbar=avg_diff)
        std_diff_list.append(std_diff)
        max_diff_list.append(max(diff))
        min_diff_list.append(min(diff))
        red_threshold = avg_diff + std_diff
        blue_threshold = avg_diff - std_diff
        colors_dict = {"red": [], "blue": [], "neutral": []}
        for i, pair in enumerate(model_dict.keys()):
            weight_pair = diff[i]
            if weight_pair >= red_threshold:
                colors_dict["red"].append(pair)
            elif weight_pair <= blue_threshold:
                colors_dict["blue"].append(pair)
            else:
                colors_dict["neutral"].append(pair)
        colors_dict_list.append(colors_dict)
    return colors_dict_list, avg_diff_list, std_diff_list, max_diff_list, min_diff_list


def semantic_split(names):
    file_names = [name + '.wv' for name in names]
    dict_names = [name + '.dict' for name in names]
    words = [name + '.words' for name in names]
    semantic_split_list = []
    for k, file_name in enumerate(file_names):
        G, avg_all_weights, avg_count = create_graph(file_name, words[k], dict_names[k])
        model_dict = from_graph_to_model_dict(G)
        with open(dict_names[k], "rb") as file:
            counter_dict = pickle.load(file)
        counter_dict = dict(reversed(sorted(counter_dict.items(), key=lambda x: x[1])))
        diff = position_diff_graph_theory(model_dict, counter_dict)
        diff = [-item for item in diff]
        semantic_split = {"semantic": [], "co-occurrence": []}
        for i, pair in enumerate(model_dict.keys()):
            weight_pair = diff[i]
            if weight_pair >= 0:
                semantic_split["semantic"].append(pair)
            if weight_pair <= 0:
                semantic_split["co-occurrence"].append(pair)
        semantic_split_list.append(semantic_split)
    return semantic_split_list


def create_color_graphs(G, color_dict):
    red_graph = G.edge_subgraph(color_dict["red"])
    blue_graph = G.edge_subgraph(color_dict["blue"])
    neutral_graph = G.edge_subgraph(color_dict["neutral"])
    return red_graph, blue_graph, neutral_graph


def create_semantic_graphs(G, semantic_split):
    semantic_graph = G.edge_subgraph(semantic_split["semantic"])
    co_graph = G.edge_subgraph(semantic_split["co-occurrence"])
    return semantic_graph, co_graph


def extract_features_from_graph(G):
    keys = ["largest_cc", "num_cc", "diam_largest_cc", "num_cycles", "avg_deg"]
    features = dict.fromkeys(keys)
    if G.nodes:
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        G0 = G.subgraph(Gcc[0])
        features["largest_cc"] = len(G0.nodes)
        features["num_cc"] = nx.number_connected_components(G)
        features["diam_largest_cc"] = nx.diameter(G0)
        features["num_cycles"] = len(nx.cycle_basis(G))
        features["avg_deg"] = 2*len(G.edges)/len(G.nodes)
    else:
        features["largest_cc"] = 0
        features["num_cc"] = 0
        features["diam_largest_cc"] = 0
        features["num_cycles"] = 0
        features["avg_deg"] = 0
    return features


def extract_features(dict_name, G, color_dict, avg_diff, std_diff, max_diff, min_diff, semantic_split_i):
    '''
    :param dict_name:
    :param G:
    :param color_dict:
    :param avg_diff:
    :param std_diff:
    :param semantic_split_i:
    :return: returns features list. The order of those features are:
    average and std of Word2Vec score in G, maximum Word2Vec score, average and std of differential positions,
    maximum and minimum of differential possitions, avg std max min of count co-occurrence,
    extract_features_from_graph of G, red, blue, neutral, semantic and co-occurrence graphs
    and the ratio between red and blue edges (0 red or blue edges is ratio 0).
    '''
    graphs = [0, 0, 0, 0, 0, 0]
    graphs[0] = G
    graphs[1], graphs[2], graphs[3] = create_color_graphs(G, color_dict)
    graphs[4], graphs[5] = create_semantic_graphs(G, semantic_split_i)
    weights = nx.get_edge_attributes(G, 'weight').values()
    avg = statistics.mean(weights)
    std = 0
    if len(weights) > 1:
        std = statistics.stdev(weights, xbar=avg)
    max_w2v = max(weights)
    features = [avg, std, max_w2v, avg_diff, std_diff, max_diff, min_diff]
    with open(dict_name, "rb") as file:
        counter_dict = pickle.load(file)
    counts = counter_dict.values()
    avg_counts = statistics.mean(counts)
    features.append(avg_counts)
    std_counts = 0
    if len(counts) > 1:
        std_counts = statistics.stdev(counts, xbar=avg_counts)
    features.append(std_counts)
    features.append(max(counts))
    features.append(min(counts))
    for iter_graph in graphs:
        features += extract_features_from_graph(iter_graph).values()
    red_len = len(graphs[1].edges)
    blue_len = len(graphs[2].edges)
    ratio = 0
    if not (red_len == 0 or blue_len == 0):
        ratio = red_len/blue_len
    features.append(ratio)
    return features


def extract_names_from_excel(excel_name):
    df = pd.read_csv(excel_name)
    names = list(df.iloc[:, 0])
    for i in range(len(names)):
        names[i] = str(names[i])
    return names


def AmosOz2Features_name(name, i, window, failed_books, error_file_name, sheet1, from_api, book_path_origin=False):
    try:
        name = name + str(window)
        if from_api:
            text = strip_headers(load_etext(int(name))).strip()
            if os.path.exists(f"temp_book{window}.txt"):
                os.remove(f"temp_book{window}.txt")
            with open(f"temp_book{window}.txt", 'w', encoding='utf-8') as file:
                file.write(text)
            AmosOz2Vec(name, f"temp_book{window}.txt", 10, window,
                       is_gui=False)  # number of characters does not matter because is_gui flag is off
            os.remove(f"temp_book{window}.txt")
        else:
            AmosOz2Vec(name, book_path_origin, 10, window,
                       is_gui=False)  # number of characters does not matter because is_gui flag is off
        file_name = name + '.wv'
        dict_name = name + '.dict'
        words_name = name + '.words'
        colors_dict_list, avg_diff_list, std_diff_list, max_diff_list, min_diff_list = decide_color([name])
        semantic_split_list = semantic_split([name])
        G = create_graph(file_name, words_name, dict_name)[0]
        features = extract_features(dict_name, G, colors_dict_list[0], avg_diff_list[0],
                                    std_diff_list[0],
                                    max_diff_list[0], min_diff_list[0], semantic_split_list[0])
        sheet1.write(i - failed_books + 1, 0, name[:-len(str(window))])
        for j in range(len(features)):
            sheet1.write(i - failed_books + 1, j + 1, float(features[j]))
        os.remove(file_name)
        os.remove(dict_name)
        os.remove(words_name)
        return -1
    except Exception as e:
        print("Error in book " + name)
        print(e)
        with open(error_file_name, 'a') as file:
            file.write(name+'\n')
        if os.path.exists(name + '.wv'):
            os.remove(name + '.wv')
        if os.path.exists(name + '.dict'):
            os.remove(name + '.dict')
        if os.path.exists(name + '.words'):
            os.remove(name + '.words')
        return failed_books + 1


def AmosOz2Features(directory_path, window, excel_name, error_file_name, from_api=False, books_csv=False):
    '''
    :param directory_path: if from_api=False, the directory path where the books are located
    :param window: window size for w2v
    :param excel_name: name of excel file to save output (without .xls)
    :param error_file_name: name of a text file that contains failed books (use .txt)
    :param from_api: boolean parameter, set to True to use gutenberg api
    :param books_csv: if from_api=True, name of csv file that contains names of books from gutenberg project (use .csv)
    :return: exports extracted features for books in excel file and a text file with names of books that did not work
    '''
    columns = ["book name", "avg of Word2Vec", "std of Word2Vec", "max of Word2Vec", "avg of differential",
               "std of differential", "max of differential", "min of differential", "avg of co-occurrence",
               "std of co-occurrence", "max of co-occurrence", "min of co-occurrence",
               "largest cc of G", "num cc of G", "diameter of largest cc in G", "number of cycles in G",
               "average degree in G",
               "largest cc of red", "num cc of red", "diameter of largest cc in red", "number of cycles in red",
               "average degree in red",
               "largest cc of blue", "num cc of blue", "diameter of largest cc in blue", "number of cycles in blue",
               "average degree in blue",
               "largest cc of neutral", "num cc of neutral", "diameter of largest cc in neutral",
               "number of cycles in neutral", "average degree in neutral",
               "largest cc of semantic", "num cc of semantic", "diameter of largest cc in semantic",
               "number of cycles in semantic", "average degree in semantic",
               "largest cc of co-occurrence", "num cc of co-occurrence", "diameter of largest cc in co-occurrence",
               "number of cycles in co-occurrence", "average degree in co-occurrence",
               "ratio between red and blue edges in G"]
    wb = Workbook()
    sheet1 = wb.add_sheet(excel_name)
    for i in range(len(columns)):
        sheet1.write(0, i, columns[i])
    failed_books = 0
    if os.path.exists(error_file_name):
        os.remove(error_file_name)
    if from_api:
        names = extract_names_from_excel(books_csv)
        for i, name in enumerate(names):
            check = AmosOz2Features_name(name, i, window, failed_books, error_file_name,
                                         sheet1, from_api)
            if check != -1:
                failed_books = check
        else:
            failed_books += 1
    else:
        for i, filename in enumerate(os.listdir(directory_path)):
            book_path_origin = os.path.join(directory_path, filename)
            name = "_".join(filename.split(".")[:-1])
            check = AmosOz2Features_name(name, i, window, failed_books, error_file_name,
                                         sheet1, from_api, book_path_origin)
            if check != -1:
                failed_books = check
        else:
            failed_books += 1
    wb.save(excel_name + ".xls")


def sub_graphs_plots(names):
    '''
    :param names: list of names of existing AmosOz2Vec models
    :return: None. plots of sub-graphs for each model
    '''
    #sub graphs plots
    file_names = [name+'.wv' for name in names]
    dict_names = [name+'.dict' for name in names]
    words = [name+'.words' for name in names]
    splits_list = semantic_split(names)
    colors_list = decide_color(names)[0]
    for i, name in enumerate(names):
        G, avg_wv, avg_count = create_graph(file_names[i], words[i], dict_names[i])
        semantic_graph, co_graph = create_semantic_graphs(G, splits_list[i])
        red_graph, blue_graph, neutral_graph = create_color_graphs(G, colors_list[i])
        draw_graph(semantic_graph, "semantic graph")
        draw_graph(co_graph, "co-occurrence graph")
        draw_graph(red_graph, "red graph")
        draw_graph(blue_graph, "blue graph")
        draw_graph(neutral_graph, "neutral graph")


# features extraction
'''directory_path = "C:/Users/kobik/Desktop/books_from_ Gutenburg"
window = 8
excel_name = "features_test"
columns = ["book name", "avg of Word2Vec", "std of Word2Vec", "max of Word2Vec", "avg of differential",
           "std of differential", "max of differential", "min of differential", "avg of co-occurrence",
           "std of co-occurrence", "max of co-occurrence", "min of co-occurrence",
           "largest cc of G", "num cc of G", "diameter of largest cc in G", "number of cycles in G", "average degree in G",
           "largest cc of red", "num cc of red", "diameter of largest cc in red", "number of cycles in red", "average degree in red",
           "largest cc of blue", "num cc of blue", "diameter of largest cc in blue", "number of cycles in blue", "average degree in blue",
           "largest cc of neutral", "num cc of neutral", "diameter of largest cc in neutral", "number of cycles in neutral", "average degree in neutral",
           "largest cc of semantic", "num cc of semantic", "diameter of largest cc in semantic", "number of cycles in semantic", "average degree in semantic",
           "largest cc of co-occurrence", "num cc of co-occurrence", "diameter of largest cc in co-occurrence", "number of cycles in co-occurrence", "average degree in co-occurrence",
           "ratio between red and blue edges in G"]
names = []
for filename in os.listdir(directory_path):
    book_path_origin = os.path.join(directory_path, filename)
    name = "_".join(filename.split(".")[:-1])
    names.append(name)
    AmosOz2Vec(name, book_path_origin, 10, window, is_gui=False) # number of characters does not matter because is_gui flag is off
bad_books = ["932-0", "9830-0"]
for bad in bad_books:
    names.remove(bad)
file_names = [name + '.wv' for name in names]
dict_names = [name + '.dict' for name in names]
words = [name + '.words' for name in names]
colors_dict_list, avg_diff_list, std_diff_list, max_diff_list, min_diff_list = decide_color(names)
semantic_split_list = semantic_split(names)
graphs = [create_graph(file_name, words[k], dict_names[k])[0] for k, file_name in enumerate(file_names)]
wb = Workbook()
sheet1 = wb.add_sheet(excel_name)
for i in range(len(columns)):
    sheet1.write(0, i, columns[i])
for i, G in enumerate(graphs):
    features = extract_features(dict_names[i], G, colors_dict_list[i], avg_diff_list[i], std_diff_list[i],
                           max_diff_list[i], min_diff_list[i], semantic_split_list[i])
    sheet1.write(i+1, 0, names[i])
    for j in range(len(features)):
        sheet1.write(i+1, j+1, float(features[j]))
    # os.remove(file_names[i])
    # os.remove(dict_names[i])
    # os.remove(words[i])
wb.save(excel_name + ".xls")'''


AmosOz2Features("C:/Users/kobik/Desktop/temp", window=15, excel_name="features_bestsellers_15",
                error_file_name="error_bestsellers_15.txt", from_api=True, books_csv="bestsellers.csv")
AmosOz2Features("C:/Users/kobik/Desktop/temp", window=15, excel_name="features_random_15",
                error_file_name="error_random_15.txt", from_api=True, books_csv="random_gutenberg.csv")
# bad_books=["932-0", "9830-0"]
# sub_graphs_plots(names=["October_A_Perfect_Peace_8"])


