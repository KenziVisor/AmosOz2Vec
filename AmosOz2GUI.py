from AmosOz2Function import AmosOz2Vec
from AmosOz2Function import graph_theory
from AmosOz2Function import graph_theory_closest
from AmosOz2Function import sorted_edges
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
import threading
import multiprocessing
import time
import os

if __name__ == '__main__':
    AmosOz2Vec_params = ['', '', '', '']
    window = tk.Tk()
    window.title("AmosOz2Vec")
    window.geometry("270x500")
    for i in range(6):
        window.columnconfigure(i, weight=1)
        window.rowconfigure(i, weight=1)


def destroy(some_window):
    window.wm_attributes("-disabled", False)
    some_window.destroy()


def browseFiles(some_string_var):
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          filetypes=(("Text files",
                                                        "*.txt*"),
                                                       ("all files",
                                                        "*.*")))
    AmosOz2Vec_params[1] = filename
    some_string_var.set(filename)


def stop_AmosOz2Vec(t1, AmosOz2Vec_window, upload_book_var, pb):
    pb.start(50)
    AmosOz2Vec_window.wm_attributes("-disabled", True)
    while t1.is_alive():
        time.sleep(2)
    AmosOz2Vec_window.wm_attributes("-disabled", False)
    pb.stop()
    # AmosOz2Vec(AmosOz2Vec_params[0], AmosOz2Vec_params[1], AmosOz2Vec_params[2], AmosOz2Vec_params[3])
    for i in range(len(AmosOz2Vec_params)):
        AmosOz2Vec_params[i] = ''
    pb.grid_forget()
    upload_book_var.set('')
    messagebox.showinfo("Great", "Completed!")


def start_AmosOz2Vec(AmosOz2Vec_window, name_entry, characters_num_entry, window_size_entry,
                     upload_book_var, pb):
    if not name_entry.get() or not characters_num_entry.get() or not window_size_entry.get() or AmosOz2Vec_params[1] == '':
        messagebox.showerror("error", "Input must be filled on each entry")
        return
    characters_num_check = characters_num_entry.get()
    window_size_check = window_size_entry.get()
    if not (characters_num_check.isdigit() and window_size_check.isdigit()):
        messagebox.showerror("error", "Characters number or window size are not integers")
        return
    AmosOz2Vec_params[0] = name_entry.get()
    AmosOz2Vec_params[2] = int(characters_num_check)
    AmosOz2Vec_params[3] = int(window_size_check)
    pb.grid(row=8, column=2)
    t1 = threading.Thread(target=AmosOz2Vec, args=AmosOz2Vec_params)
    t2 = threading.Thread(target=stop_AmosOz2Vec, args=(t1, AmosOz2Vec_window, upload_book_var, pb))
    t1.start()
    t2.start()


def handle_AmosOz2Vec_button():
    window.wm_attributes("-disabled", True)
    AmosOz2Vec_window = tk.Toplevel(window)
    AmosOz2Vec_window.title("Load Book")
    AmosOz2Vec_window.geometry("700x300")
    for i in range(8):
        AmosOz2Vec_window.columnconfigure(i, weight=1)
        AmosOz2Vec_window.rowconfigure(i, weight=1)
    name_label = tk.Label(
        AmosOz2Vec_window,
        text="Enter name",
        fg="black",  # Set the text color to white
        bg="white",  # Set the background color to black
        width=35,
        height=1,
        relief="flat"
    )
    name_label.grid(row=0, column=0)

    name_entry = tk.Entry(
                AmosOz2Vec_window,
                fg="black",
                bg="white",
                width=45)
    name_entry.grid(row=0, column=2)

    characters_num_label = tk.Label(
        AmosOz2Vec_window,
        text="Enter number of characters",
        fg="black",  # Set the text color to white
        bg="white",  # Set the background color to black
        width=35,
        height=1,
        relief="flat"
    )
    characters_num_label.grid(row=2, column=0)

    characters_num_entry = tk.Entry(
                        AmosOz2Vec_window,
                        fg="black",
                        bg="white",
                        width=45)
    characters_num_entry.grid(row=2, column=2)

    window_size_label = tk.Label(
        AmosOz2Vec_window,
        text="Enter window size",
        fg="black",  # Set the text color to white
        bg="white",  # Set the background color to black
        width=35,
        height=1,
        relief="flat"
    )
    window_size_label.grid(row=4, column=0)

    window_size_entry = tk.Entry(
                        AmosOz2Vec_window,
                        fg="black",
                        bg="white",
                        width=45)
    window_size_entry.grid(row=4, column=2)

    upload_book_var = tk.StringVar()
    upload_book_label = tk.Label(
        AmosOz2Vec_window,
        textvariable=upload_book_var,
        fg="black",  # Set the text color to white
        bg="white",  # Set the background color to black
        width=50,
        height=1,
    )
    upload_book_label.grid(row=6, column=2)

    upload_book_button = tk.Button(
        AmosOz2Vec_window,
        text="Upload your book",
        width=35,
        height=1,
        bg="black",
        fg="white",
        command=lambda: browseFiles(upload_book_var)
    )
    upload_book_button.grid(row=6, column=0)

    pb = ttk.Progressbar(
        AmosOz2Vec_window,
        orient='horizontal',
        mode='indeterminate',
        length=280
    )

    start_button = tk.Button(
        AmosOz2Vec_window,
        text="Start",
        width=35,
        height=2,
        bg="black",
        fg="white",
        command=lambda: start_AmosOz2Vec(AmosOz2Vec_window, name_entry, characters_num_entry, window_size_entry,
                                         upload_book_var, pb)
    )
    start_button.grid(row=8, column=0)

    AmosOz2Vec_window.protocol("WM_DELETE_WINDOW", lambda: destroy(AmosOz2Vec_window))
    AmosOz2Vec_window.mainloop()


def stop_graph_theory(p1, graph_theory_window, pb):
    pb.start(50)
    graph_theory_window.wm_attributes("-disabled", True)
    while p1.is_alive():
        time.sleep(2)
    graph_theory_window.wm_attributes("-disabled", False)
    pb.stop()
    pb.grid_forget()
    messagebox.showinfo("Great", "Completed!")


def start_graph_theory(graph_theory_window, name_entry, pb):
    if not name_entry.get():
        messagebox.showerror("error", "Input must be filled on each entry")
        return
    names = name_entry.get().split(', ')
    for name in names:
        if not os.path.exists(os.path.join(os.getcwd(), name+".wv")):
            messagebox.showerror("error", f"{name} is not configured")
            return
    pb.grid(row=2, column=1)
    p1 = multiprocessing.Process(target=graph_theory, args=[names])
    t2 = threading.Thread(target=stop_graph_theory, args=(p1, graph_theory_window, pb))
    p1.start()
    t2.start()


def handle_graph_theory_button():
    window.wm_attributes("-disabled", True)
    graph_theory_window = tk.Tk()
    graph_theory_window.title("Characters Heatmap")
    graph_theory_window.geometry("500x300")
    for i in range(2):
        graph_theory_window.columnconfigure(i, weight=1)
        graph_theory_window.rowconfigure(i, weight=1)
    name_label = tk.Label(
        graph_theory_window,
        text="Enter names",
        fg="black",  # Set the text color to white
        bg="white",  # Set the background color to black
        width=20,
        height=1,
        relief="flat"
    )
    name_label.grid(row=0, column=0)

    name_entry = tk.Entry(
                graph_theory_window,
                fg="black",
                bg="white",
                width=45)
    name_entry.grid(row=0, column=1)

    pb = ttk.Progressbar(
        graph_theory_window,
        orient='horizontal',
        mode='indeterminate',
        length=280
    )

    start_button = tk.Button(
        graph_theory_window,
        text="Start",
        width=20,
        height=2,
        bg="black",
        fg="white",
        command=lambda: start_graph_theory(graph_theory_window, name_entry, pb)
    )
    start_button.grid(row=2, column=0)

    graph_theory_window.protocol("WM_DELETE_WINDOW", lambda: destroy(graph_theory_window))
    graph_theory_window.mainloop()


def stop_graph_theory_closest(p1, graph_theory_closest_window, pb):
    pb.start(50)
    graph_theory_closest_window.wm_attributes("-disabled", True)
    while p1.is_alive():
        time.sleep(2)
    graph_theory_closest_window.wm_attributes("-disabled", False)
    pb.stop()
    pb.grid_forget()
    messagebox.showinfo("Great", "Completed!")


def start_graph_theory_closest(graph_theory_closest_window, name_entry, n_entry, pb):
    if not name_entry.get() or not n_entry.get():
        messagebox.showerror("error", "Input must be filled on each entry")
        return
    names = name_entry.get().split(', ')
    n_try = n_entry.get()
    if not n_try.isdigit():
        messagebox.showerror("error", "Number of closest words must be an integer")
        return
    n = int(n_try)
    for name in names:
        if not os.path.exists(os.path.join(os.getcwd(), name+".wv")):
            messagebox.showerror("error", f"{name} is not configured")
            return
    pb.grid(row=4, column=2)
    p1 = multiprocessing.Process(target=graph_theory_closest, args=(names, n))
    t2 = threading.Thread(target=stop_graph_theory_closest, args=(p1, graph_theory_closest_window, pb))
    p1.start()
    t2.start()


def handle_graph_theory_closest_button():
    window.wm_attributes("-disabled", True)
    graph_theory_closest_window = tk.Tk()
    graph_theory_closest_window.title("Closest Words Graph")
    graph_theory_closest_window.geometry("700x500")
    for i in range(4):
        graph_theory_closest_window.columnconfigure(i, weight=1)
        graph_theory_closest_window.rowconfigure(i, weight=1)
    name_label = tk.Label(
        graph_theory_closest_window,
        text="Enter names",
        fg="black",  # Set the text color to white
        bg="white",  # Set the background color to black
        width=35,
        height=1,
        relief="flat"
    )
    name_label.grid(row=0, column=0)

    name_entry = tk.Entry(
                graph_theory_closest_window,
                fg="black",
                bg="white",
                width=45)
    name_entry.grid(row=0, column=2)

    n_label = tk.Label(
        graph_theory_closest_window,
        text="Enter number of closest words",
        fg="black",  # Set the text color to white
        bg="white",  # Set the background color to black
        width=35,
        height=1,
        relief="flat"
    )
    n_label.grid(row=2, column=0)

    n_entry = tk.Entry(
        graph_theory_closest_window,
        fg="black",
        bg="white",
        width=45)
    n_entry.grid(row=2, column=2)

    pb = ttk.Progressbar(
        graph_theory_closest_window,
        orient='horizontal',
        mode='indeterminate',
        length=280
    )

    start_button = tk.Button(
        graph_theory_closest_window,
        text="Start",
        width=35,
        height=2,
        bg="black",
        fg="white",
        command=lambda: start_graph_theory_closest(graph_theory_closest_window, name_entry, n_entry, pb)
    )
    start_button.grid(row=4, column=0)
    graph_theory_closest_window.protocol("WM_DELETE_WINDOW", lambda: destroy(graph_theory_closest_window))
    graph_theory_closest_window.mainloop()


def stop_summarize_results(p1, graph_theory_closest_window, pb):
    pb.start(50)
    graph_theory_closest_window.wm_attributes("-disabled", True)
    while p1.is_alive():
        time.sleep(2)
    graph_theory_closest_window.wm_attributes("-disabled", False)
    pb.stop()
    pb.grid_forget()
    messagebox.showinfo("Great", "Completed!")


def start_summarize_results(summarize_window, name_entry, name_excel_entry, model_var,
                            counter_var, manual_var, pb):
    if not name_entry.get() or not name_excel_entry.get():
        messagebox.showerror("error", "Input must be filled on each entry")
        return
    names = name_entry.get().split(', ')
    name_excel = name_excel_entry.get()
    for name in names:
        if not os.path.exists(os.path.join(os.getcwd(), name+".wv")):
            messagebox.showerror("error", f"{name} is not configured")
            return
    model_result = model_var.get()
    counter_result = counter_var.get()
    manual_result = manual_var.get()
    pb.grid(row=7, column=2)
    p1 = multiprocessing.Process(target=sorted_edges, args=(names, name_excel,
                                                            model_result, counter_result, manual_result))
    t2 = threading.Thread(target=stop_graph_theory_closest, args=(p1, summarize_window, pb))
    p1.start()
    t2.start()


def is_checked(cb):
    cb.set(not cb.get())


def handle_summarize_results_button():
    window.wm_attributes("-disabled", True)
    summarize_window = tk.Tk()
    summarize_window.title("Sorted Edges")
    summarize_window.geometry("800x500")
    for i in range(8):
        summarize_window.columnconfigure(i, weight=1)
        summarize_window.rowconfigure(i, weight=1)
    name_label = tk.Label(
        summarize_window,
        text="Enter names",
        fg="black",  # Set the text color to white
        bg="white",  # Set the background color to black
        width=35,
        height=1,
        relief="flat"
    )
    name_label.grid(row=0, column=0)

    name_entry = tk.Entry(
                summarize_window,
                fg="black",
                bg="white",
                width=45)
    name_entry.grid(row=0, column=2)

    name_excel_label = tk.Label(
        summarize_window,
        text="Enter excel file name (without .xls)",
        fg="black",  # Set the text color to white
        bg="white",  # Set the background color to black
        width=35,
        height=1,
        relief="flat"
    )
    name_excel_label.grid(row=2, column=0)

    name_excel_entry = tk.Entry(
        summarize_window,
        fg="black",
        bg="white",
        width=45)
    name_excel_entry.grid(row=2, column=2)

    pb = ttk.Progressbar(
        summarize_window,
        orient='horizontal',
        mode='indeterminate',
        length=280
    )

    model_checkbox_label = tk.Label(
        summarize_window,
        text="Word2Vec models",
        fg="black",  # Set the text color to white
        bg="white",  # Set the background color to black
        width=35,
        height=1,
        relief="flat"
    )
    model_checkbox_label.grid(row=4, column=0)

    model_var = tk.BooleanVar()
    model_checkbox = tk.Checkbutton(
        summarize_window,
        variable=model_var,
        onvalue=True,
        offvalue=False,
        command=lambda: is_checked(model_var))
    model_checkbox.grid(row=5, column=0)

    counter_checkbox_label = tk.Label(
        summarize_window,
        text="counters",
        fg="black",  # Set the text color to white
        bg="white",  # Set the background color to black
        width=35,
        height=1,
        relief="flat"
    )
    counter_checkbox_label.grid(row=4, column=1)

    counter_var = tk.BooleanVar()
    counter_checkbox = tk.Checkbutton(
        summarize_window,
        variable=counter_var,
        onvalue=True,
        offvalue=False,
        command=lambda: is_checked(counter_var))
    counter_checkbox.grid(row=5, column=1)

    manual_checkbox_label = tk.Label(
        summarize_window,
        text="manuals",
        fg="black",  # Set the text color to white
        bg="white",  # Set the background color to black
        width=35,
        height=1,
        relief="flat"
    )
    manual_checkbox_label.grid(row=4, column=2)

    manual_var = tk.BooleanVar()
    manual_checkbox = tk.Checkbutton(
        summarize_window,
        variable=manual_var,
        onvalue=True,
        offvalue=False,
        command=lambda: is_checked(manual_var))
    manual_checkbox.grid(row=5, column=2)

    start_button = tk.Button(
        summarize_window,
        text="Start",
        width=35,
        height=2,
        bg="black",
        fg="white",
        command=lambda: start_summarize_results(summarize_window, name_entry, name_excel_entry, model_var,
                                                counter_var, manual_var, pb)
    )
    start_button.grid(row=7, column=0)
    summarize_window.protocol("WM_DELETE_WINDOW", lambda: destroy(summarize_window))
    summarize_window.mainloop()


def handle_instructions_button():
    window.wm_attributes("-disabled", True)
    instructions_window = tk.Tk()
    instructions_window.title("Instructions")
    instructions_window.geometry("500x500")
    manual_checkbox_label = tk.Label(
        instructions_window,
        text="Welcome to AmosOz2Vec!\n"
             "Ronen Haim Portnikh\n"
             "Sort your edges",
        fg="black",  # Set the text color to white
        bg="white",  # Set the background color to black
        width=100,
        height=1,
        relief="flat"
    )
    manual_checkbox_label.grid(row=4, column=2)
    instructions_window.protocol("WM_DELETE_WINDOW", lambda: destroy(instructions_window))
    instructions_window.mainloop()
# name, book_path_origin, number_of_characters, window size


if __name__ == '__main__':
    label = tk.Label(
        text="AmosOz2Vec",
        fg="black",  # Set the text color to white
        bg="white",  # Set the background color to black
        width=10,
        height=0,
    )
    label.grid(row=0, column=0)

    AmosOz2Vec_button = tk.Button(
        text="Load book",
        width=35,
        height=5,
        bg="black",
        fg="white",
        command=handle_AmosOz2Vec_button
    )
    AmosOz2Vec_button.grid(row=1, column=0)

    graph_theory_button = tk.Button(
        text="Draw characters heatmap",
        width=35,
        height=5,
        bg="black",
        fg="white",
        command=handle_graph_theory_button
    )
    graph_theory_button.grid(row=2, column=0)

    graph_theory_closest_button = tk.Button(
        text="Draw characters n-closest words graph",
        width=35,
        height=5,
        bg="black",
        fg="white",
        command=handle_graph_theory_closest_button
    )
    graph_theory_closest_button.grid(row=3, column=0)

    summarize_results_button = tk.Button(
        text="Summarize your results",
        width=35,
        height=5,
        bg="black",
        fg="white",
        command=handle_summarize_results_button
    )
    summarize_results_button.grid(row=4, column=0)

    Instructions_button = tk.Button(
        text="Instructions",
        width=35,
        height=5,
        bg="black",
        fg="white",
        command=handle_instructions_button
    )
    Instructions_button.grid(row=5, column=0)

    window.mainloop()
