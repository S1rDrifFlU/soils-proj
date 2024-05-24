import tkinter as tk
from tkinter import ttk
import pandas as pd
from tkinter import filedialog
import threading
import tkinter.messagebox as messagebox
from module_of_project import (  # type: ignore
    preprocess_data,
    split_data,
    define_regressors,
    define_search_spaces,
    run_hyperparameter_tuning,
    evaluate_regressors,
)


processing_running = False
stop_processing_flag = False
cancel_flag = False  
selected_filepath = ""
n_features_entry = None
select_dataset_button = None
run_button = None
stop_button = None
columns_to_drop_entry = None
columns_to_drop = []
processing_thread = None
result_frame = None


def preprocess_data_gui(filepath, columns_to_drop, target_variable):
    df = pd.read_excel(filepath)
    df_preprocessed = preprocess_data(df, columns_to_drop)
    X = df_preprocessed.drop(columns=[target_variable])
    y = df_preprocessed[target_variable]
    return X, y


def disable_elements():
    n_features_entry.config(state=tk.DISABLED)
    select_dataset_button.config(state=tk.DISABLED)
    columns_to_drop_entry.config(state=tk.DISABLED)
    save_columns_button.config(state=tk.DISABLED)
    save_target_button.config(state=tk.DISABLED)
    target_variable_entry.config(state=tk.DISABLED)
    run_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)


def enable_elements():
    n_features_entry.config(state=tk.NORMAL)
    select_dataset_button.config(state=tk.NORMAL)
    columns_to_drop_entry.config(state=tk.NORMAL)
    save_columns_button.config(state=tk.NORMAL)
    save_target_button.config(state=tk.NORMAL)
    target_variable_entry.config(state=tk.NORMAL)
    run_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)


def stop_processing():
    global processing_running, stop_processing_flag, cancel_flag
    cancel_flag = True  
    stop_processing_flag = True
    enable_elements()


def display_results_table(results_df):
    global result_frame
    if result_frame:
        result_frame.destroy()
    result_frame = tk.Frame(window)
    result_frame.pack(fill=tk.BOTH, expand=1)

    xscrollbar = tk.Scrollbar(result_frame, orient=tk.HORIZONTAL)
    table = ttk.Treeview(result_frame, columns=list(results_df), show="headings", xscrollcommand=xscrollbar.set)
    xscrollbar.config(command=table.xview)
    xscrollbar.pack(side=tk.BOTTOM, fill=tk.X)

    for col in results_df.columns:
        table.heading(col, text=col, anchor='center')
        table.column(col, anchor='center')
    data = results_df.values.tolist()
    for row in data:
        table.insert("", tk.END, values=row, tags='centered')

    table.tag_configure('centered', anchor='center')

    table.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)



def open_file_dialog():
    global selected_filepath
    try:
        filepath = filedialog.askopenfilename()
        if filepath:
            selected_filepath = filepath
            print(selected_filepath)
            return filepath
    except Exception as e:
        messagebox.showerror("Error", f"Error opening file: {e}")

def choose_columns_to_drop():
    global columns_to_drop
    columns_to_drop = columns_to_drop_entry.get().split(',')
    if columns_to_drop:
        enable_elements()

def choose_target_variable():
    global target_variable
    target_variable = target_variable_entry.get()
    if target_variable:
        enable_elements()


def run_hyperparameter_tuning_process():
    global processing_thread
    if not selected_filepath:
        messagebox.showinfo("Info", "Please select a dataset first.")
        return
    n_features_to_select = int(n_features_entry.get())

    def process_data_and_display_results():
        global processing_running, cancel_flag
        processing_running = True
        stop_processing_flag = False
        cancel_flag = False  
        disable_elements()

        try:
            X, y = preprocess_data_gui(selected_filepath, columns_to_drop=columns_to_drop, target_variable=target_variable)
            X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
            regressors = define_regressors()
            search_spaces = define_search_spaces(regressors)
            search_methods = ["gp_minimize"]
            n_splits = 5
            n_repeats = 3
            scoring = "r2"
            number_of_iterations = 3

            
            if cancel_flag:
                return

            best_models, experiment_results, selected_features_mask = run_hyperparameter_tuning(
                X_train,
                y_train,
                test_size=0.2,
                n_splits=n_splits,
                n_repeats=n_repeats,
                scoring=scoring,
                n_features_to_select=n_features_to_select,
                search_methods=search_methods,
            )

            
            if cancel_flag:
                return

            results_df = evaluate_regressors(
                best_models, X, y, test_size=0.2, number_of_iterations=number_of_iterations, scoring=scoring, selected_features_mask=selected_features_mask
            )

            window.after(0, display_results_table, results_df) 
        except Exception as e:
            messagebox.showerror("Error", f"Error during processing: {e}")
        finally:
            processing_running = False
            window.after(0, enable_elements)  

    processing_thread = threading.Thread(target=process_data_and_display_results)
    processing_thread.start()



window = tk.Tk()
window.title("Machine Learning App")
window.geometry("800x600")

top_frame = tk.Frame(window)
top_frame.pack(pady=10)  

n_features_label = tk.Label(top_frame, text="Number of Features to Select:")
n_features_label.grid(row=0, column=0)  

n_features_entry = tk.Entry(top_frame)
n_features_entry.grid(row=0, column=1)  

select_dataset_button = tk.Button(top_frame, text="Select Dataset", command=open_file_dialog)
select_dataset_button.grid(row=0, column=2, padx=10)  

middle_frame = tk.Frame(window)
middle_frame.pack(pady=10)  

columns_to_drop_label = tk.Label(middle_frame, text="Columns to Drop (comma-separated):")
columns_to_drop_label.grid(row=0, column=0)  

columns_to_drop_entry = tk.Entry(middle_frame)
columns_to_drop_entry.grid(row=0, column=1, sticky="ew") 

save_columns_button = tk.Button(middle_frame, text="Save Columns", command=choose_columns_to_drop)
save_columns_button.grid(row=0, column=2, padx=10)  

bottom_frame = tk.Frame(window)
bottom_frame.pack(pady=10)  

target_variable_label = tk.Label(bottom_frame, text="Target Variable:")
target_variable_label.grid(row=0, column=0)  

target_variable_entry = tk.Entry(bottom_frame)
target_variable_entry.grid(row=0, column=1, sticky="ew")  

save_target_button = tk.Button(bottom_frame, text="Save Target", command=choose_target_variable)
save_target_button.grid(row=0, column=2, padx=10)  

run_button = tk.Button(bottom_frame, text="Run Hyperparameter Tuning", command=run_hyperparameter_tuning_process, state=tk.DISABLED)
run_button.grid(row=0, column=3, padx=10)  

stop_button = tk.Button(bottom_frame, text="Stop", command=stop_processing, state=tk.DISABLED)
stop_button.grid(row=0, column=4) 

window.mainloop()

