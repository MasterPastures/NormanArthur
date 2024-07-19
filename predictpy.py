import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
from collections import defaultdict
from fuzzywuzzy import fuzz
import random

def predict_missing_values(df, column, threshold=70):
    value_combinations = defaultdict(lambda: defaultdict(int))
    
    for _, row in df.iterrows():
        if pd.notnull(row[column]):
            key = tuple(str(row[col]) for col in df.columns if col != column and pd.notnull(row[col]))
            value_combinations[key][row[column]] += 1
    
    def find_most_similar_combination(row):
        if pd.isnull(row[column]):
            current_key = tuple(str(row[col]) for col in df.columns if col != column and pd.notnull(row[col]))
            
            if current_key in value_combinations:
                return max(value_combinations[current_key], key=value_combinations[current_key].get)
            
            max_similarity = 0
            best_value = None
            
            for key in value_combinations:
                similarity = sum(fuzz.ratio(str(k), str(v)) for k, v in zip(key, current_key)) / len(key)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_value = max(value_combinations[key], key=value_combinations[key].get)
            
            if max_similarity >= threshold:
                return best_value
            
            return df[column].mode().values[0]
        else:
            return row[column]
    
    imputed = df[column].isnull()
    df[column] = df.apply(find_most_similar_combination, axis=1)
    return df, imputed

class DataImputationGUI:
    def __init__(self, master, data=None):
        self.master = master
        self.master.title("Data Imputation Tool")
        self.master.geometry("1200x600")

        self.df = data
        self.imputed_cells = set()

        self.create_widgets()

    def create_widgets(self):
        button_frame = ttk.Frame(self.master, padding="10")
        button_frame.pack(fill=tk.X)

        ttk.Button(button_frame, text="Load Data", command=self.load_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Impute Data", command=self.impute_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Data", command=self.save_data).pack(side=tk.LEFT, padx=5)

        self.tree = ttk.Treeview(self.master, selectmode="extended")
        self.tree.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        y_scrollbar = ttk.Scrollbar(self.master, orient=tk.VERTICAL, command=self.tree.yview)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        x_scrollbar = ttk.Scrollbar(self.master, orient=tk.HORIZONTAL, command=self.tree.xview)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.tree.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)

        self.tree.bind("<Double-1>", self.on_double_click)

        if self.df is not None:
            self.display_data()

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.imputed_cells.clear()
                self.display_data()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")

    def impute_data(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        string_columns = self.df.select_dtypes(include=['object']).columns.tolist()
        for column in string_columns:
            self.df, imputed = predict_missing_values(self.df, column)
            for idx in imputed[imputed].index:
                self.imputed_cells.add((idx, column))
                self.df.at[idx, column] = f"[Imputed value]: {self.df.at[idx, column]}"

        self.display_data()
        messagebox.showinfo("Info", "Data imputation completed.")

    def save_data(self):
        if self.df is None:
            messagebox.showwarning("Warning", "No data to save.")
            return

        # Create a copy of the dataframe without the "[Imputed value]:" prefix
        df_to_save = self.df.copy()
        for idx, col in self.imputed_cells:
            value = df_to_save.at[idx, col]
            if isinstance(value, str) and value.startswith("[Imputed value]: "):
                df_to_save.at[idx, col] = value.replace("[Imputed value]: ", "")

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                df_to_save.to_csv(file_path, index=False)
                messagebox.showinfo("Info", f"Data saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save data: {str(e)}")

    def display_data(self):
        for i in self.tree.get_children():
            self.tree.delete(i)

        self.tree["columns"] = list(self.df.columns)
        self.tree["show"] = "headings"

        for column in self.df.columns:
            self.tree.heading(column, text=column)
            self.tree.column(column, width=100)

        for index, row in self.df.iterrows():
            self.tree.insert("", "end", values=list(row))

    def on_double_click(self, event):
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)

        if not item or not column:
            return

        column_id = int(column[1:]) - 1
        column_name = self.df.columns[column_id]
        current_value = self.tree.item(item, "values")[column_id]

        if current_value.startswith("[Imputed value]: "):
            current_value = current_value.replace("[Imputed value]: ", "")

        new_value = simpledialog.askstring("Edit Value", f"Edit value for {column_name}:", initialvalue=current_value)

        if new_value is not None:
            self.tree.set(item, column, new_value)
            self.df.at[int(item), column_name] = new_value
            if (int(item), column_name) in self.imputed_cells:
                self.imputed_cells.remove((int(item), column_name))

    def run(self):
        self.master.mainloop()

def generate_test_data(num_rows, columns, choices):
    data = []
    for i in range(num_rows):
        row = {'patient_id': i + 1}
        for column, options in choices.items():
            row[column] = random.choice(options) if random.random() > 0.1 else np.nan
        data.append(row)
    
    return pd.DataFrame(data, columns=columns)

def main():
    # Define the structure and content for the test data
    columns = ['patient_id', 'diagnosis', 'symptoms', 'treatment', 'brain_region_affected', 
               'age', 'gender', 'blood_type', 'medication', 'family_history']
    choices = {
        'diagnosis': ['Alzheimer\'s disease', 'Parkinson\'s disease', 'Huntington\'s disease', 'Multiple Sclerosis', 'ALS'],
        'symptoms': ['Memory loss', 'Tremor', 'Cognitive decline', 'Involuntary movements', 'Muscle weakness', 'Balance problems'],
        'treatment': ['Donepezil', 'Levodopa', 'Tetrabenazine', 'Interferon beta', 'Riluzole'],
        'brain_region_affected': ['Hippocampus', 'Substantia nigra', 'Striatum', 'White matter', 'Motor cortex'],
        'age': list(range(30, 90)),
        'gender': ['Male', 'Female'],
        'blood_type': ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'],
        'medication': ['Aricept', 'Exelon', 'Razadyne', 'Namenda', 'Sinemet', 'Rytary', 'Madopar', 'Xenazine', 'Austedo', 'Avonex', 'Betaseron', 'Copaxone', 'Rilutek', 'Radicava'],
        'family_history': ['Yes', 'No']
    }

    # Generate test data
    num_rows = 100
    test_data = generate_test_data(num_rows, columns, choices)
    
    # Create and run the GUI with the generated data
    root = tk.Tk()
    app = DataImputationGUI(root, data=test_data)
    app.run()

if __name__ == "__main__":
    main()

    """ 7/16: to-add:
    1. List of made imputations per unique identifier
    2. List of made corrections per unique identifier
    3. Convert csv to df for syn integrations"""