import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
from collections import defaultdict
from fuzzywuzzy import fuzz
import random
import json

def predict_missing_values(df, column, threshold=70):
    value_combinations = defaultdict(lambda: defaultdict(int))
    imputed_values = []
    
    for _, row in df.iterrows():
        if pd.notnull(row[column]):
            key = tuple(str(row[col]) for col in df.columns if col != column and pd.notnull(row[col]))
            value_combinations[key][row[column]] += 1
    
    overall_frequencies = df[column].value_counts()
    
    def find_most_similar_combination(row):
        if pd.isnull(row[column]):
            current_key = tuple(str(row[col]) for col in df.columns if col != column and pd.notnull(row[col]))
            
            best_value = None
            other_candidates = []
            
            if current_key in value_combinations:
                sorted_values = sorted(value_combinations[current_key].items(), key=lambda x: x[1], reverse=True)
                best_value = sorted_values[0][0]
                other_candidates = [v[0] for v in sorted_values[1:3]]
            
            if not best_value:
                max_similarity = 0
                for key in value_combinations:
                    similarity = sum(fuzz.ratio(str(k), str(v)) for k, v in zip(key, current_key)) / len(key)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        sorted_values = sorted(value_combinations[key].items(), key=lambda x: x[1], reverse=True)
                        best_value = sorted_values[0][0]
                        other_candidates = [v[0] for v in sorted_values[1:3]]
            
            if not best_value or max_similarity < threshold:
                best_value = overall_frequencies.index[0]
                other_candidates = overall_frequencies.index[1:3].tolist()
            
            while len(other_candidates) < 2:
                remaining_values = [v for v in overall_frequencies.index if v not in [best_value] + other_candidates]
                if remaining_values:
                    other_candidates.append(random.choice(remaining_values))
                else:
                    break
            
            imputed_values.append({
                'id': row['patient_id'],
                'missing annotation column': column,
                'suggested value': best_value,
                'other possible replacements': other_candidates
            })
            return best_value
        else:
            return row[column]
    
    imputed = df[column].isnull()
    df[column] = df.apply(find_most_similar_combination, axis=1)
    return df, imputed, imputed_values

class DataImputationGUI:
    def __init__(self, master, data=None):
        self.master = master
        self.master.title("Data Imputation Tool")
        self.master.geometry("1200x600")

        self.df = data
        self.original_df = data.copy() if data is not None else None
        self.all_imputed_values = []

        self.create_widgets()

    def create_widgets(self):
        button_frame = ttk.Frame(self.master, padding="10")
        button_frame.pack(fill=tk.X)

        ttk.Button(button_frame, text="Load Data", command=self.load_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Impute Data", command=self.impute_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Undo Imputations", command=self.undo_imputations).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Imputed Dataframe As", command=self.save_imputed_dataframe).pack(side=tk.LEFT, padx=5)

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
                self.original_df = self.df.copy()
                self.all_imputed_values = []
                self.display_data()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {str(e)}")

    def impute_data(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load data first.")
            return

        self.all_imputed_values = []
        string_columns = self.df.select_dtypes(include=['object']).columns.tolist()
        for column in string_columns:
            self.df, imputed, imputed_values = predict_missing_values(self.df, column)
            self.all_imputed_values.extend(imputed_values)

        self.display_data()
        
        # Show information about imputed values
        imputed_count = len(self.all_imputed_values)
        message = f"Data imputation completed. {imputed_count} values were imputed.\n\n"
        message += "The next step is to save the annotation suggestions to a JSON file.\n"
        message += "This file will contain the suggested values and other possible replacements for each imputed cell."
        
        result = messagebox.askokcancel("Imputation Complete", message, icon=messagebox.INFO)
        
        if result:
            # Prompt to save JSON file
            self.write_imputed_values_json(self.all_imputed_values)
        else:
            messagebox.showinfo("Info", "You can save the annotation suggestions later by imputing the data again.")

    def undo_imputations(self):
        if self.original_df is None:
            messagebox.showwarning("Warning", "No original data available.")
            return

        self.df = self.original_df.copy()
        self.all_imputed_values = []
        self.display_data()
        messagebox.showinfo("Info", "Imputations undone. Original data restored.")

    def save_imputed_dataframe(self):
        if self.df is None:
            messagebox.showwarning("Warning", "No data to save.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df.to_csv(file_path, index=False)
                messagebox.showinfo("Info", f"Imputed dataframe saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save imputed dataframe: {str(e)}")

    def display_data(self):
        for i in self.tree.get_children():
            self.tree.delete(i)

        self.tree["columns"] = list(self.df.columns)
        self.tree["show"] = "headings"

        for column in self.df.columns:
            self.tree.heading(column, text=column)
            self.tree.column(column, width=100)

        for index, row in self.df.iterrows():
            values = list(row)
            item = self.tree.insert("", "end", values=values)
            
            # Highlight imputed cells
            for imputed_value in self.all_imputed_values:
                if imputed_value['id'] == row['patient_id']:
                    col_idx = self.df.columns.get_loc(imputed_value['missing annotation column'])
                    self.tree.item(item, tags=(f'imputed_{col_idx}',))
                    self.tree.tag_configure(f'imputed_{col_idx}', background='light yellow')

    def on_double_click(self, event):
        item = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)

        if not item or not column:
            return

        column_id = int(column[1:]) - 1
        column_name = self.df.columns[column_id]
        current_value = self.tree.item(item, "values")[column_id]

        new_value = simpledialog.askstring("Edit Value", f"Edit value for {column_name}:", initialvalue=current_value)

        if new_value is not None:
            self.tree.set(item, column, new_value)
            self.df.at[int(item), column_name] = new_value

    def write_imputed_values_json(self, imputed_values):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Save Annotation Suggestions"
        )
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json_data = {'annotation suggestions': imputed_values}
                    json_str = json.dumps(json_data, indent=2)
                    json_str = json_str.replace('"', "'")  # Replace double quotes with single quotes
                    f.write(json_str)
                messagebox.showinfo("Info", f"Annotation suggestions saved to {file_path}\n\nThis file contains the suggested values and other possible replacements for each imputed cell.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save annotation suggestions: {str(e)}")
        else:
            messagebox.showinfo("Info", "No file was saved. You can save the annotation suggestions later by imputing the data again.")

    def run(self):
        self.master.mainloop()

def generate_test_data(num_rows, columns, choices):
    data = []
    for i in range(num_rows):
        row = {'patient_id': i + 1}
        for column, options in choices.items():
            if column == 'age':
                row[column] = random.choice(options)
            else:
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