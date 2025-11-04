import csv
from pyswip import Prolog
from itertools import islice
import tkinter as tk
from tkinter import ttk, scrolledtext

class PrologKBGenerator:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.data = []
        self.columns = []
        self.feature_mapping = {}

    def load_data(self):
        with open(self.input_file, mode='r') as file:
            reader = csv.reader(file)
            self.columns = next(reader)
            for row in reader:
                self.data.append(row)

        self.feature_mapping = {
            f"Feature{i}": self.columns[i + 1] for i in range(1, len(self.columns) - 1)
        }

    def generate_kb(self):
        with open(self.output_file, 'w') as file:
            for row in self.data:
                diagnosis = row[1].lower()
                fact = f"data({row[0]}, {diagnosis}, {', '.join(map(str, row[2:]))})."
                file.write(fact + "\n")

    def query_kb(self, query, limit=10):
        prolog = Prolog()
        prolog.consult(self.output_file)
        result = list(islice(prolog.query(query), limit))
        return result

def print_results_to_window(results, columns, feature_mapping, query_description, text_widget):
    text_widget.configure(state='normal')
    text_widget.delete(1.0, tk.END)

    text_widget.insert(tk.END, f"{query_description}\n", ('title',))
    text_widget.insert(tk.END, "=" * 70 + "\n")

    if not results:
        text_widget.insert(tk.END, "Nessun risultato trovato.\n", ('error',))
        text_widget.configure(state='disabled')
        return

    for i, result in enumerate(results, start=1):
        text_widget.insert(tk.END, f"\nRisultato {i}:\n", ('header',))
        text_widget.insert(tk.END, f"  ID: {result['ID']}\n", ('normal',))
        for column in columns:
            col_name = feature_mapping.get(column, column)
            text_widget.insert(tk.END, f"  {col_name}: {float(result.get(column, 0)):.3f}\n")
        text_widget.insert(tk.END, "-" * 50 + "\n")
    text_widget.configure(state='disabled')

#--- Interfaccia Principale ---
def run_queries_and_display():
    input_file = "data.csv"
    output_file = "kb.pl"

    kb_generator = PrologKBGenerator(input_file, output_file)
    kb_generator.load_data()
    kb_generator.generate_kb()

    window = tk.Tk()
    window.title("💡 Query su Base di Conoscenza Prolog")
    window.geometry("900x750")
    window.configure(bg="#f0f2f5")

    style = ttk.Style()
    style.theme_use('clam')
    style.configure('TButton', font=('Segoe UI', 10, 'bold'), padding=6)

    title_label = tk.Label(window, text="Prolog Query Explorer", font=("Segoe UI", 16, 'bold'), bg="#f0f2f5", fg="#1f4e79")
    title_label.pack(pady=10)

    text_frame = ttk.LabelFrame(window, text="Risultati delle Query", padding=10)
    text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    text_widget = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, width=110, height=30, font=("Consolas", 10))
    text_widget.pack(fill=tk.BOTH, expand=True)

    text_widget.tag_config('title', font=('Segoe UI', 12, 'bold'), foreground='#003366')
    text_widget.tag_config('header', font=('Segoe UI', 10, 'bold'), foreground='#00509e')
    text_widget.tag_config('error', font=('Segoe UI', 10, 'bold'), foreground='red')

    queries = [
        {
            "description": "Campioni con radius_mean > 16.0 e smoothness_mean < 0.1",
            "query": "data(ID, Diagnosis, Feature1, Feature2, Feature3, Feature4, Feature5, Feature6, Feature7, Feature8, Feature9, Feature10, Feature11, Feature12, Feature13, Feature14, Feature15, Feature16, Feature17, Feature18, Feature19, Feature20, Feature21, Feature22, Feature23, Feature24, Feature25, Feature26, Feature27, Feature28, Feature29, Feature30), Feature1 > 16.0, Feature6 < 0.1.",
            "columns": ["Feature1", "Feature6"]
        },
        {
            "description": "Campioni benigni (Diagnosis = b) con area_mean < 600",
            "query": "data(ID, b, Feature1, Feature2, Feature3, Feature4, Feature5, Feature6, Feature7, Feature8, Feature9, Feature10, Feature11, Feature12, Feature13, Feature14, Feature15, Feature16, Feature17, Feature18, Feature19, Feature20, Feature21, Feature22, Feature23, Feature24, Feature25, Feature26, Feature27, Feature28, Feature29, Feature30), Feature4 < 600.",
            "columns": ["Feature4"]
        },
        {
            "description": "Campioni maligni (Diagnosis = m) con compactness_mean > 0.2 e texture_mean > 20",
            "query": "data(ID, m, Feature1, Feature2, Feature3, Feature4, Feature5, Feature6, Feature7, Feature8, Feature9, Feature10, Feature11, Feature12, Feature13, Feature14, Feature15, Feature16, Feature17, Feature18, Feature19, Feature20, Feature21, Feature22, Feature23, Feature24, Feature25, Feature26, Feature27, Feature28, Feature29, Feature30), Feature6 > 0.2, Feature2 > 20.",
            "columns": ["Feature6", "Feature2"]
        },
        {
            "description": "Campioni con concavity_mean > 0.3 o symmetry_mean < 0.18",
            "query": "data(ID, Diagnosis, Feature1, Feature2, Feature3, Feature4, Feature5, Feature6, Feature7, Feature8, Feature9, Feature10, Feature11, Feature12, Feature13, Feature14, Feature15, Feature16, Feature17, Feature18, Feature19, Feature20, Feature21, Feature22, Feature23, Feature24, Feature25, Feature26, Feature27, Feature28, Feature29, Feature30), (Feature8 > 0.3 ; Feature24 < 0.18).",
            "columns": ["Feature8", "Feature24"]
        },
        {
            "description": "Campioni con fractal_dimension_mean < 0.06 e smoothness_mean > 0.09",
            "query": "data(ID, Diagnosis, Feature1, Feature2, Feature3, Feature4, Feature5, Feature6, Feature7, Feature8, Feature9, Feature10, Feature11, Feature12, Feature13, Feature14, Feature15, Feature16, Feature17, Feature18, Feature19, Feature20, Feature21, Feature22, Feature23, Feature24, Feature25, Feature26, Feature27, Feature28, Feature29, Feature30), Feature10 < 0.06, Feature6 > 0.09.",
            "columns": ["Feature10", "Feature6"]
        }
    ]

    btn_frame = ttk.LabelFrame(window, text="Esegui Query", padding=10)
    btn_frame.pack(fill=tk.X, padx=10, pady=10)

    def execute_query(query_info):
        results = kb_generator.query_kb(query_info["query"], limit=10)
        print_results_to_window(results, query_info["columns"], kb_generator.feature_mapping, query_info["description"], text_widget)

    for q in queries:
        button = ttk.Button(btn_frame, text=q["description"], command=lambda q=q: execute_query(q))
        button.pack(fill=tk.X, pady=4, padx=5)

    window.mainloop()

if __name__ == "__main__":
    run_queries_and_display()
