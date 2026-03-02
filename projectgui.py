import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np

# Load trained model
model = joblib.load("student_performance_model.pkl")

# Create main window
root = tk.Tk()
root.title("Student Performance Predictor")
root.geometry("500x620")
root.configure(bg="#f0f0f0")
root.resizable(False, False)

# Title
title = tk.Label(root, text="Student Performance Prediction",
                font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#333")
title.pack(pady=10)

subtitle = tk.Label(root, text="Enter student details to predict final grade (G3)",
                    font=("Arial", 9), bg="#f0f0f0", fg="#666")
subtitle.pack(pady=(0, 10))

# --- Input fields matching the 10 model features ---
# Feature info:
#   school:    GP=0, MS=1
#   Medu:      0-4 (Mother's education)
#   Fedu:      0-4 (Father's education)
#   studytime: 1-4 (Weekly study time)
#   failures:  0-3 (Number of past class failures)
#   higher:    no=0, yes=1 (Wants higher education)
#   Dalc:      1-5 (Workday alcohol consumption)
#   Walc:      1-5 (Weekend alcohol consumption)
#   G1:        0-19 (First period grade)
#   G2:        0-19 (Second period grade)

input_frame = tk.Frame(root, bg="#f0f0f0")
input_frame.pack(pady=5, padx=30, fill="x")

entries = {}

# --- Dropdown fields ---

# School
tk.Label(input_frame, text="School:", font=("Arial", 10),
        bg="#f0f0f0", anchor="w").grid(row=0, column=0, pady=4, sticky="w")
school_var = tk.StringVar(value="GP")
school_combo = ttk.Combobox(input_frame, textvariable=school_var,
                            values=["GP", "MS"], state="readonly", width=18)
school_combo.grid(row=0, column=1, pady=4, padx=(10, 0))
entries["school"] = ("combo", school_var, {"GP": 0, "MS": 1})

# Higher education
tk.Label(input_frame, text="Wants Higher Edu:", font=("Arial", 10),
        bg="#f0f0f0", anchor="w").grid(row=5, column=0, pady=4, sticky="w")
higher_var = tk.StringVar(value="yes")
higher_combo = ttk.Combobox(input_frame, textvariable=higher_var,
                            values=["yes", "no"], state="readonly", width=18)
higher_combo.grid(row=5, column=1, pady=4, padx=(10, 0))
entries["higher"] = ("combo", higher_var, {"yes": 1, "no": 0})

# --- Numeric (spinbox) fields ---

numeric_fields = [
    ("Medu", "Mother's Education (0-4):", 1, 0, 4),
    ("Fedu", "Father's Education (0-4):", 2, 0, 4),
    ("studytime", "Study Time (1-4 hrs/wk):", 3, 1, 4),
    ("failures", "Past Failures (0-3):", 4, 0, 3),
    ("Dalc", "Workday Alcohol (1-5):", 6, 1, 5),
    ("Walc", "Weekend Alcohol (1-5):", 7, 1, 5),
    ("G1", "First Period Grade (0-19):", 8, 0, 19),
    ("G2", "Second Period Grade (0-19):", 9, 0, 19),
]

for feat_name, label_text, row, min_val, max_val in numeric_fields:
    tk.Label(input_frame, text=label_text, font=("Arial", 10),
            bg="#f0f0f0", anchor="w").grid(row=row, column=0, pady=4, sticky="w")
    spin = tk.Spinbox(input_frame, from_=min_val, to=max_val, width=19,
                    font=("Arial", 10))
    spin.grid(row=row, column=1, pady=4, padx=(10, 0))
    entries[feat_name] = ("spin", spin, min_val, max_val)


# Prediction Function
def predict():
    try:
        values = {}

        # School (label-encoded)
        school_text = entries["school"][1].get()
        values["school"] = entries["school"][2][school_text]

        # Higher (label-encoded)
        higher_text = entries["higher"][1].get()
        values["higher"] = entries["higher"][2][higher_text]

        # Numeric fields
        for feat_name, label_text, row, min_val, max_val in numeric_fields:
            val = float(entries[feat_name][1].get())
            if val < min_val or val > max_val:
                messagebox.showwarning("Warning",
                    f"{feat_name} should be between {min_val} and {max_val}")
                return
            values[feat_name] = val

        # Build input array in the exact order the model expects
        feature_order = ['school', 'Medu', 'Fedu', 'studytime', 'failures',
                        'higher', 'Dalc', 'Walc', 'G1', 'G2']
        input_array = np.array([[values[f] for f in feature_order]])

        prediction = model.predict(input_array)

        result_label.config(
            text=f"Predicted Final Grade (G3): {round(prediction[0], 2)}",
            fg="green"
        )

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numbers for all fields.")
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed:\n{e}")


# Predict Button
predict_btn = tk.Button(root, text="Predict",
                        command=predict,
                        bg="#4CAF50", fg="white",
                        font=("Arial", 12, "bold"),
                        relief="flat", padx=20, pady=5,
                        cursor="hand2")
predict_btn.pack(pady=15)

# Result Label
result_label = tk.Label(root, text="", font=("Arial", 13, "bold"), bg="#f0f0f0")
result_label.pack(pady=10)

# Run GUI
root.mainloop()