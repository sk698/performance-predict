"""
Student Performance Predictor - GUI Application
================================================
A modern, premium desktop application that uses a trained
Decision Tree Regressor model to predict student final grades (G3)
based on three key features: failures, G1 (period 1 grade), and G2 (period 2 grade).
"""

import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import os
import sys
import math


# ─── Color Palette & Design Tokens ──────────────────────────────────────────
class Theme:
    # Dark mode colors
    BG_PRIMARY = "#0f0f1a"
    BG_SECONDARY = "#1a1a2e"
    BG_CARD = "#16213e"
    BG_INPUT = "#1e2a47"
    BG_HOVER = "#253354"

    ACCENT_PRIMARY = "#6c63ff"
    ACCENT_SECONDARY = "#a29bfe"
    ACCENT_GLOW = "#7c73ff"

    TEXT_PRIMARY = "#e8e8f0"
    TEXT_SECONDARY = "#a0a0b8"
    TEXT_MUTED = "#6c6c80"

    SUCCESS = "#00c896"
    WARNING = "#ffa726"
    ERROR = "#ff5252"

    BORDER = "#2a2a4a"
    SHADOW = "#0a0a14"

    FONT_FAMILY = "Segoe UI"
    FONT_FAMILY_MONO = "Consolas"


# ─── Helper: Rounded Rectangle on Canvas ────────────────────────────────────
def create_rounded_rect(canvas, x1, y1, x2, y2, radius=20, **kwargs):
    """Draw a rounded rectangle on a canvas."""
    points = [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


# ─── Main Application ───────────────────────────────────────────────────────
class StudentPerformanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎓 Student Performance Predictor")
        self.root.geometry("700x820")
        self.root.configure(bg=Theme.BG_PRIMARY)
        self.root.resizable(False, False)

        # Center the window on screen
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = (screen_w - 700) // 2
        y = (screen_h - 820) // 2
        self.root.geometry(f"700x820+{x}+{y}")

        # Load the ML model
        self.model = self._load_model()

        # Build UI
        self._build_styles()
        self._build_header()
        self._build_input_section()
        self._build_predict_button()
        self._build_result_section()
        self._build_footer()

    # ── Load Model ───────────────────────────────────────────────────────
    def _load_model(self):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "student_performance_model.pkl")
        try:
            model = joblib.load(model_path)
            return model
        except FileNotFoundError:
            messagebox.showerror(
                "Model Not Found",
                f"Could not find the model file at:\n{model_path}\n\n"
                "Please make sure 'student_performance_model.pkl' is in the "
                "same directory as this application."
            )
            sys.exit(1)
        except Exception as e:
            messagebox.showerror("Error Loading Model", str(e))
            sys.exit(1)

    # ── Styles ───────────────────────────────────────────────────────────
    def _build_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Configure scale (slider) styling
        style.configure("Custom.Horizontal.TScale",
                        background=Theme.BG_CARD,
                        troughcolor=Theme.BG_INPUT,
                        sliderthickness=18)

    # ── Header ───────────────────────────────────────────────────────────
    def _build_header(self):
        header_frame = tk.Frame(self.root, bg=Theme.BG_PRIMARY)
        header_frame.pack(fill="x", padx=30, pady=(30, 5))

        # Decorative accent line
        accent_canvas = tk.Canvas(header_frame, height=4, bg=Theme.BG_PRIMARY,
                                  highlightthickness=0)
        accent_canvas.pack(fill="x", pady=(0, 15))
        accent_canvas.create_rectangle(0, 0, 640, 4, fill=Theme.ACCENT_PRIMARY,
                                       outline="")

        # App icon + title
        title_frame = tk.Frame(header_frame, bg=Theme.BG_PRIMARY)
        title_frame.pack(fill="x")

        icon_label = tk.Label(title_frame, text="🎓",
                              font=(Theme.FONT_FAMILY, 32),
                              bg=Theme.BG_PRIMARY, fg=Theme.TEXT_PRIMARY)
        icon_label.pack(side="left", padx=(0, 12))

        text_frame = tk.Frame(title_frame, bg=Theme.BG_PRIMARY)
        text_frame.pack(side="left", fill="x")

        title_label = tk.Label(text_frame,
                               text="Student Performance Predictor",
                               font=(Theme.FONT_FAMILY, 22, "bold"),
                               bg=Theme.BG_PRIMARY, fg=Theme.TEXT_PRIMARY)
        title_label.pack(anchor="w")

        subtitle_label = tk.Label(
            text_frame,
            text="Predict final grades using ML  •  Decision Tree Model  •  R² = 91.84%",
            font=(Theme.FONT_FAMILY, 10),
            bg=Theme.BG_PRIMARY, fg=Theme.TEXT_SECONDARY
        )
        subtitle_label.pack(anchor="w")

    # ── Input Section ────────────────────────────────────────────────────
    def _build_input_section(self):
        # Card container
        card_outer = tk.Frame(self.root, bg=Theme.BORDER, padx=1, pady=1)
        card_outer.pack(fill="x", padx=30, pady=(20, 0))

        card = tk.Frame(card_outer, bg=Theme.BG_CARD, padx=25, pady=20)
        card.pack(fill="both", expand=True)

        # Section title
        section_title = tk.Label(card, text="📊  Input Features",
                                 font=(Theme.FONT_FAMILY, 14, "bold"),
                                 bg=Theme.BG_CARD, fg=Theme.ACCENT_SECONDARY)
        section_title.pack(anchor="w", pady=(0, 5))

        section_desc = tk.Label(
            card,
            text="Enter the student's data below to predict their final grade (G3).",
            font=(Theme.FONT_FAMILY, 9),
            bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED
        )
        section_desc.pack(anchor="w", pady=(0, 18))

        # ── Feature 1: Number of past class failures (0-4) ──
        self.failures_var = tk.IntVar(value=0)
        self._build_feature_input(
            card,
            label="Failures",
            description="Number of past class failures (0 = none, 1, 2, 3, 4+)",
            variable=self.failures_var,
            min_val=0, max_val=4,
            icon="❌"
        )

        # Separator
        sep1 = tk.Frame(card, bg=Theme.BORDER, height=1)
        sep1.pack(fill="x", pady=12)

        # ── Feature 2: G1 — First period grade (0-20) ──
        self.g1_var = tk.IntVar(value=10)
        self._build_feature_input(
            card,
            label="G1 — First Period Grade",
            description="Grade in the first evaluation period (0 to 20)",
            variable=self.g1_var,
            min_val=0, max_val=20,
            icon="📝"
        )

        # Separator
        sep2 = tk.Frame(card, bg=Theme.BORDER, height=1)
        sep2.pack(fill="x", pady=12)

        # ── Feature 3: G2 — Second period grade (0-20) ──
        self.g2_var = tk.IntVar(value=10)
        self._build_feature_input(
            card,
            label="G2 — Second Period Grade",
            description="Grade in the second evaluation period (0 to 20)",
            variable=self.g2_var,
            min_val=0, max_val=20,
            icon="📋"
        )

    def _build_feature_input(self, parent, label, description, variable,
                              min_val, max_val, icon):
        """Build a single feature input row with label, slider, and value display."""
        frame = tk.Frame(parent, bg=Theme.BG_CARD)
        frame.pack(fill="x", pady=4)

        # Label row
        label_row = tk.Frame(frame, bg=Theme.BG_CARD)
        label_row.pack(fill="x")

        lbl = tk.Label(label_row, text=f"{icon}  {label}",
                       font=(Theme.FONT_FAMILY, 12, "bold"),
                       bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY)
        lbl.pack(side="left")

        # Value display (pill badge)
        value_lbl = tk.Label(label_row, text=str(variable.get()),
                             font=(Theme.FONT_FAMILY_MONO, 13, "bold"),
                             bg=Theme.ACCENT_PRIMARY, fg="white",
                             padx=14, pady=2)
        value_lbl.pack(side="right")

        # Description
        desc_lbl = tk.Label(frame, text=description,
                            font=(Theme.FONT_FAMILY, 9),
                            bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED)
        desc_lbl.pack(anchor="w", pady=(2, 6))

        # Slider
        slider = tk.Scale(
            frame,
            from_=min_val, to=max_val,
            orient="horizontal",
            variable=variable,
            showvalue=False,
            bg=Theme.BG_CARD,
            fg=Theme.TEXT_PRIMARY,
            troughcolor=Theme.BG_INPUT,
            activebackground=Theme.ACCENT_GLOW,
            highlightthickness=0,
            bd=0,
            sliderrelief="flat",
            length=590,
            sliderlength=22
        )
        slider.pack(fill="x")

        # Range labels
        range_frame = tk.Frame(frame, bg=Theme.BG_CARD)
        range_frame.pack(fill="x")
        tk.Label(range_frame, text=str(min_val),
                 font=(Theme.FONT_FAMILY, 8), bg=Theme.BG_CARD,
                 fg=Theme.TEXT_MUTED).pack(side="left")
        tk.Label(range_frame, text=str(max_val),
                 font=(Theme.FONT_FAMILY, 8), bg=Theme.BG_CARD,
                 fg=Theme.TEXT_MUTED).pack(side="right")

        # Update value label when slider changes
        def on_change(*args):
            value_lbl.config(text=str(variable.get()))
        variable.trace_add("write", on_change)

    # ── Predict Button ───────────────────────────────────────────────────
    def _build_predict_button(self):
        btn_frame = tk.Frame(self.root, bg=Theme.BG_PRIMARY)
        btn_frame.pack(fill="x", padx=30, pady=20)

        self.predict_btn = tk.Button(
            btn_frame,
            text="✨  Predict Final Grade",
            font=(Theme.FONT_FAMILY, 14, "bold"),
            bg=Theme.ACCENT_PRIMARY,
            fg="white",
            activebackground=Theme.ACCENT_GLOW,
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            padx=30, pady=12,
            command=self._predict
        )
        self.predict_btn.pack(fill="x")

        # Hover effects
        self.predict_btn.bind("<Enter>",
                              lambda e: self.predict_btn.config(
                                  bg=Theme.ACCENT_GLOW))
        self.predict_btn.bind("<Leave>",
                              lambda e: self.predict_btn.config(
                                  bg=Theme.ACCENT_PRIMARY))

    # ── Result Section ───────────────────────────────────────────────────
    def _build_result_section(self):
        # Card container
        card_outer = tk.Frame(self.root, bg=Theme.BORDER, padx=1, pady=1)
        card_outer.pack(fill="x", padx=30, pady=(0, 0))

        self.result_card = tk.Frame(card_outer, bg=Theme.BG_SECONDARY,
                                     padx=25, pady=20)
        self.result_card.pack(fill="both", expand=True)

        # Placeholder text
        self.result_placeholder = tk.Label(
            self.result_card,
            text="🔮  Click 'Predict Final Grade' to see the prediction result",
            font=(Theme.FONT_FAMILY, 11),
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED
        )
        self.result_placeholder.pack(pady=10)

        # Hidden result widgets (shown after prediction)
        self.result_content = tk.Frame(self.result_card, bg=Theme.BG_SECONDARY)

        # Result title
        self.result_title = tk.Label(
            self.result_content, text="📈  Prediction Result",
            font=(Theme.FONT_FAMILY, 14, "bold"),
            bg=Theme.BG_SECONDARY, fg=Theme.ACCENT_SECONDARY
        )
        self.result_title.pack(anchor="w", pady=(0, 10))

        # Big grade display
        self.grade_frame = tk.Frame(self.result_content, bg=Theme.BG_SECONDARY)
        self.grade_frame.pack(fill="x", pady=5)

        self.grade_label = tk.Label(
            self.grade_frame, text="—",
            font=(Theme.FONT_FAMILY, 48, "bold"),
            bg=Theme.BG_SECONDARY, fg=Theme.SUCCESS
        )
        self.grade_label.pack(side="left")

        self.grade_unit = tk.Label(
            self.grade_frame, text="/ 20",
            font=(Theme.FONT_FAMILY, 20),
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_MUTED
        )
        self.grade_unit.pack(side="left", padx=(5, 0), anchor="s", pady=(0, 10))

        # Performance tier
        self.tier_label = tk.Label(
            self.grade_frame, text="",
            font=(Theme.FONT_FAMILY, 13, "bold"),
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY,
            padx=16, pady=4
        )
        self.tier_label.pack(side="right", anchor="s", pady=(0, 15))

        # Details
        self.details_label = tk.Label(
            self.result_content, text="",
            font=(Theme.FONT_FAMILY, 10),
            bg=Theme.BG_SECONDARY, fg=Theme.TEXT_SECONDARY,
            justify="left"
        )
        self.details_label.pack(anchor="w", pady=(5, 0))

    # ── Footer ───────────────────────────────────────────────────────────
    def _build_footer(self):
        footer = tk.Frame(self.root, bg=Theme.BG_PRIMARY)
        footer.pack(fill="x", padx=30, pady=(15, 15))

        tk.Label(
            footer,
            text="Powered by scikit-learn  •  Decision Tree Regressor  •  Trained on 100K student records",
            font=(Theme.FONT_FAMILY, 8),
            bg=Theme.BG_PRIMARY, fg=Theme.TEXT_MUTED
        ).pack()

    # ── Prediction Logic ─────────────────────────────────────────────────
    def _predict(self):
        failures = self.failures_var.get()
        g1 = self.g1_var.get()
        g2 = self.g2_var.get()

        try:
            # Model expects: [failures, G1, G2]
            import numpy as np
            features = np.array([[failures, g1, g2]])
            prediction = self.model.predict(features)[0]

            # Clamp prediction to 0-20 range
            prediction = max(0, min(20, prediction))
            prediction_rounded = round(prediction, 1)

            # Determine performance tier
            tier, tier_color = self._get_tier(prediction)

            # Update UI
            self.result_placeholder.pack_forget()
            self.result_content.pack(fill="x")

            self.grade_label.config(text=f"{prediction_rounded}",
                                     fg=tier_color)
            self.tier_label.config(text=tier, bg=tier_color, fg="white")

            self.details_label.config(
                text=f"Input:  Failures = {failures}  |  G1 = {g1}  |  G2 = {g2}\n"
                     f"The model predicts this student will score approximately "
                     f"{prediction_rounded} out of 20 in their final exam."
            )

        except Exception as e:
            messagebox.showerror("Prediction Error",
                                 f"An error occurred during prediction:\n{e}")

    def _get_tier(self, grade):
        """Return a performance tier label and color based on the grade."""
        if grade >= 16:
            return "⭐ Excellent", "#00c896"
        elif grade >= 14:
            return "🟢 Very Good", "#26de81"
        elif grade >= 12:
            return "🔵 Good", "#4fc3f7"
        elif grade >= 10:
            return "🟡 Satisfactory", "#ffa726"
        elif grade >= 8:
            return "🟠 Below Average", "#ff7043"
        else:
            return "🔴 Needs Improvement", "#ff5252"


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = StudentPerformanceApp(root)
    root.mainloop()
