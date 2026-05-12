# 🎨 AI Evaluator for Modern UI Color Harmony

![Version](https://img.shields.io/badge/Version-1.0--rc.1-blue?style=for-the-badge) ![ReleaseType](https://img.shields.io/badge/Release%20Candidate-yellowgreen?style=for-the-badge)
![Lang](https://img.shields.io/badge/Language-Python%20(3.9+)-blue?style=for-the-badge) ![License](https://img.shields.io/badge/License-AGPL--3.0-important?style=for-the-badge)

A machine learning model that utilizes XGBoost Regression to evaluate the harmony of 5-color palettes based on the **"Modern UI Rule" (Analogous + Accent)**.

---

## 🎯 Overview

### The Challenge: Solving the "Curse of Dimensionality"

Traditional color harmony models often fail because of the sheer scale of the search space. A 5-color palette using 24-bit HEX codes results in approximately $(16.7 \times 10^6)^5 \approx 1.3 \times 10^{36}$ possible combinations. Attempting to map raw RGB/HEX values directly to a subjective "harmony score" with a limited dataset leads to severe **underfitting**.

### The Solution: Feature Engineering

Instead of forcing the AI to learn from raw pixels, this project transforms color data into **Perceptual UI Features**:

- **WCAG 2.1 Contrast Ratios:** Evaluating accessibility between Background, Primary, and Accent colors.
- **$\Delta E_{ab}^*$ (Delta E):** Measuring visual distance in the **CIELAB color space** to detect "Dead Palettes" or overlapping roles.
- **Hue Angular Distance:** Calculating the mathematical relationship between Primary and Secondary colors to ensure Analogous harmony.

---

## ✨ Key Features

- **AI Scoring Engine:** Predicts a harmony score (1.0 - 10.0) using a trained XGBoost model.
- **Palette Health Check:** A Radar Chart visualizing five dimensions: Accessibility, Harmony, Diversity, Vibrancy, and Layering.
- **Dynamic Visual Feedback:** A custom Gauge Chart that changes color (Red → Yellow → Green) based on the AI's confidence in your palette.
- **Glassmorphism UI:** A high-end Streamlit dashboard designed with modern web aesthetics.
- **Automated Data Synthesizer:** Includes a script to generate 5,000+ samples with "Hard Negatives" (common design mistakes) to robustly train the AI.

*Note: This project currently uses mixed English/Vietnamese text in code and UI.*

## 🛠️ Tech Stack

- **Core:** Python 3.9+
- **Machine Learning:** XGBoost, Scikit-learn, Pandas, Numpy
- **Color Science:** Colorsys (HSV conversion), CIELAB transformation logic
- **Visualization:** Plotly (Radar/Gauge charts), Seaborn/Matplotlib (Evaluation)
- **Frontend:** Streamlit with custom CSS (Glassmorphism & Glow effects)

---

## 🚀 Installation & Running

**1. Clone the repository**

```bash
git clone https://github.com/chairman-q/AI-UI-color-harmony.git
```

**2. Create virtual environment**

```bash
python -m venv .venv
```

**3. Activate virtual environment:**

- Windows (PowerShell)

```bash
.venv\Scripts\Activate.ps1
```

- macOS/Linux

```bash
source .venv/bin/activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

**5. Run the app**

```bash
streamlit run app.py
```

---

## 🧪 Model Training & Evaluation (Optional)

If you wish to re-generate the dataset or re-train the "UI Brain" from scratch:

1. **Synthesize Data:** Creates `data_modern_ui.csv` with 5,000 samples.

```bash
python scripts/data_synthesizer.py
```

2. **Train Model:** Performs 5-Fold Cross-Validation and saves the model to `models/saved_models/`.

```bash
python scripts/train_models.py
```

3. **Evaluate:** Generates Regression Plots and Feature Importance charts in the `visualize/` folder.

```bash
python visualize/evaluate_models.py
```

---

## 📂 Project Structure

```
ML4D_UICOLORHARMONY
├── data/raw/               # Generated datasets (CSV)
├── models/saved_models/    # Trained XGBoost models (.json)
├── scripts/
│   ├── data_synthesizer.py # Logic for synthetic data & hard negatives
│   └── train_models.py     # Training & Feature Engineering pipeline
├── visualize/
│   ├── evaluate_models.py  # Model validation & importance plotting
│   └── *.png               # Exported visual reports
├── app.py                  # Streamlit Glassmorphism Dashboard
├── README.md               # Project documentation
└── requirements.txt        # List of dependencies
```

---

## 🌐 Deployment

This project is being deployed at [uiharmony.streamlit.app](https://uiharmony.streamlit.app).

---

## 📜 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

*Note: AGPL-3.0 is chosen to ensure that any improvements made to the core AI scoring engine by the community remain open-source, even if the software is hosted as a web service.*