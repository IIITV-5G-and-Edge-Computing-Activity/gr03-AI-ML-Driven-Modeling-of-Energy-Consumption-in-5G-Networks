# AI/ML for 5G-Energy Consumption Modeling

This repository contains the code and methodology to address the energy consumption modeling in 5G networks. The goal is to design a machine learning model to accurately estimate and optimize energy consumption in base stations under varying configurations and conditions.

---

## Problem Statement

5G networks provide enhanced services but at the cost of increased energy consumption due to higher processing requirements, denser deployments, and wider bandwidths. To optimize energy usage, this challenge focuses on:

1. Modeling energy consumption based on architecture, configurations, traffic, and energy-saving methods.
2. Generalizing energy consumption predictions across:
   - Different base station products.
   - New base station configurations.

---

## Objectives

### **Objective A:** 
Develop a model to estimate energy consumption of base station products, considering:
- Engineering configurations.
- Traffic conditions.
- Energy-saving methods.

### **Objective B:** 
Generalize the model across **new base station products** using data from existing products.

### **Objective C:** 
Generalize the model for **new configurations** with minimal additional data.

---

## Metrics

The model performance is evaluated using the **Weighted Mean Absolute Percentage Error (WMAPE)**, where weights emphasize new devices or configurations in the test set. The final ranking is based on minimizing WMAPE.

---

## Key Learnings

- Tackling a regression problem for energy consumption prediction.
- Understanding domain knowledge in 5G networks and energy consumption patterns.
- Designing and training ANN architectures with optimal activation functions.
- Optimizing MAE, MAPE, and WMAPE metrics.
- Ensuring no future data leakage in predictions.

---

## Environment Setup

### **Kaggle Environment**
- **Accelerator:** GPU P100
- **Language:** Python
- **Persistence:** No persistence
- **Environment:** Pin to original environment (2023-09-07)
- **Internet Access:** Enabled

Ensure the above settings in the Kaggle environment for reproducibility and to avoid memory issues.

### **Packages**
Install the following dependencies:

```bash
pip install numpy==1.23.5
pip install pandas==2.0.3
pip install fastai==2.7.12
pip install plotly==5.17.0
pip install plotnine==0.12.3
pip install fastinference==0.0.36
pip install scipy==1.7.3
pip install sklearn==1.2.2
pip install torch==2.0.0
pip install tensorflow==2.12.0
pip install shap==0.42.1
```

## Steps to Reproduce

Follow the steps below to reproduce the results:

1. **Clone this repository:**
   ```bash
   git clone https://github.com/AyushPathak2610/AI-ML-Driven-Modeling-of-Energy-Consumption-in-5G-Networks.git
   cd AI-ML-Driven-Modeling-of-Energy-Consumption-in-5G-Networks
   ```
2. **Set up the Kaggle environment** as described in the environment setup section.

3. **Install the required dependencies** using the provided commands.

4. **Run the provided notebooks/scripts** in the Kaggle GPU environment.

5. **Analyze the results** and evaluate performance based on **WMAPE (Weighted Mean Absolute Percentage Error)**.

---

## Methodology

The solution follows these steps:

1. **Data Preprocessing:**
   - Handle missing values.
   - Normalize data for consistency.
   - Engineer features for base station configurations and energy-saving methods.

2. **Model Architecture:**
   - Design an Artificial Neural Network (ANN) for energy consumption prediction.
   - Optimize the model for cross-device and cross-configuration generalization.

3. **Evaluation:**
   - Use **WMAPE** to assess the model’s accuracy and generalization capabilities.

4. **Optimization:**
   - Experiment with hyperparameters.
   - Explore different activation functions.
   - Refine data sampling methods to minimize WMAPE.

---

## Results and Insights

- **Objective A:** Accurate estimation of energy consumption for existing base station products.
- **Objective B:** Effective generalization to new base station products.
- **Objective C:** Robust predictions for new configurations.

Detailed results are available in the `results/` directory.

---

## Team Contributions

1. **Shreeyansh Sohala:** Datasets preparation  
2. **Shivam Pawar:** Feature Engineering  
3. **Ayush Pathak:** Worked on the Keras ANN model  
4. **Kshitij Tiwari:** Worked on the Fast AI ANN model  
5. **Aniyeshu Verma:** Conclusion and Interpretation  

