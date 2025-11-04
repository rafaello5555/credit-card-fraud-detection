import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def run_eda(data):
    stats = {
        "num_rows": len(data),
        "num_columns": len(data.columns),
        "head": data.head(),
        "class_labels": data.Class.unique(),
        "class_counts": data.Class.value_counts().values,
        "amount_min": np.min(data.Amount.values),
        "amount_max": np.max(data.Amount.values),
        "amount_90pct": np.percentile(data.Amount.values, 90)
    }

    # Pie chart
    fig1, ax1 = plt.subplots()
    ax1.pie(stats["class_counts"], labels=stats["class_labels"], autopct='%1.3f%%')
    ax1.set_title('Target Variable Value Counts')
    stats["pie_chart"] = fig1

    # Histogram
    fig2, ax2 = plt.subplots()
    ax2.hist(data.Amount.values, bins=20, color='green')
    stats["histogram"] = fig2

    return stats



