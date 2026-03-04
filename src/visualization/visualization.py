### Distribution des types de transaction
import matplotlib.pyplot as plt

def plot_transaction_types(df):
    df["type"].value_counts().plot(kind="bar")
    plt.title("Distribution des types de transaction")
    plt.xlabel("Type")
    plt.ylabel("Count")
    plt.show()