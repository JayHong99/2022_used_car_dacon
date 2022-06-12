import shap
import matplotlib.pyplot as plt

def shap_explain(model, X, feature_names) : 
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    fig = shap.summary_plot(shap_values, X, feature_names, plot_size = (20, 30))
    plt.show()
    plt.savefig('SHAP.png')
