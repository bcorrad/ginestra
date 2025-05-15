import matplotlib.pyplot as plt
import optuna.visualization as vis
def optuna_plot(study, filename='optuna_plot.png'):
    fig = vis.plot_optimization_history(study)
    fig.write_image(filename)
    print(f"Plot saved to {filename}")