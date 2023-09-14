import os

from experiment.presets.bias import bias_names
from util.batch import batch_load_eval
from util.plot import plot_target_acc_box, print_acc_stats

PREFIX = "v6"

if __name__ == "__main__":
    """Generate a plot for each of the shift types and methods, displaying box plots with accuracy per configuration."""

    # For each shift type and amount
    for bias in bias_names:
        store_path = os.path.join(os.getcwd(), '../results', PREFIX, f"{bias}_val")
        results = batch_load_eval(store_path)

        # iterate over methods
        results = results.groupby('identifier')
        for model, data in results:
            file_name = f"{bias}_{model}"
            plot_path = os.path.join(os.getcwd(), '../results', PREFIX, file_name)

            # Format display title
            bias_formatted = ' '.join([w.capitalize() for w in bias.split('_')])
            model = str(model).replace("$", "$^\\ast$")
            title = f"{bias_formatted} - {model}"

            # Plot and save
            plot_target_acc_box(data, title, save=plot_path)

            # print median to terminal
            print(title)
            print_acc_stats(data)
            print()
