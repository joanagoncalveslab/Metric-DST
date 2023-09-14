import os

from util.batch import batch_generate
from experiment.presets.bias import bias_types

PREFIX = "v6"


def gen(builder, num, name):
    batch_path = os.path.join(os.getcwd(), '../results', PREFIX, name)
    batch_generate(builder, num, batch_path)


if __name__ == "__main__":
    """Generates one train and one validation dataset batch per bias type. Stored in the <PREFIX> subdirectory."""
    for bias in bias_types:
        gen(bias(), 10, bias.__name__)
        gen(bias(), 50, f"{bias.__name__}_val")
