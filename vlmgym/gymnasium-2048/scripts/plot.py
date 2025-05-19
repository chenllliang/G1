import argparse
import re

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("ggplot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-path",
        required=True,
        help="path to input log file",
    )
    parser.add_argument(
        "-t",
        "--title",
        help="figure title",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        help="path to output png file",
    )
    args = parser.parse_args()
    return args


def read_metrics(filename: str) -> pd.DataFrame:
    data = []

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            if "mean score" not in line:
                continue

            episode, winning_rate, mean_score, max_tile = re.search(
                (
                    r"episode (\d+): winning rate = (\d+\.\d+), "
                    r"mean score = (\d+\.\d+), max tile = (\d+)"
                ),
                line,
            ).groups()

            data.append(
                {
                    "episode": int(episode),
                    "winning_rate": float(winning_rate),
                    "mean_score": float(mean_score),
                    "max_tile": int(max_tile),
                }
            )

    return pd.DataFrame(data)


def plot() -> None:
    args = parse_args()
    metrics = read_metrics(filename=args.input_path)
    metrics.set_index("episode", inplace=True)
    axs = metrics.plot(subplots=True, title=args.title, grid=True)
    fig = axs[0].figure
    fig.tight_layout()
    fig.show()
    if args.output_path is not None:
        fig.savefig(args.output_path)


if __name__ == "__main__":
    plot()
