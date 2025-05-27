from simgrid import Engine, Host
import argparse
import sys
import os
import csv
import torch
from torch.types import Tensor

from simulation.lm import LM
from simulation.glm import GLM
from simulation.dataclasses import ModelCoefficients
from simulation.aggregator import aggregator


def model_data(m: str) -> dict[str, list[float]]:
    path = os.path.abspath(f"./{m}_mm.csv")

    x_data = []
    y_data = []

    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header

        for row in reader:
            float_row = [float(num) for num in row]

            *x, y = float_row
            x_data.append(x)
            y_data.append([y])

    return {"x": x_data, "y": y_data}


def model_beta(m: str) -> list[list[float]]:
    path = os.path.abspath(f"./{m}_beta.csv")

    result: list[list[float]] = []

    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header

        for row in reader:
            result.append([float(row[0])])

    return result


def model_run(m: LM | GLM, n: int):
    data: dict[str, list[float]] = model_data(m.model_name)
    beta = ModelCoefficients(
        torch.tensor(model_beta(m.model_name), dtype=torch.float64)
    )

    y_len = len(data["y"])
    ncols = len(data["x"][0])

    assert len(data["x"]) == y_len, "len(x) != len(y)"
    assert n * (ncols + 1) < y_len, "split > ncols"

    data_x: Tensor = torch.tensor(data["x"], dtype=torch.float64)
    data_y: Tensor = torch.tensor(data["y"], dtype=torch.float64)

    aggregator_name = f"{m.__name__}Aggregator"

    for x, y in zip(chunk_nx(data_x, n), chunk_nx(data_y, n)):
        # INFO: this is where simulation nodes are started
        actor_name: str = m.next_name()

        Engine.instance.add_actor(
            actor_name,
            Host.by_name("Observer"),
            m,
            actor_name,
            aggregator_name,
            x,
            y,
        )

    # TODO: Add actors names which should be
    # sending message to know when to stop.
    e.add_actor(
        aggregator_name, Host.by_name("Observer"), aggregator, aggregator_name, n, beta
    )


def chunk_nx(mat: Tensor, n: int) -> list[Tensor]:
    if n == 1:
        return [mat]

    nsplits = mat.shape[0] // n

    chunks = []
    mat_remaining = mat

    for _ in range(n - 1):
        chunk = mat_remaining[:nsplits]
        mat_remaining = mat_remaining[nsplits:]
        chunks.append(chunk)

    chunks.append(mat_remaining)

    return chunks


def positive_int(value: str):
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"Invalid value for -n: {value}. Must be ≥ 1.")
    return ivalue


def parse_args():
    parser = argparse.ArgumentParser(
        description="Choose the model to run the simulation."
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["lm", "glm", "both"],
        default="both",
        help="Choose from: lm, glm, both (default: both)",
        required=False,
    )

    parser.add_argument(
        "-n",
        "--n",
        type=positive_int,
        default=7,
        help="The number of actors (default: 7)",
        required=False,
    )

    return parser.parse_args()


if __name__ == "__main__":
    e = Engine(sys.argv)
    e.load_platform("./obs_platform.xml")

    args = parse_args()

    match args.model:
        case "lm":
            model_run(LM, args.n)
        case "glm":
            model_run(GLM, args.n)
        case _:
            for m in [LM, GLM]:
                model_run(m, args.n)

    e.run()
