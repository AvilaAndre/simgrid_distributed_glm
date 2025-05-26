from simgrid import Engine, Host, this_actor
import numpy as np
import sys
import os
import csv
import torch

from simulation.LM import LM
from simulation.dataclasses import ModelCoefficients
from simulation.aggregator import aggregator


def model_data(m):
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


def model_beta(m):
    path = os.path.abspath(f"./{m}_beta.csv")

    result = []

    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header

        for row in reader:
            result.append([float(row[0])])

    return result


def start_run(n, data, module):
    y_len = len(data["y"])
    ncols = len(data["x"][0])

    assert len(data["x"]) == y_len, "len(x) != len(y)"
    assert n * (ncols + 1) < y_len, "split > ncols"

    data_x = torch.tensor(data["x"], dtype=torch.float64)
    data_y = torch.tensor(data["y"], dtype=torch.float64)

    aggregator_name = "LMAggregator"

    for x, y in zip(chunk_nx(data_x, n), chunk_nx(data_y, n)):
        # INFO: this is where simulation nodes are started
        actor_name = LM.next_name()

        Engine.instance.add_actor(
            actor_name, Host.by_name("Observer"), LM, actor_name, aggregator_name, x, y
        )

    e.add_actor(
        aggregator_name, Host.by_name("Observer"), aggregator, aggregator_name, n, beta
    )


def chunk_nx(mat, n):
    if n == 1:
        return mat

    nsplits = mat.shape[0] // n

    chunks = []
    mat_remaining = mat

    for _ in range(n - 1):
        chunk = mat_remaining[:nsplits]
        mat_remaining = mat_remaining[nsplits:]
        chunks.append(chunk)

    chunks.append(mat_remaining)

    return chunks


if __name__ == "__main__":
    e = Engine(sys.argv)
    e.load_platform("./obs_platform.xml")
    # e.load_deployment("./actors.xml")

    n = 7

    for m in ["lm"]:  # FIXME: add "glm"
        data = model_data(m)

        beta = ModelCoefficients(torch.tensor(model_beta(m), dtype=torch.float64))

        # experiment_result = experiment(m).(n, data, beta)

        if m == "lm":
            start_run(n, data, LM)

    e.run()
