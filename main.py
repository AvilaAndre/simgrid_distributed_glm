from simgrid import Actor, Engine, Host, NetZone, this_actor
from collections import defaultdict
import numpy as np
import torch
import sys
import os
import csv

from simulation.LM import LM
from linear_model import LinearModel

global_values = defaultdict(dict)


def print_global_values():
    for key in global_values.keys():
        this_actor.info(f"{key}{global_values[key]}")


def watcher(run_until: float, time_interval: float = 10.0):
    while Engine.clock < run_until:
        this_actor.sleep_for(min(time_interval, run_until - Engine.clock))
        print_global_values()

    this_actor.info(f"{list(Engine.instance.all_actors)}")
    this_actor.info(f"{list(Engine.instance.all_hosts)}")
    this_actor.info("Killing every actor but myself.")
    this_actor.info(f"{list(Engine.instance.all_actors)}")
    Actor.kill_all()


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


def linear_model(n, data, central_lm):
    nodes = start_run(n, data, LM)
    central_lm = LinearModel.fit(
        torch.tensor(data["x"], dtype=torch.float64),
        torch.tensor(data["y"], dtype=torch.float64),
    )
    # check(central_lm, nodes)


def check(central, nodes):
    res = all(
        np.allclose(node.model.coefficients, central.coefficients) for node in nodes
    )

    print(
        "check",
        [np.allclose(node.model.coefficients, central.coefficients) for node in nodes],
        res,
    )

    return res, central.coefficients


def start_run(n, data, module):
    y_len = len(data["y"])
    ncols = len(data["x"][0])

    assert len(data["x"]) == y_len, "len(x) != len(y)"
    assert n * (ncols + 1) < y_len, "split > ncols"

    data_x = torch.tensor(data["x"], dtype=torch.float64)
    data_y = torch.tensor(data["y"], dtype=torch.float64)

    actors = []
    for x, y in zip(chunk_nx(data_x, n), chunk_nx(data_y, n)):
        # INFO: this is where simulation nodes are started
        actor_name = LM.next_name()

        Engine.instance.add_actor(
            actor_name, Host.by_name("Observer"), LM, actor_name, x, y
        )
        actors.append(actor_name)

    return actors


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


    e.add_actor("watcher", Host.by_name("Observer"), watcher, 1000.0, 10.0)

    n = 7

    for m in ["lm"]:  # FIXME: add "glm"
        data = model_data(m)
        beta = {"coefficients": torch.tensor(model_beta(m), dtype=torch.float64)}

        # experiment_result = experiment(m).(n, data, beta)

        if m == "lm":
            linear_model(n, data, beta)

    e.run()
    """


    e.register_actor("peer", Peer)
    e.load_deployment("./actors.xml")

    e.netzone_root.add_host("observer", 25e6)

    # Add a watcher of the changes
    Actor.create("watcher", Host.by_name("observer"), watcher, 1000.0, 10.0)

    random.seed(200)

    e.run_until(10000)

    this_actor.info("Simulation finished")

    """
