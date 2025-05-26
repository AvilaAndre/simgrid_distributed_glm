from simgrid import ActivitySet, Engine, Mailbox, this_actor
from dataclasses import dataclass
import sys
import torch
from torch.types import Tensor

from linear_model import LinearModel
from .messages import CoefficientsMsg


# TODO: Remove this
@dataclass
class FlowUpdatingMsg:
    sender: str
    flow: float
    estimate: float

    def size(self) -> int:
        return (
            sys.getsizeof(self)
            + sys.getsizeof(self.sender)
            + sys.getsizeof(self.flow)
            + sys.getsizeof(self.estimate)
        )


@dataclass
class LMConcatMessage:
    origin: str
    r_remote: Tensor


@dataclass
class LMState:
    model: LinearModel
    r_remotes: dict[str, Tensor]
    nodes: list[str]


class LM:
    TICK_INTERVAL: float = 100.0
    _next_id = 0

    def next_name() -> str:
        name = f"LinearModel_{LM._next_id}"
        LM._next_id += 1
        return name

    def __init__(self, name: str, x: torch.Tensor, y: torch.Tensor):
        self.name: str = name

        this_actor.on_exit(
            lambda killed: this_actor.info(
                "Exiting now (killed)." if killed else "Exiting now (finishing)."
            )
        )

        model: LinearModel = LinearModel.fit(x, y)

        self.state: LMState = LMState(model, {}, [])

        # setup mailbox
        self.mailbox = Mailbox.by_name(self.name)
        self.pending_comms = ActivitySet()

        self.run()

    def run(self):
        this_actor.info(f"LinearModel {self.name} started.")

        self.start()

        msgs_to_rcv = len(self.state.nodes)

        while msgs_to_rcv > 0:
            self.receive_concat_r_msg(self.mailbox.get())
            msgs_to_rcv -= 1

        # Sending coefficients to aggregator
        Mailbox.by_name("LMAggregator").put(
            CoefficientsMsg(self.state.model.coefficients), 0
        )  # TODO: Add message size

    def start(self):
        nodes_filtered = []

        for actor_name in [actor.name for actor in Engine.instance.all_actors]:
            if actor_name != self.name and actor_name.split("_")[0] == "LinearModel":
                self.send_concat_r(actor_name)
                nodes_filtered.append(actor_name)

        self.state.nodes = nodes_filtered

    def send_concat_r(self, target):
        msg = LMConcatMessage(self.name, self.state.model.r_local)
        Mailbox.by_name(target).put_async(msg, 0)  # TODO: Add message size

    def receive_concat_r_msg(self, msg: LMConcatMessage):
        sender, r_remote = msg.origin, msg.r_remote

        if sender in self.state.r_remotes.keys():
            return
        else:
            self.state.r_remotes[sender] = r_remote

            if len(self.state.nodes) == len(self.state.r_remotes.keys()):
                self.state.model = LinearModel.update_distributed(
                    self.state.model,
                    torch.cat(list(self.state.r_remotes.values()), dim=0),
                )

                # TODO: Signal that simulation is done
                # {:ok, parent} = Registry.meta(Registry.Simulation, :parent)
                # send(parent, :simulation_is_done)
