from simgrid import Engine, Mailbox, this_actor
import torch

from linear_model import LinearModel
from .dataclasses import LMState, LMConcatMessage, ModelCoefficients


class LM:
    _next_id = 0
    model_name = "lm"

    def next_name() -> str:
        name = f"LinearModel_{LM._next_id}"
        LM._next_id += 1
        return name

    def __init__(
        self, name: str, aggregator_name: str, x: torch.Tensor, y: torch.Tensor
    ):
        this_actor.on_exit(
            lambda killed: this_actor.info(
                "Exiting now (killed)." if killed else "Exiting now (finishing)."
            )
        )

        self.name: str = name
        self.state = LMState(LinearModel.fit(x, y), {}, [])
        # setup mailbox
        self.mailbox = Mailbox.by_name(self.name)
        self.aggregator_mb = Mailbox.by_name(aggregator_name)

        self.run()

    def run(self):
        this_actor.info(f"Actor {self.name} started.")

        self.broadcast_concat_r()

        # Keeps waiting for message until killed
        while True:
            self.receive_concat_r_msg(self.mailbox.get())

    def broadcast_concat_r(self):
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

        if sender not in self.state.r_remotes.keys():
            self.state.r_remotes[sender] = r_remote

            if len(self.state.nodes) == len(self.state.r_remotes.keys()):
                self.state.model = LinearModel.update_distributed(
                    self.state.model,
                    torch.cat(list(self.state.r_remotes.values()), dim=0),
                )

                self.send_coefficients_and_exit()

    def send_coefficients_and_exit(self):
        self.aggregator_mb.put(
            ModelCoefficients(self.state.model.coefficients), 0
        )  # TODO: Add message size

        # kill actor
        this_actor.exit()
