from simgrid import Engine, Mailbox, this_actor
import torch

from family import FamilyEnum
from generalized_linear_model import GeneralizedLinearModel
from .dataclasses import GLMState, GLMSumRowsMessage, GLMConcatMessage


class GLM:
    _next_id = 0

    def next_name() -> str:
        name = f"GeneralizedLinearModel_{GLM._next_id}"
        GLM._next_id += 1
        return name

    def __init__(
        self, name: str, aggregator_name: str, x: torch.Tensor, y: torch.Tensor
    ):
        this_actor.on_exit(
            lambda killed: this_actor.info(
                "Exiting now (killed)." if killed else "Exiting now (finishing)."
            )
        )

        xtype = x.dtype
        r, c = x.shape
        beta = torch.zeros((c, 1), dtype=xtype)
        r_local = GeneralizedLinearModel.distributed_binomial_single_iter_n(x, y, beta)
        model = GeneralizedLinearModel(r_local, beta, FamilyEnum.BINOMIAL, 0)

        self.name: str = name
        self.state = GLMState(model, (x, y), {}, r, [], False)
        # setup mailbox
        self.mailbox = Mailbox.by_name(self.name)
        self.aggregator_mb = Mailbox.by_name(aggregator_name)

        self.run()

    def run(self):
        this_actor.info(f"Actor {self.name} started.")

        self.broadcast_sum_rows()

        # Keeps waiting for message until killed
        while True:
            msg_received = self.mailbox.get()

            if isinstance(msg_received, GLMSumRowsMessage):
                self.receive_sum_rows_msg(msg_received)
            elif isinstance(msg_received, GLMConcatMessage):
                # self.receive_concat_r_msg(msg_received)
                # TODO: this
                this_actor.warning("NOT YET IMPLEMENTED")
            else:
                this_actor.warning(
                    f"Actor {self.name} received invalid message of type {type(msg_received)}"
                )

    def broadcast_sum_rows(self):
        nodes_filtered = []

        for actor_name in [actor.name for actor in Engine.instance.all_actors]:
            if (
                actor_name != self.name
                and actor_name.split("_")[0] == "GeneralizedLinearModel"
            ):
                self.send_sum_rows(actor_name)
                nodes_filtered.append(actor_name)

        self.state.nodes = nodes_filtered

    def broadcast_nodes(self):
        for node in self.state.nodes:
            self.send_concat_r(node)

    def send_concat_r(self, target: str):
        msg = GLMConcatMessage(
            self.name, self.state.model.r_local, self.state.model.iter
        )
        Mailbox.by_name(target).put_async(msg, 0)  # TODO: Add message size

    def send_sum_rows(self, target):
        msg = GLMSumRowsMessage(self.name, self.state.total_nrow)
        Mailbox.by_name(target).put_async(msg, 0)  # TODO: Add message size

    def receive_sum_rows_msg(self, msg: GLMSumRowsMessage):
        sender, nrows = msg.origin, msg.nrows

        if sender in self.state.r_remotes.keys():
            return
        else:
            self.state.r_remotes[sender] = nrows

            if len(self.state.nodes) == len(self.state.r_remotes.keys()):
                self.state.total_nrow += sum(self.state.r_remotes.values())
                self.broadcast_nodes()
                self.state.r_remotes = {}
