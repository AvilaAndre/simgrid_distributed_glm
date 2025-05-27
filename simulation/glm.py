from simgrid import Engine, Mailbox, this_actor
import torch
from torch.types import Tensor

from family import FamilyEnum
from generalized_linear_model import GeneralizedLinearModel
from .dataclasses import (
    GLMState,
    GLMSumRowsMessage,
    GLMConcatMessage,
    ModelCoefficients,
)


class GLM:
    _next_id = 0
    model_name = "glm"

    def next_name() -> str:
        name = f"GeneralizedLinearModel_{GLM._next_id}"
        GLM._next_id += 1
        return name

    def __init__(self, name: str, aggregator_name: str, x: Tensor, y: Tensor):
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

        # allow graceful exit
        self.finished = False

        self.run()

    def run(self):
        this_actor.info("Started.")

        self.broadcast_sum_rows()

        # Keeps waiting for message until killed
        while not self.finished:
            msg_received = self.mailbox.get()

            if isinstance(msg_received, GLMSumRowsMessage):
                self.receive_sum_rows_msg(msg_received)
            elif isinstance(msg_received, GLMConcatMessage):
                if not self.state.finished:
                    self.receive_concat_r_msg(msg_received)
            else:
                this_actor.warning(
                    f"Received invalid message of type {type(msg_received)}"
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

        if sender not in self.state.r_remotes.keys():
            self.state.r_remotes[sender] = nrows

            if len(self.state.nodes) == len(self.state.r_remotes.keys()):
                self.state.total_nrow += sum(self.state.r_remotes.values())
                self.broadcast_nodes()
                self.state.r_remotes = {}

    def receive_concat_r_msg(self, msg: GLMConcatMessage):
        sender, r_remote, iter = msg.origin, msg.r_remote, msg.iter

        if iter not in self.state.r_remotes.keys():
            self.state.r_remotes[iter] = {}

        self.handle_iter(sender, r_remote, iter)

    def handle_iter(self, sender: str, r_remote: Tensor, iter: int):
        if iter not in self.state.r_remotes or sender in self.state.r_remotes[iter]:
            return
        else:
            self.state.r_remotes[iter][sender] = r_remote

            if iter == self.state.model.iter and len(self.state.nodes) == len(
                self.state.r_remotes[iter].keys()
            ):
                r_local_with_all_r_remotes = torch.cat(
                    [self.state.model.r_local]
                    + list(self.state.r_remotes[iter].values())
                )

                r_local, beta, stop = (
                    GeneralizedLinearModel.distributed_binomial_single_solve_n(
                        r_local_with_all_r_remotes,
                        self.state.model.coefficients,
                        self.state.total_nrow,
                        GeneralizedLinearModel.default_maxit,
                        GeneralizedLinearModel.default_tol,
                        self.state.model.iter,
                    )
                )

                self.state.model.r_local = r_local
                self.state.model.coefficients = beta
                self.state.model.iter += 1

                self.state.finished = stop

                if self.state.finished:
                    self.send_coefficients_and_exit()
                    return
                else:
                    self.state.model.r_local = (
                        GeneralizedLinearModel.distributed_binomial_single_iter_n(
                            self.state.data[0],
                            self.state.data[1],
                            beta,
                        )
                    )

                    self.broadcast_nodes()

    def send_coefficients_and_exit(self):
        # no need for message size as it is a message for the aggregator
        self.aggregator_mb.put(ModelCoefficients(self.state.model.coefficients), 0)
        self.finished = True
