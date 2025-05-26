from simgrid import Actor, Mailbox, this_actor
from .dataclasses import ModelCoefficients
import numpy as np


def aggregator(name: str, n: int, central_lm: ModelCoefficients):
    mailbox = Mailbox.by_name(name)

    this_actor.on_exit(
        lambda killed: this_actor.info(
            "Exiting now (killed)." if killed else "Exiting now (finishing)."
        )
    )

    this_actor.info(f"{name} started")

    coefficient_msgs = []
    for i in range(n):
        msg = mailbox.get()

        if type(msg) is ModelCoefficients:
            coefficient_msgs.append(msg)

    check(central_lm, coefficient_msgs)

    Actor.kill_all()


def check(central: ModelCoefficients, coefficients_msgs: list[ModelCoefficients]):
    res = all(
        np.allclose(msg.coefficients, central.coefficients) for msg in coefficients_msgs
    )

    this_actor.info(f"Are the coefficients from every peer equal to central's? {res}")
