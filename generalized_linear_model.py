import torch
from torch.types import Tensor
from typing import TypeVar

from family import FamilyEnum, Gaussian, Binomial

T = TypeVar("T", bound="GeneralizedLinearModel")


class GeneralizedLinearModel:
    default_maxit = 25
    default_tol = 1.0e-10

    def __init__(
        self,
        r_local: Tensor,
        coefficients: Tensor,
        family: FamilyEnum,
        iter: int,
    ):
        self.r_local: Tensor = r_local
        self.coefficients: Tensor = coefficients
        self.family: FamilyEnum = family
        self.iter: int = iter

    @classmethod
    def fit(cls: type[T], x: Tensor, y: Tensor, opts: dict[str, any] = {}) -> T:
        opts.setdefault("family", FamilyEnum.GAUSSIAN)
        opts.setdefault("maxit", cls.default_maxit)
        opts.setdefault("tol", cls.default_tol)

        r_local: Tensor
        beta: Tensor
        iter: int

        match opts["family"]:
            case FamilyEnum.GAUSSIAN:
                r_local, beta, iter = cls.fit_gaussian_n(x, y, opts)
            case FamilyEnum.BINOMIAL:
                r_local, beta, iter = cls.fit_binomial_n(x, y, opts)
            case _:
                print(f"Value {opts['family']} is invalid for GLM family")
                opts["family"] = FamilyEnum.GAUSSIAN
                r_local, beta, iter = cls.fit_gaussian_n(x, y, opts)

        return cls(r_local, beta, opts["family"], iter)

    @classmethod
    def fit_gaussian_n(
        cls,
        x: Tensor,
        y: Tensor,
        opts: dict[str, any],
    ) -> tuple[Tensor, Tensor, int]:
        maxit = opts["maxit"]
        tol = opts["tol"]

        xtype = x.dtype
        _, c = x.shape
        beta = torch.zeros((c, 1), dtype=xtype)
        r_local = torch.zeros((c + 1, c + 1), dtype=xtype)
        stop = False
        iter = 1

        while not stop:
            eta = x @ beta
            mu = Gaussian.linkinv(eta)
            dmu = Gaussian.mu_eta(eta)

            z = eta + (y - mu) / dmu
            w = (dmu**2) / Gaussian.variance(mu)

            x_tilde = torch.sqrt(w) * x
            z_tilde = torch.sqrt(w) * z

            beta_old = beta

            r_local, beta = cls.ols_n(torch.cat([x_tilde, z_tilde], dim=1))

            vcov = cls.vcov(r_local, Gaussian, x.shape[0])
            delta = (beta_old - beta) / torch.sqrt(torch.diagonal(vcov))
            diff = torch.abs(delta).max()
            stop = cls.stop(maxit, tol, iter, diff)

            iter = iter if stop else iter + 1

            if not stop:
                iter += 1

        return r_local, beta, iter

    @classmethod
    def fit_binomial_n(
        cls,
        x: Tensor,
        y: Tensor,
        opts: dict[str, any],
    ) -> tuple[Tensor, Tensor, int]:
        maxit = opts["maxit"] - 1
        tol = opts["tol"]

        xtype = x.dtype
        _, c = x.shape
        beta = torch.zeros((c, 1), dtype=xtype)
        r_local = torch.zeros((c + 1, c + 1), dtype=xtype)
        stop = False
        iter = 1

        while not stop:
            eta = x @ beta
            mu = Binomial.linkinv(eta)
            dmu = Binomial.mu_eta(eta)

            z = eta + (y - mu) / dmu
            w = (dmu**2) / Binomial.variance(mu)

            x_tilde = torch.sqrt(w) * x
            z_tilde = torch.sqrt(w) * z

            beta_old = beta

            r_local, beta = cls.ols_n(torch.cat([x_tilde, z_tilde], dim=1))

            vcov = cls.vcov(r_local, Binomial, x.shape[0])
            delta = (beta_old - beta) / torch.sqrt(torch.diagonal(vcov))
            diff = torch.abs(delta).max()
            stop = cls.stop(maxit, tol, iter, diff)

            iter = iter if stop else iter + 1

            if not stop:
                iter += 1

        return r_local, beta, iter

    @classmethod
    def ols_n(cls, r_xy_or_xy: Tensor) -> tuple[Tensor, Tensor]:
        _, r_s = torch.linalg.qr(r_xy_or_xy)

        r = r_s[:-1, :-1].clone()
        theta = r_s[:-1, -1:].clone()

        b: Tensor = torch.linalg.solve_triangular(r, theta, upper=True)

        return (r_s, b)

    @classmethod
    def vcov(
        cls,
        r_local: Tensor,
        family: [Gaussian | Binomial],
        total_nrow: int,
    ) -> Tensor:
        r = r_local[:-1, :-1]
        rss = r_local[-1, -1]

        ncol = r_local.shape[1]
        inv_r = torch.linalg.inv(r.T @ r)

        dispersion = 1 if family is Binomial else (rss**2) / (total_nrow - ncol)

        return inv_r * dispersion

    @classmethod
    def stop(cls, maxit: int, tol: float, iter: int, diff: float) -> bool:
        return iter >= maxit or diff < tol

    @classmethod
    def distributed_binomial_single_iter_n(
        cls, x: Tensor, y: Tensor, beta: Tensor
    ) -> Tensor:
        eta = x @ beta
        mu = Binomial.linkinv(eta)
        dmu = Binomial.mu_eta(eta)
        z = eta + (y - mu) / dmu
        w = (dmu**2) / Binomial.variance(mu)

        x_tilde = torch.sqrt(w) * x
        z_tilde = torch.sqrt(w) * z

        _, r_local = torch.linalg.qr(torch.cat([x_tilde, z_tilde], dim=1))

        return r_local

    @classmethod
    def distributed_binomial_single_solve_n(
        cls,
        r_local_with_all_r_remotes: Tensor,
        beta: Tensor,
        total_nrow: int,
        maxit: int,
        tol: float,
        iter: int,
    ) -> tuple[Tensor, Tensor, bool]:
        beta_old = beta
        r_local, beta = cls.ols_n(r_local_with_all_r_remotes)

        vcov = cls.vcov(r_local, Binomial, total_nrow)
        delta = (beta_old - beta) / torch.sqrt(torch.diagonal(vcov))
        diff = torch.abs(delta).max()
        stop = cls.stop(maxit, tol, iter, diff)

        return (r_local, beta, stop)

    # TODO: Remove
    # unimplemented yet!

    @classmethod
    def fit_n(cls, x: Tensor, y: Tensor):
        r_xy_or_xy: Tensor = torch.cat([x, y], dim=1)
        return cls.ols_n(r_xy_or_xy)

    @classmethod
    @classmethod
    def update_distributed(cls: type[T], lm: T, r_remote: Tensor) -> T:
        r_local, beta = cls.update_distributed_n(lm.r_local, r_remote)
        return GeneralizedLinearModel(r_local, beta)

    @classmethod
    def update_distributed_n(cls, r_local, r_remote) -> tuple[Tensor, Tensor]:
        return cls.ols_n(torch.cat([r_local, r_remote], dim=0))
