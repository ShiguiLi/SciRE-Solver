import torch
import torch.nn.functional as F
import math
from .utils import expand_dims
import numpy as np


class SciRE_Solver:
    def __init__(
        self,
        noise_schedule,
        algorithm_type="scire",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.0,
        dynamic_thresholding_ratio=0.995,
    ):

        self.noise_schedule = noise_schedule
        assert algorithm_type in ["scire","solver++"]
        self.algorithm_type = algorithm_type
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method in dpm-solver.
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0)
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        if self.algorithm_type == "scire++":
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'NSR': uniform NSR for the time steps.
                - 'logSNR': uniform logSNR for the time steps. (Used in dpm-solver)
                - 'time_uniform': uniform time for the time steps.
                - 'time_quadratic': quadratic time for the time steps. 
                - 'edm': edm time for the time steps. (Used in EDM)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == "NSR":
            NSR_T = self.noise_schedule.marginal_NSR(torch.tensor(t_T).to(device))
            NSR_0 = self.noise_schedule.marginal_NSR(torch.tensor(t_0).to(device))
            k = 3.1 #recommended to take (k\in[2,7])
            trans_T = -torch.log(NSR_T + k * NSR_0)
            trans_0 = -torch.log(NSR_0 + k * NSR_0)
            trans_steps = torch.linspace(trans_T.cpu().item(), trans_0.cpu().item(), N + 1).to(device)
            NSR_steps = torch.exp(-trans_steps)
            return self.noise_schedule.inverse_NSR(NSR_steps - k * NSR_0)    
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == "time_quadratic":
            t_order = 2
            t = torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1).pow(t_order).to(device)
            return t
        elif skip_type == "edm":
            rho = 7.0  # 7.0 is the value used in the paper

            sigma_min: float = t_0
            sigma_max: float = t_T
            ramp = np.linspace(0, 1, N + 1)
            min_inv_rho = sigma_min ** (1 / rho)
            max_inv_rho = sigma_max ** (1 / rho)
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
            lambdas = torch.Tensor(-np.log(sigmas)).to(device)
            t = self.noise_schedule.inverse_lambda(lambdas)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'NSR' or 'logSNR' or 'time_uniform' or 'time_quadratic' or 'edm'".format(skip_type))

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3,] * (
                    K - 2
                ) + [2, 1]
            elif steps % 3 == 1:
                orders = [3,] * (
                    K - 1
                ) + [1]
            else:
                orders = [3,] * (
                    K - 1
                ) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [
                    2,
                ] * K
            else:
                K = steps // 2 + 1
                orders = [2,] * (
                    K - 1
                ) + [1]
        elif order == 1:
            K = 1
            orders = [
                1,
            ] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == "logSNR":
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[
                torch.cumsum(
                    torch.tensor(
                        [
                            0,
                        ]
                        + orders
                    ),
                    0,
                ).to(device)
            ]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        return self.data_prediction_fn(x, s)

    def scire_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        """
        DDIM from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        dims = x.dim()
        NSR_s, NSR_t = ns.marginal_NSR(s), ns.marginal_NSR(t)
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        alpha_t = torch.exp(log_alpha_t)

        if model_s is None:
            model_s = self.model_fn(x, s)
        x_t = (
                (torch.exp(log_alpha_t - log_alpha_s)) * x
                + (NSR_t - NSR_s) * expand_dims(alpha_t, dims) * model_s
        )
        if return_intermediate:
            return x_t, {'model_s': model_s}
        else:
            return x_t

    def singlestep_scire_solver_second_update(self, x, s, t, r1=0.5, model_s=None, return_intermediate=False, solver_type="scire"):
        if self.algorithm_type not in ['scire', 'scire++']:
            raise ValueError("'algorithm_type' must be 'scire' or 'scire++', got {}".format(algorithm_type))
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        ee=torch.exp(torch.tensor(1))/torch.expm1(torch.tensor(1))
        if self.algorithm_type == "scire":
            NSR_s, NSR_t = ns.marginal_NSR(s), ns.marginal_NSR(t)
            h = NSR_t - NSR_s
            NSR_s1 = NSR_s + r1 * h
            s1 = ns.inverse_NSR(NSR_s1)
            alpha_s, alpha_s1, alpha_t = ns.marginal_alpha(s), ns.marginal_alpha(
                s1), ns.marginal_alpha(t)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (
                    (alpha_s1/alpha_s) * x
                    + (r1*(NSR_t - NSR_s) * alpha_s1) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1)
            x_t = (
                    (alpha_t/alpha_s) * x
                    + ((NSR_t - NSR_s)* alpha_t) * model_s
                    # + (3 / 4 * h /r1  * alpha_t) * (model_s1 - model_s)
                    + (3/4/r1* (NSR_t - NSR_s)  * alpha_t) * (model_s1 - model_s)
                    # + (0.5*ee/r1* (NSR_t - NSR_s)  * alpha_t) * (model_s1 - model_s)
                    # + (ee/2/r1* h  * alpha_t) * (model_s1 - model_s)
                )

        elif self.algorithm_type == "scire++":
            lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
            h = lambda_t - lambda_s
            lambda_s1 = lambda_s + r1 * h
            s1 = ns.inverse_lambda(lambda_s1)
            log_alpha_s, log_alpha_s1, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(
                s1), ns.marginal_log_mean_coeff(t)
            sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
            alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)
            phi_11 = torch.expm1(r1 * h)
            phi_1 = torch.expm1(h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (
                    torch.exp(log_alpha_s1 - log_alpha_s) * x
                    - (sigma_s1 * phi_11) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1)
            x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (3 / h) * (sigma_t * (phi_1 - h)) * (model_s1 - model_s)
                )
        else:
            raise ValueError("'algorithm_type' must be 'scire' or 'scire++', got {}".format(algorithm_type))
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1}
        else:
            return x_t

    def singlestep_scire_solver_third_update(
        self, x, s, t, r1=1.0 / 3.0, r2=2.0 / 3.0, model_s=None, model_s1=None, return_intermediate=False, solver_type="scire++"
    ):
        if self.algorithm_type not in ['scire', 'scire++']:
            raise ValueError("'algorithm_type' must be either 'scire' or 'scire++', got {}".format(algorithm_type))
        if r1 is None:
            r1 = 1.0 / 3.0
        if r2 is None:
            r2 = 2.0 / 3.0
        ns = self.noise_schedule
        NSR_s, NSR_t = ns.marginal_NSR(s), ns.marginal_NSR(t)
        alpha_s,  alpha_t = ns.marginal_alpha(s),  ns.marginal_alpha(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        
        if self.algorithm_type == "scire":#SciRE-Solver-3
            h = NSR_t - NSR_s
            NSR_s1 = NSR_s + r1 * h
            NSR_s2 = NSR_s + r2 * h
            s1 = ns.inverse_NSR(NSR_s1)
            s2 = ns.inverse_NSR(NSR_s2)

            alpha_s1, alpha_s2= ns.marginal_alpha(s1), ns.marginal_alpha(s2)
            sigma_s1, sigma_s2 = ns.marginal_std(s1), ns.marginal_std(s2)
            # ee=torch.exp(torch.tensor(1))/torch.expm1(torch.tensor(1))
            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (
                        (alpha_s1/alpha_s)* x
                        + (r1*(NSR_t - NSR_s)* alpha_s1)* model_s
                )
                model_s1 = self.model_fn(x_s1, s1)
            # if solver_type == 'rde':
            x_s2 = (
                    (alpha_s2/alpha_s)* x
                    + (r2*(NSR_t - NSR_s) * alpha_s2) * model_s
                    + (r2/r1*3/4* (NSR_t - NSR_s) * alpha_s2)* (model_s1 - model_s)
                )
            model_s2 = self.model_fn(x_s2, s2)
            x_t = (
                    (alpha_t/alpha_s) * x
                    + (h * alpha_t) * model_s
                    # + (3 / 4 / r2 *h* alpha_t) * (model_s2 - model_s)
                    + (3 / 4 / r2 *(NSR_t - NSR_s)* alpha_t) * (model_s2 - model_s)
                )

        if return_intermediate:
            return x_t, {"model_s": model_s, "model_s1": model_s1, "model_s2": model_s2}
        else:
            return x_t

    def multistep_scire_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"):
        """
        Multistep solver SciRE-Solver-2 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ["dpmsolver", "taylor"]:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_1), ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if solver_type == "dpmsolver":
                x_t = (sigma_t / sigma_prev_0) * x - (alpha_t * phi_1) * model_prev_0 - 0.5 * (alpha_t * phi_1) * D1_0
            elif solver_type == "taylor":
                x_t = (sigma_t / sigma_prev_0) * x - (alpha_t * phi_1) * model_prev_0 + (alpha_t * (phi_1 / h + 1.0)) * D1_0
        else:
            phi_1 = torch.expm1(h)
            if solver_type == "dpmsolver":
                x_t = (torch.exp(log_alpha_t - log_alpha_prev_0)) * x - (sigma_t * phi_1) * model_prev_0 - 0.5 * (sigma_t * phi_1) * D1_0
            elif solver_type == "taylor":
                x_t = (torch.exp(log_alpha_t - log_alpha_prev_0)) * x - (sigma_t * phi_1) * model_prev_0 - (sigma_t * (phi_1 / h - 1.0)) * D1_0
        return x_t

    def multistep_scire_solver_third_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"):
        """
        Multistep solver SciRE-Solver-3 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.marginal_lambda(t_prev_2),
            ns.marginal_lambda(t_prev_1),
            ns.marginal_lambda(t_prev_0),
            ns.marginal_lambda(t),
        )
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1.0 / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            phi_2 = phi_1 / h + 1.0
            phi_3 = phi_2 / h - 0.5
            x_t = (sigma_t / sigma_prev_0) * x - (alpha_t * phi_1) * model_prev_0 + (alpha_t * phi_2) * D1 - (alpha_t * phi_3) * D2
        else:
            phi_1 = torch.expm1(h)
            phi_2 = phi_1 / h - 1.0
            phi_3 = phi_2 / h - 0.5
            x_t = (torch.exp(log_alpha_t - log_alpha_prev_0)) * x - (sigma_t * phi_1) * model_prev_0 - (sigma_t * phi_2) * D1 - (sigma_t * phi_3) * D2
        return x_t

    def singlestep_scire_solver_update(self, x, s, t, order, return_intermediate=False, solver_type="dpmsolver", r1=None, r2=None):
        """
        Singlestep SciRE-Solver with the order `order` from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of SciRE-Solver. We only support order == 1 or 2 or 3.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
            r1: A `float`. The hyperparameter of the second-order or third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.scire_solver_first_update(x, s, t, return_intermediate=return_intermediate)
        elif order == 2:
            return self.singlestep_scire_solver_second_update(x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1)
        elif order == 3:
            return self.singlestep_scire_solver_third_update(x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1, r2=r2)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def multistep_scire_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type="dpmsolver"):
        """
        Multistep SciRE-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of SciRE-Solver. We only support order == 1 or 2 or 3.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def scire_solver_adaptive(self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-5, solver_type="dpmsolver"):
        """
        The adaptive step size solver based on singlestep SciRE-Solver.

        Args:
            x: A pytorch tensor. The initial value at time `t_T`.
            order: A `int`. The (higher) order of the solver. We only support order == 2 or 3.
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            h_init: A `float`. The initial step size (for logSNR).
            atol: A `float`. The absolute tolerance of the solver. For image data, the default setting is 0.0078, followed [1].
            rtol: A `float`. The relative tolerance of the solver. The default setting is 0.05.
            theta: A `float`. The safety hyperparameter for adapting the step size. The default setting is 0.9, followed [1].
            t_err: A `float`. The tolerance for the time. We solve the diffusion ODE until the absolute error between the
                current time and `t_0` is less than `t_err`. The default setting is 1e-5.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_0: A pytorch tensor. The approximated solution at time `t_0`.

        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.
        """
        ns = self.noise_schedule
        s = t_T * torch.ones((1,)).to(x)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_intermediate=True)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, solver_type=solver_type, **kwargs)
        elif order == 3:
            r1, r2 = 1.0 / 3.0, 2.0 / 3.0
            lower_update = lambda x, s, t: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, return_intermediate=True, solver_type=solver_type)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_third_update(
                x, s, t, r1=r1, r2=r2, solver_type=solver_type, **kwargs
            )
        else:
            raise ValueError("For adaptive step size solver, order must be 2 or 3, got {}".format(order))
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.0):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(theta * h * torch.float_power(E, -1.0 / order).float(), lambda_0 - lambda_s)
            nfe += order
        print("adaptive solver nfe", nfe)
        return x

    def add_noise(self, x, t, noise=None):
        """
        Compute the noised input xt = alpha_t * x + sigma_t * noise.

        Args:
            x: A `torch.Tensor` with shape `(batch_size, *shape)`.
            t: A `torch.Tensor` with shape `(t_size,)`.
        Returns:
            xt with shape `(t_size, batch_size, *shape)`.
        """
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        if noise is None:
            noise = torch.randn((t.shape[0], *x.shape), device=x.device)
        x = x.reshape((-1, *x.shape))
        xt = expand_dims(alpha_t, x.dim()) * x + expand_dims(sigma_t, x.dim()) * noise
        if t.shape[0] == 1:
            return xt.squeeze(0)
        else:
            return xt

    def inverse(
        self,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=2,
        skip_type="time_uniform",
        method="multistep",
        lower_order_final=True,
        denoise_to_zero=False,
        solver_type="dpmsolver",
        atol=0.0078,
        rtol=0.05,
        return_intermediate=False,
    ):
        """
        Inverse the sample `x` from time `t_start` to `t_end` by SciRE-Solver.
        For discrete-time DPMs, we use `t_start=1/N`, where `N` is the total time steps during training.
        """
        t_0 = 1.0 / self.noise_schedule.total_N if t_start is None else t_start
        t_T = self.noise_schedule.T if t_end is None else t_end
        assert (
            t_0 > 0 and t_T > 0
        ), "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        return self.sample(
            x,
            steps=steps,
            t_start=t_0,
            t_end=t_T,
            order=order,
            skip_type=skip_type,
            method=method,
            lower_order_final=lower_order_final,
            denoise_to_zero=denoise_to_zero,
            solver_type=solver_type,
            atol=atol,
            rtol=rtol,
            return_intermediate=return_intermediate,
        )

    def sample(
        self,
        model_fn,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=2,
        skip_type="time_uniform",
        # method="multistep",
        method="singlestep_fixed",
        # method="singlestep",
        lower_order_final=True,
        denoise_to_zero=False,
        solver_type="rde",
        atol=0.0078,
        rtol=0.05,
        return_intermediate=False,
    ):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        # t_0 = 1e-4
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert (
            t_0 > 0 and t_T > 0
        ), "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        if return_intermediate:
            assert method in ["multistep", "singlestep", "singlestep_fixed"], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in ["multistep", "singlestep", "singlestep_fixed"], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device
        intermediates = []
        with torch.no_grad():
            if method == "adaptive":
                x = self.dpm_solver_adaptive(x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol, solver_type=solver_type)
            elif method == "multistep":
                assert steps >= order
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
                assert timesteps.shape[0] - 1 == steps
                # Init the initial values.
                step = 0
                t = timesteps[step]
                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)
                # Init the first `order` values by lower order multistep SciRE-Solver.
                for step in range(1, order):
                    t = timesteps[step]
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step, solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    t_prev_list.append(t)
                    model_prev_list.append(self.model_fn(x, t))
                # Compute the remaining values by `order`-th order multistep SciRE-Solver.
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    # We only use lower order for steps < 10
                    # [CHANGE] remove the above restriction
                    if lower_order_final:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step_order, solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = t
                    # We do not need to evaluate the final model value.
                    if step < steps:
                        model_prev_list[-1] = self.model_fn(x, t)
            elif method in ["singlestep", "singlestep_fixed"]:
                if method == "singlestep":
                    timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(
                        steps=steps, order=order, skip_type=skip_type, t_T=t_T, t_0=t_0, device=device
                    )
                elif method == "singlestep_fixed":
                    K = steps // order
                    orders = [
                        order,
                    ] * K
                    timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
                for step, order in enumerate(orders):
                    s, t = timesteps_outer[step], timesteps_outer[step + 1]
                    timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order, device=device)
                    lambda_inner = self.noise_schedule.marginal_NSR(timesteps_inner)
                    h = lambda_inner[-1] - lambda_inner[0]
                    r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
                    r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
                    x = self.singlestep_scire_solver_update(x, s, t, order, solver_type=solver_type, r1=r1, r2=r2)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
            else:
                raise ValueError("Got wrong method {}".format(method))
            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        if return_intermediate:
            return x, intermediates
        else:
            return x