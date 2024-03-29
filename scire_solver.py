import torch
import torch.nn.functional as F
import math


'''
    Args:
        algorithm_type: 'scire_v1' and 'scire_v2' are two versions of SciRE-Solver with different discretization approaches
                         that can result in different sampling performance.

            "scire_v1" is designed using the discretization method of noise prediction diffusion ODE.
            "scire_v2" is designed using the discretization method of data prediction diffusion ODE.

            "scire_v1" and "scire_v2" both support four types of the diffusion model and 
            three types of guided sampling by DMs by the corresponding setting in 'model_wrapper'. 
        order: A `int`. The order of SciRE-Solver (v1 and v2). We only support order == 1 or 2 or 3.
        x: A pytorch tensor with shape `(batch_size, *shape)`, representing initial values at time `s` in each loop iteration.
        s: A pytorch tensor. The starting time, with the shape (1,).
        t: A pytorch tensor . The ending time, with the shape (1,).
        betas, alpha_t, sigma_t: noise schedule.
        NSR_t: = sigma_t / alpha_t, which we refer to as the noise-to-signal-ratio function in the paper.
        phi: the coefficient function of the recursive difference method.
        phi_step: the coefficient of the recursive difference method in the current iteration.
        r1: A `float`. The hyperparameter of the second-order or third-order solver.
        r2: A `float`. The hyperparameter of the third-order solver.
        x_t: A pytorch tensor. The approximated solution at time `t`.
        model_s: A pytorch tensor. The model function evaluated at time `s`.
        model_prev_list: A list of pytorch tensor. The previous computed model values.
        t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
        solver_type: 'scire'.  
        return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
'''     

class SciRE_Solver:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="scire_v1",
        # algorithm_type="scire_v2",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.,
        dynamic_thresholding_ratio=0.995,
    ):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["scire_v1", "scire_v2"]
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
        The dynamic thresholding method. 
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
        if self.algorithm_type == "scire_v2":
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        if skip_type == 'NSR':
            NSR_T = self.noise_schedule.marginal_NSR(torch.tensor(t_T).to(device))
            NSR_0 = self.noise_schedule.marginal_NSR(torch.tensor(t_0).to(device))
            k = 3.1 #recommended to take (k\in[2,7])
            trans_T = -torch.log(NSR_T + k * NSR_0)
            trans_0 = -torch.log(NSR_0 + k * NSR_0)
            trans_steps = torch.linspace(trans_T.cpu().item(), trans_0.cpu().item(), N + 1).to(device)
            NSR_steps = torch.exp(-trans_steps)
            return self.noise_schedule.inverse_NSR(NSR_steps - k * NSR_0)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'NSR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3,] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3,] * (K - 1) + [1]
            else:
                orders = [3,] * (K - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2,] * K
            else:
                K = steps // 2 + 1
                orders = [2,] * (K - 1) + [1]
        elif order == 1:
            K = 1
            orders = [1,] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[torch.cumsum(torch.tensor([0,] + orders), 0).to(device)]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from NSR_s to infty by first-order discretization. 
        """
        return self.data_prediction_fn(x, s)

    def scire_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False, method=None):
        """ Equivalent to DDIM, but with a simplified iteration approach here"""
        ns = self.noise_schedule
        dims = x.dim()
        NSR_s, NSR_t = ns.marginal_NSR(s), ns.marginal_NSR(t)
        if self.algorithm_type == "scire_v1":
            alpha_s = ns.marginal_alpha(s)
            if model_s is None:
                model_s = self.model_fn(alpha_s*x, s)
            x_t = (
                    x 
                    + (NSR_t - NSR_s) * model_s
                )
        elif self.algorithm_type == "scire_v2":
            sigma_s = ns.marginal_std(s)
            if model_s is None:
                model_s = self.model_fn(sigma_s*x, s)
            x_t = ( 
                    x
                    + (1/NSR_t - 1/NSR_s) * model_s
                )
        else:
            raise ValueError("'algorithm_type' must be 'scire_v1' or 'scire_v2', got {}".format(algorithm_type))
        if return_intermediate:
            return x_t, {'model_s': model_s}
        else:
            return x_t

    def singlestep_scire_solver_second_update(self, x, s, t, r1=0.5, model_s=None, return_intermediate=False, solver_type='scire', phi_step=None):
        if self.algorithm_type not in ['scire_v1', 'scire_v2']:
            raise ValueError("'algorithm_type' must be 'scire_v1' or 'scire_v2', got {}".format(algorithm_type))
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        ########################################################################################################
        # The coefficient fuction phi in the recursive difference method can be set to either 
        # (torch.exp(torch.tensor(1)) - 1.) / torch.exp(torch.tensor(1)) or 2/3 during each iteration (fixed). 
        ########################################################################################################
        NSR_s, NSR_t = ns.marginal_NSR(s), ns.marginal_NSR(t)
        if self.algorithm_type == "scire_v1":
            h = NSR_t - NSR_s
            NSR_s1 = NSR_s + r1 * h
            s1 = ns.inverse_NSR(NSR_s1)
            alpha_s, alpha_s1 = ns.marginal_alpha(s), ns.marginal_alpha(s1)
            if model_s is None:
                model_s = self.model_fn(alpha_s*x, s)
            x_s1 = (
                     x
                     + (r1*h) * model_s
                 )
            model_s1 = self.model_fn(alpha_s1*x_s1, s1)
            x_t = (
                    x
                    + h * model_s
                    + (0.5/r1/phi_step*h)* (model_s1 - model_s)
                )
        elif self.algorithm_type == "scire_v2":
            tau_t, tau_s = 1/NSR_t, 1/NSR_s
            h = tau_t-tau_s
            tau_s1 =tau_s+r1*h
            s1 = ns.inverse_NSR(1/tau_s1)
            sigma_s, sigma_s1= ns.marginal_std(s), ns.marginal_std(s1)
            if model_s is None:
                model_s = self.model_fn(sigma_s*x, s)
            x_s1 = ( 
                     x
                     + (r1 * h) * model_s 
                 )
            model_s1 = self.model_fn(sigma_s1*x_s1, s1)
            x_t = (
                    x 
                    + h * model_s
                    + (0.5/r1/phi_step * h) * (model_s1 - model_s)
                )
        else:
            raise ValueError("'algorithm_type' must be 'scire_v1' or 'scire_v2', got {}".format(algorithm_type))
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1}
        else:
            return x_t

    def singlestep_scire_solver_third_update(self, x, s, t, r1=1./3., r2=2./3., model_s=None, model_s1=None, return_intermediate=False, solver_type='scire', phi_step=None):
        if self.algorithm_type not in ['scire_v1', 'scire_v2']:
            raise ValueError("'algorithm_type' must be either 'scire' or 'ei', got {}".format(algorithm_type))
        if r1 is None:
            r1 = 1. / 3.
        if r2 is None:
            r2 = 2. / 3.
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        NSR_s, NSR_t = ns.marginal_NSR(s), ns.marginal_NSR(t)
        if self.algorithm_type == "scire_v1":
            h = NSR_t - NSR_s
            NSR_s1 = NSR_s + r1 * h
            NSR_s2 = NSR_s + r2 * h
            s1 = ns.inverse_NSR(NSR_s1)
            s2 = ns.inverse_NSR(NSR_s2)
            alpha_s, alpha_s1, alpha_s2 = ns.marginal_alpha(s), ns.marginal_alpha(s1), ns.marginal_alpha(s2)

            if model_s is None:
                model_s = self.model_fn(alpha_s*x, s)
            if model_s1 is None:
                x_s1 = (
                         x 
                         + r1*(NSR_t - NSR_s) * model_s
                     )
                model_s1 = self.model_fn(alpha_s1*x_s1, s1)
            x_s2 = (
                     x 
                     + r2*(NSR_t - NSR_s)* model_s
                     + (0.5*r2/r1/phi_step )* (NSR_t - NSR_s)* (model_s1 - model_s)
                 )
            model_s2 = self.model_fn(alpha_s2*x_s2, s2)
            x_t = (
                    x 
                    + (NSR_t - NSR_s) * model_s
                    + (0.5/r2/phi_step )* (NSR_t - NSR_s) * (model_s2 - model_s)
                )
        elif self.algorithm_type == "scire_v2":  
            tau_t, tau_s = 1/NSR_t, 1/NSR_s
            h = tau_t-tau_s
            tau_s1 =tau_s+r1*h
            s1 = ns.inverse_NSR(1/tau_s1)
            tau_s2 =tau_s+r2*h
            s2 = ns.inverse_NSR(1/tau_s2)
            sigma_s, sigma_s1, sigma_s2 = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(s2)
            if model_s is None:
                model_s = self.model_fn(sigma_s*x, s)
            if model_s1 is None:
                x_s1 = (
                        x 
                        + r1*h* model_s
                     )
                model_s1 = self.model_fn(sigma_s1*x_s1, s1)
            x_s2 = (
                    x
                    + r2* h * model_s
                    + (0.5 * r2/r1/phi_step) * h * (model_s1 - model_s)
                 )
            model_s2 = self.model_fn(sigma_s2*x_s2, s2)
            x_t = (
                    x
                    + h * model_s
                    + ( 0.5/r2/phi_step) * h * (model_s2 - model_s)
                )
        else:
            raise ValueError("'algorithm_type' must be 'scire_v1' or 'scire_v2', got {}".format(algorithm_type))
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1, 'model_s2': model_s2}
        else:
            return x_t

    def multistep_scire_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type="scire", phi_step=None):
        if self.algorithm_type not in ['scire_v1', 'scire_v2']:
            raise ValueError("'algorithm_type' must be either 'scire_v1' or 'scire_v2', got {}".format(algorithm_type))
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        NSR_prev_1, NSR_prev_0, NSR_t = ns.marginal_NSR(t_prev_1), ns.marginal_NSR(t_prev_0), ns.marginal_NSR(t)
        r = torch.log(NSR_prev_1/NSR_prev_0)/torch.log(NSR_prev_0/NSR_t)
        D1_0 = (model_prev_0 - model_prev_1)/r

        if self.algorithm_type == "scire_v1":
            h = NSR_t - NSR_prev_0

            x_t = (
                    x
                    + h * model_prev_0
                    + (0.5/phi_step*h) * D1_0
                )
        elif self.algorithm_type == "scire_v2":
            h = 1/NSR_t - 1/NSR_prev_0

            x_t = (
                    x
                    + h * model_prev_0
                    + (0.5/phi_step* h) * D1_0
                )
        else:
            raise ValueError("'algorithm_type' must be 'scire_v1' or 'scire_v2', got {}".format(algorithm_type))
        return x_t
    
    def multistep_scire_solver_third_update(self, x, model_prev_list, t_prev_list, t, solver_type='scire', phi_step=None):
        if self.algorithm_type not in ['scire_v1', 'scire_v2']:
            raise ValueError("'algorithm_type' must be either 'scire_v1' or 'scire_v2', got {}".format(algorithm_type))
        ns = self.noise_schedule
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        NSR_prev_2, NSR_prev_1, NSR_prev_0, NSR_t = ns.marginal_NSR(t_prev_2), ns.marginal_NSR(t_prev_1), ns.marginal_NSR(t_prev_0), ns.marginal_NSR(t)
        r0, r1 = phi_step * (NSR_prev_0 - NSR_prev_1) /(NSR_t - NSR_prev_0), (phi_step/(1+phi_step))*(NSR_prev_1 - NSR_prev_2) / (NSR_prev_0 - NSR_prev_1)
        
        D1, D2 = (model_prev_0 - model_prev_1), (model_prev_0 -model_prev_1  +  model_prev_2 -model_prev_1)

        if self.algorithm_type == "scire_v1":
            h = NSR_t - NSR_prev_0

            x_t = (
                    x
                    + h * model_prev_0
                    + (0.5/(r0*phi_step)*h) * D1
                    + (h/6/(r0*phi_step)/(r1*phi_step)) * D2
                )
        elif self.algorithm_type == "scire_v2":
            h = 1/NSR_t - 1/NSR_prev_0

            x_t = (
                    x
                    + h * model_prev_0
                    + (0.5/(r0*phi_step) * h) * D1
                    + (h/6/(r0*phi_step)/(r1*phi_step)) * D2
                )
        else:
            raise ValueError("'algorithm_type' must be 'scire_v1' or 'scire_v2', got {}".format(algorithm_type))            
        return x_t

    def singlestep_scire_solver_update(self, x, s, t, order, return_intermediate=False, solver_type='scire', r1=None, r2=None, phi_step=None):
        """
        Singlestep SciRE-Solver with the order `order` from time `s` to time `t`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.scire_solver_first_update(x, s, t, return_intermediate=return_intermediate)
        elif order == 2:
            return self.singlestep_scire_solver_second_update(x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1, phi_step=phi_step)
        elif order == 3:
            return self.singlestep_scire_solver_third_update(x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1, r2=r2, phi_step=phi_step)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def multistep_scire_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type='scire', phi_step=None):

        if order == 1:
            return self.scire_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_scire_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type, phi_step=phi_step)
        elif order == 3:
            return self.multistep_scire_solver_third_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type, phi_step=phi_step)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))


    def add_noise(self, x, t, noise=None):
        """
        Compute the noised input xt = alpha_t * x + sigma_t * noise. 
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


    def sample(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='scire',
        atol=0.0078, rtol=0.05, return_intermediate=False,  
    ):
        
        t_T = self.noise_schedule.T if t_start is None else t_start
        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        if return_intermediate:
            assert method in ['multistep', 'singlestep_agile', 'singlestep_fixed'], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in ['multistep', 'singlestep_agile', 'singlestep_fixed'], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device
        # Simplify each iteration of SciRE-Solver
        if self.algorithm_type == "scire_v1":
            x = 1/self.noise_schedule.marginal_alpha(torch.tensor(t_T).to(device)) * x
        #This step for scire_v2 can be omitted as self.noise_schedule.marginal_std(torch.tensor(t_T).to(device)) approximates to 1.
        elif self.algorithm_type == "scire_v2": 
            x = 1/self.noise_schedule.marginal_std(torch.tensor(t_T).to(device)) * x
            
        intermediates = []
        with torch.no_grad():
            if method == 'multistep':
                assert steps >= order
                # compute phi_1(m) as described in paper
                phi = self.noise_schedule.series_phi(1, steps)
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
                assert timesteps.shape[0] - 1 == steps
                # Init the initial values.
                step = 0
                t = timesteps[step]
                t_prev_list = [t]
                if self.algorithm_type == "scire_v1":
                    model_prev_list = [self.model_fn(self.noise_schedule.marginal_alpha(t)*x, t)]
                elif self.algorithm_type == "scire_v2":
                    model_prev_list = [self.model_fn(self.noise_schedule.marginal_std(t)*x, t)]
                else:
                    raise ValueError("'algorithm_type' must be 'scire_v1' or 'scire_v2', got {}".format(algorithm_type))
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)
                # Init the first `order` values by lower order multistep SciRE-Solver.
                for step in range(1, order):
                    phi_step = phi[-step]
                    t = timesteps[step]
                    x = self.multistep_scire_solver_update(x, model_prev_list, t_prev_list, t, step, solver_type=solver_type, phi_step = phi_step)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    t_prev_list.append(t)
                    if self.algorithm_type == "scire_v1":
                        model_prev_list.append(self.model_fn(self.noise_schedule.marginal_alpha(t)*x, t))
                    elif self.algorithm_type == "scire_v2":
                        model_prev_list.append(self.model_fn(self.noise_schedule.marginal_std(t)*x, t))
                # Compute the remaining values by `order`-th order multistep SciRE-Solver.
                K = (steps + 1 - order)//2
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    # We only use lower order for steps < 10
                    if lower_order_final and steps < 10:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    if step < K+order: 
                        phi_step = phi[2*(step-order)+1]
                    else:
                        phi_step = phi[2 *(order + 2*K- step)]
                    x = self.multistep_scire_solver_update(x, model_prev_list, t_prev_list, t, step_order, solver_type=solver_type, phi_step = phi_step)
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
                        if self.algorithm_type == "scire_v1":
                            model_prev_list[-1] = self.model_fn(self.noise_schedule.marginal_alpha(t)*x, t)
                        elif self.algorithm_type == "scire_v2":
                            model_prev_list[-1] = self.model_fn(self.noise_schedule.marginal_std(t)*x, t)

            elif method in ['singlestep_agile', 'singlestep_fixed']:
                K = steps // order
                # compute phi_1(m) as described in paper
                phi = self.noise_schedule.series_phi(1, steps//order)
                
                if method == 'singlestep_agile':
                    timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(steps=steps, order=order, skip_type=skip_type, t_T=t_T, t_0=t_0, device=device)
                elif method == 'singlestep_fixed':
                    orders = [order,] * K
                    timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
                for step, order in enumerate(orders):
                    s, t = timesteps_outer[step], timesteps_outer[step + 1]
                    if step < K//2: 
                        phi_step = phi[2*step+1]
                    else:
                        phi_step = phi[2*(K-step-1)]
                    if self.noise_schedule.schedule == 'discrete': 
                        timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order, device=device)
                        NSR_inner = self.noise_schedule.marginal_NSR(timesteps_inner)
                        h = NSR_inner[-1] - NSR_inner[0]
                        r1 = None if order <= 1 else (NSR_inner[1] - NSR_inner[0]) / h
                        r2 = None if order <= 2 else (NSR_inner[2] - NSR_inner[0]) / h
                        x = self.singlestep_scire_solver_update(x, s, t, order, solver_type=solver_type, r1=r1, r2=r2, phi_step=phi_step)
                    elif self.noise_schedule.schedule == 'linear':
                        x = self.singlestep_scire_solver_update(x, s, t, order, solver_type=solver_type, phi_step=phi_step)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                        
            else:
                raise ValueError("Got wrong method {}".format(method))
            # This step for scire_v1 can be omitted as self.noise_schedule.marginal_alpha(torch.tensor(t_0).to(device)) approximates to 1.
            if self.algorithm_type == "scire_v1": 
                x = self.noise_schedule.marginal_alpha(torch.tensor(t_0).to(device)) * x
            elif self.algorithm_type == "scire_v2":
                x = self.noise_schedule.marginal_std(torch.tensor(t_0).to(device)) * x
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



###########################################################################
# noise schedule of VP, and the NSR and rNSR functions.
###########################################################################  

class NoiseScheduleVP:
    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            dtype=torch.float32,
        ):

        if schedule not in ['discrete', 'linear', 'cosine']:
            raise ValueError("Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear' or 'cosine'".format(schedule))
            
        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                alphas = torch.sqrt((1 - betas).cumprod(dim=0))
            else:
                assert alphas_cumprod is not None
                alphas = torch.sqrt( alphas_cumprod)
            self.total_N = len(alphas)
            self.T = 1.
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
            self.alpha_array = alphas.reshape((1, -1,)).to(dtype=dtype)
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.
            self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            self.cosine_alpha_0 = math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.)
            self.schedule = schedule
            if schedule == 'cosine':
                # For the cosine schedule, following the set in dpm-solver, although T = 0.9946 may be not the optimal setting.
                self.T = 0.9946
            else:
                self.T = 1.

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'discrete':
            # return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.alpha_array.to(t.device)).reshape((-1))
            return torch.exp(interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device),torch.log(self.alpha_array).to(t.device)).reshape((-1)))
        elif self.schedule == 'linear':
            return torch.exp(-0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0)
        elif self.schedule == 'cosine':
            alpha_fn = lambda s: torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.)
            alpha_t =  alpha_fn(t) / self.cosine_alpha_0
            return alpha_t

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.pow(self.marginal_alpha(t), 2))


    def marginal_NSR(self, t):
        """
        Compute NSR_t = sigma_t / alpha_t of a given continuous-time label t in [0, T].
        """
        alpha_t = self.marginal_alpha(t)
        sigma_t = torch.sqrt(1. - torch.pow(alpha_t, 2))
        return sigma_t/alpha_t

    def inverse_NSR(self, nsr):
        """
        Compute the continuous-time label t in [0, T] of a given NSR_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.log(1 + nsr ** 2).to(nsr)
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.log(1 + nsr ** 2).to(nsr)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(torch.log(self.alpha_array).to(nsr.device), [1]), torch.flip(self.t_array.to(nsr.device), [1]))
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.log(1 + nsr ** 2).to(nsr)
            t_fn = lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + torch.log(self.cosine_alpha_0))) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            t = t_fn(log_alpha)
            return t
        

    def series_phi(self, x, n):
        """
        Calculate the sum of the first n terms of the phi series and store it.
        """
        result = 0
        numerator = 1
        phi=[]
        for i in range(1,n+3):
            term = (-x) ** (i-1) / numerator
            result += term
            numerator *= (i + 1)
            if i != 1:
                phi.append(result)
        del phi[0] 
        return phi
    
        
#########################################################################
# a piecewise linear interpolation function for 'discrete' schedule.
#########################################################################

def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)
    
    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels.
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K), torch.tensor(K - 2, device=x.device), cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


#############################################################
# the model_wrapper function in dpm-solver (many thanks).
''' 
    [1] Lu, Cheng and Zhou, Yuhao and Bao, Fan and Chen, Jianfei and Li, Chongxuan and Zhu, Jun,
        "Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps," 
        in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 5775-5787 (2022). 
'''
############################################################# 
'''
    "scire_v1" and "scire_v2" both support four types of the diffusion model by setting `model_type`:
        1. "noise": noise prediction model. (Trained by predicting noise).
        2. "x_start": data prediction model. (Trained by predicting the data x_0 at time 0).
        3. "v": velocity prediction model. (Trained by predicting the velocity).
            The "v" prediction is derivation detailed in Appendix D of [1], and is used in Imagen-Video [2].
            [2] Salimans, Tim, and Jonathan, Ho, "Progressive distillation for fast sampling of diffusion models,"
                        arXiv preprint arXiv:2202.00512 (2022).
            [3] Ho, Jonathan, et al., "Imagen Video: High Definition Video Generation with Diffusion Models,"
                arXiv preprint arXiv:2210.02303 (2022).
        4. "score": marginal score function. (Trained by denoising score matching).
            Note that the score function and the noise prediction model follows a simple relationship:
                ```  noise(x_t, t) = -sigma_t * score(x_t, t)  ```   
    "scire_v1" and "scire_v2" both support three types of guided sampling by DMs by setting `guidance_type`:
        1. "uncond": unconditional sampling by DMs.
            The input `model` has the following format:
                ``  model(x, t_input, **model_kwargs) -> noise | x_start | v | score  ``
        2. "classifier": classifier guidance sampling [3] by DMs and another classifier.
            The input `model` has the following format:
                ``  model(x, t_input, **model_kwargs) -> noise | x_start | v | score  `` 
            The input `classifier_fn` has the following format:
                ``  classifier_fn(x, t_input, cond, **classifier_kwargs) -> logits(x, t_input, cond)  ``
            [4] P. Dhariwal and A. Q. Nichol, "Diffusion models beat GANs on image synthesis,"
                in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 8780-8794.
        3. "classifier-free": classifier-free guidance sampling by conditional DMs.
            The input `model` has the following format:
                ``  model(x, t_input, cond, **model_kwargs) -> noise | x_start | v | score  `` 
            And if cond == `unconditional_condition`, the model output is the unconditional DM output.
            [5] Ho, Jonathan, and Tim Salimans, "Classifier-free diffusion guidance,"
                arXiv preprint arXiv:2207.12598 (2022).
'''

def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs={},
    guidance_type="uncond",
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.,
    classifier_fn=None,
    classifier_kwargs={},
):

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DMs, we just use `t_continuous`.
        """
        if noise_schedule.schedule == 'discrete':
            return (t_continuous - 1. / noise_schedule.total_N) * 1000.
        elif noise_schedule.schedule == 'linear':
            return 1000. * torch.max(t_continuous - 1. / noise_schedule.total_N,
                                     torch.zeros_like(t_continuous).to(t_continuous))
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return (x - alpha_t * output) / sigma_t
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return alpha_t * output + sigma_t * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -sigma_t * output

    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for SciRE-Solver.
        """
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * sigma_t * cond_grad
        elif guidance_type == "classifier-free":
            if guidance_scale == 1. or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn            

def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,)*(dims - 1)]

