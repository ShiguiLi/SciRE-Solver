import torch
import torch.nn.functional as F
import math


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
            
        print('schedule=', schedule)
        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
            self.log_alpha_array = log_alphas.reshape((1, -1,)).to(dtype=dtype)
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.
            self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
            self.schedule = schedule
            if schedule == 'cosine':
                # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
                # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
                self.T = 0.9946
            else:
                self.T = 1.

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == 'discrete':
            return interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            return -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            log_alpha_t =  log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1. - torch.exp(2. * self.marginal_log_mean_coeff(t)))


    def marginal_NSR(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return torch.exp(log_std - log_mean_coeff)

    def inverse_NSR(self, nsr):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.log(1 + nsr ** 2).to(nsr)
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.log(1 + nsr ** 2).to(nsr)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(nsr.device), [1]), torch.flip(self.t_array.to(nsr.device), [1]))
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.log(1 + nsr ** 2).to(nsr)
            t_fn = lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2. * (1. + self.cosine_s) / math.pi - self.cosine_s
            t = t_fn(alpha)
            return t


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
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
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
        The noise predicition model function that is used for DPM-Solver.
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


class SciRE_Solver:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="scire",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.,
        dynamic_thresholding_ratio=0.995,
    ):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["scire","scire++"]
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
        if self.algorithm_type == "scire++":
            # print('scire++')
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

    def scire_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        """DDIM
        """
        ns = self.noise_schedule
        dims = x.dim()
        NSR_s, NSR_t = ns.marginal_NSR(s), ns.marginal_NSR(t)
        alpha_s, alpha_t = ns.marginal_alpha(s), ns.marginal_alpha(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        if self.algorithm_type == "scire":
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                (alpha_t /alpha_s) * x
                + ((NSR_t - NSR_s) *alpha_t) * model_s
                )
        elif self.algorithm_type == "scire++":
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                (sigma_t /sigma_s) * x
                + ((1/NSR_t - 1/NSR_s) *sigma_t) * model_s
                )
        else:
            raise ValueError("'algorithm_type' must be 'scire' or 'scire++', got {}".format(algorithm_type))
        if return_intermediate:
            return x_t, {'model_s': model_s}
        else:
            return x_t

    def singlestep_scire_solver_second_update(self, x, s, t, r1=0.5, model_s=None, return_intermediate=False, solver_type='scire'):
        if self.algorithm_type not in ['scire', 'scire++']:
            raise ValueError("'algorithm_type' must be 'scire' or 'scire++', got {}".format(algorithm_type))
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        NSR_s, NSR_t = ns.marginal_NSR(s), ns.marginal_NSR(t)
        phi = (torch.exp(torch.tensor(1)) - 1)/torch.exp(torch.tensor(1))
        if self.algorithm_type == "scire":
            h = NSR_t - NSR_s
            NSR_s1 = NSR_s + r1 * h
            s1 = ns.inverse_NSR(NSR_s1)
            alpha_s, alpha_s1, alpha_t = ns.marginal_alpha(s), ns.marginal_alpha(s1), ns.marginal_alpha(t)
            # print('singlestep_ee=',ee)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (
                    (alpha_s1/alpha_s) * x
                    + (r1*(NSR_t - NSR_s) * alpha_s1) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1)
            x_t = (
                    (alpha_t/alpha_s) * x
                    + ((NSR_t - NSR_s) *alpha_t )* model_s
                   # + (3 /2 * (NSR_t - NSR_s)*alpha_t) * (model_s1 - model_s)#p3
                    + (0.5/phi/r1*(NSR_t - NSR_s)* alpha_t)* (model_s1 - model_s)#ee
                )
        elif self.algorithm_type == "scire++":
            h = 1/NSR_t-1/NSR_s
            NSR_s1 =NSR_s/(1-r1+r1*NSR_s/NSR_t)
            # NSR_s1 =1/(1/NSR_s+r1*h)
            s1 = ns.inverse_NSR(NSR_s1)
            sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (
                    (sigma_s1 / sigma_s) * x
                    + (r1*(1/NSR_t-1/NSR_s) * sigma_s1) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1)
            x_t = (
                    (sigma_t / sigma_s) * x
                    + ((1/NSR_t-1/NSR_s) * sigma_t) * model_s
                   + (0.5/phi/r1* (1/NSR_t-1/NSR_s)*sigma_t) * (model_s1 - model_s)#ee
                   # + (0.5*3 /2/r1*(1/NSR_t-1/NSR_s) * sigma_t) * (model_s1 - model_s)#p(3)
                )
            # print('(1/NSR_t-1/NSR_s) =', 1/NSR_t-1/NSR_s)
        else:
            raise ValueError("'algorithm_type' must be 'scire' or 'scire++', got {}".format(algorithm_type))
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1}
        else:
            return x_t

    def singlestep_scire_solver_third_update(self, x, s, t, r1=1./3., r2=2./3., model_s=None, model_s1=None, return_intermediate=False, solver_type='scire'):
        if self.algorithm_type not in ['scire', 'scire++']:
            raise ValueError("'algorithm_type' must be either 'scire' or 'ei', got {}".format(algorithm_type))
        if r1 is None:
            r1 = 1. / 3.
        if r2 is None:
            r2 = 2. / 3.
        ns = self.noise_schedule
        dims = len(x.shape) - 1
        NSR_s, NSR_t = ns.marginal_NSR(s), ns.marginal_NSR(t)
        phi = (torch.exp(torch.tensor(1)) - 1.)/torch.exp(torch.tensor(1))
        if self.algorithm_type == "scire":
            h = NSR_t - NSR_s
            NSR_s1 = NSR_s + r1 * h
            NSR_s2 = NSR_s + r2 * h
            s1 = ns.inverse_NSR(NSR_s1)
            s2 = ns.inverse_NSR(NSR_s2)

            alpha_s, alpha_s1, alpha_s2, alpha_t = ns.marginal_alpha(s), ns.marginal_alpha(s1), ns.marginal_alpha(s2), ns.marginal_alpha(t)
            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (
                         (alpha_s1 /alpha_s)  * x
                        + (r1*(NSR_t - NSR_s) * alpha_s1) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = (
                    (alpha_s2 /alpha_s) * x
                    + (r2*(NSR_t - NSR_s) * alpha_s2)* model_s
                    + (0.5*r2/r1/phi * (NSR_t - NSR_s) * alpha_s2)* (model_s1 - model_s)
                    # + (0.5 / r1 * (NSR_t - NSR_s) * alpha_s2) * (model_s1 - model_s)
                )
            model_s2 = self.model_fn(x_s2, s2)
            x_t = (
                   (alpha_t/alpha_s) * x
                    + ((NSR_t - NSR_s) * alpha_t) * model_s
                    # + (3 / 4 * ee * (NSR_t - NSR_s) * alpha_t) * (model_s2 - model_s)
                    + (0.5/r2/phi* (NSR_t - NSR_s) * alpha_t) * (model_s2 - model_s)
                    # + (3 / 4 / r2 * (NSR_t - NSR_s) * alpha_t) * (model_s2 - model_s)
                )
        elif self.algorithm_type == "scire++":  
            h = 1/NSR_t-1/NSR_s
            NSR_s1 =NSR_s/(1-r1+r1*NSR_s/NSR_t)
            s1 = ns.inverse_NSR(NSR_s1)
            NSR_s2 = NSR_s/(1-r2+r2*NSR_s/NSR_t)
            s2 = ns.inverse_NSR(NSR_s2)
            sigma_s, sigma_s1, sigma_s2, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(s2), ns.marginal_std(t)
            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (
                        (sigma_s1 / sigma_s)* x
                        + (r1*(1/NSR_t-1/NSR_s) *sigma_s1) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = (
                    (sigma_s2 / sigma_s) * x
                    + (r2*(1/NSR_t-1/NSR_s) * sigma_s2)* model_s
                    + (0.5 * r2/r1 * 3/2*(1/NSR_t-1/NSR_s) *sigma_s2)* (model_s1 - model_s)
                    # + (0.5*r2/phi/r1*(1/NSR_t-1/NSR_s) *sigma_s2)* (model_s1 - model_s)
                )
            model_s2 = self.model_fn(x_s2, s2)
            x_t = (
                   (sigma_t/sigma_s) * x
                    + ((1/NSR_t-1/NSR_s) * sigma_t) * model_s
                    + ( 0.5/r2/phi*(1/NSR_t-1/NSR_s) * sigma_t) * (model_s2 - model_s)
                    # + ( ee /r2/2* (1/NSR_t-1/NSR_s) * sigma_t) * (model_s2 - model_s)
                )
        else:
            raise ValueError("'algorithm_type' must be 'scire' or 'scire++', got {}".format(algorithm_type))
        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1, 'model_s2': model_s2}
        else:
            return x_t

    def multistep_scire_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type="scire"):
        if self.algorithm_type not in ['scire', 'scire++']:
            raise ValueError("'algorithm_type' must be either 'scire' or 'scire++', got {}".format(algorithm_type))
        ns = self.noise_schedule
        phi = (torch.exp(torch.tensor(1)) - 1.)/torch.exp(torch.tensor(1))
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        NSR_prev_1, NSR_prev_0, NSR_t = ns.marginal_NSR(t_prev_1), ns.marginal_NSR(t_prev_0), ns.marginal_NSR(t)
        alpha_prev_0, alpha_t = ns.marginal_alpha(t_prev_0), ns.marginal_alpha(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
            
        # h_0 = NSR_prev_0 - NSR_prev_1
        # h = NSR_t - NSR_prev_0
        # r0 = h_0 / h
        # print('t_prev_1-t_prev_0=',t_prev_1-t_prev_0)
        # r0 = h_0*(NSR_t-NSR_prev_1) / h**2
        # D1_0 = (1. / r0)*(model_prev_0 - model_prev_1) 
        # D1_0 = model_prev_0 - model_prev_1
        # D1_0 = (t_prev_0-t)*model_prev_0 +(t-t_prev_1)* model_prev_1
        D1_0 = model_prev_0 - model_prev_1

        if self.algorithm_type == "scire":
            h_0 = NSR_prev_0 - NSR_prev_1
            h = NSR_t - NSR_prev_0
            # r0 =2/3* h_0 / h
            r0 =h_0 / h*phi
            # print('r0=',r0)
            x_t = (
                    (alpha_t/alpha_prev_0) * x
                    + ((NSR_t - NSR_prev_0) *alpha_t ) * model_prev_0
                    + (0.5/r0/phi*(NSR_t - NSR_prev_0)* alpha_t)*D1_0#p_3
                    # + (0.5*ee*(NSR_t - NSR_prev_0)* alpha_t)*D1_0#ee
                    # + (0.5*ee/r0*(NSR_t - NSR_prev_0)* alpha_t)*D1_0#ee
                    # + (3/4/r0*(NSR_t - NSR_prev_0)* alpha_t)*D1_0#p_3
                )
        elif self.algorithm_type == "scire++":
            # h_0 = 1/NSR_prev_0 - 1/NSR_prev_1
            # h = 1/NSR_t - 1/NSR_prev_0
            # r0 = 3/2* h/ h_0
            h_0 = NSR_prev_0 - NSR_prev_1
            h = NSR_t - NSR_prev_0
            r0 =h_0 / h*phi
            # r0 = (h+h_0)/ (h-h_0) 
            # r0 =torch.log(h) - torch.log(h_0)
            # r0 =torch.exp (h_0/ h)  
            # print('r0=',r0)
            x_t = (
                (sigma_t /sigma_prev_0) * x
                +(sigma_t * (1/NSR_t - 1/NSR_prev_0)) * model_prev_0
                + (0.5/r0/phi*(1/NSR_t - 1/NSR_prev_0)*sigma_t) * D1_0#ee
                # + (3/4/r0* (1/NSR_t - 1/NSR_prev_0)*sigma_t) * D1_0#ee
                # + (0.5*ee/r0* (1/NSR_t - 1/NSR_prev_0)*sigma_t) * D1_0#ee
                # + (0.5*ee* (1/NSR_t - 1/NSR_prev_0)*sigma_t) * D1_0#ee
                # + (3/4/r0* (1/NSR_t - 1/NSR_prev_0)*sigma_t) * D1_0#p_3
                )
        else:
            raise ValueError("'algorithm_type' must be 'scire' or 'scire++', got {}".format(algorithm_type))
        return x_t
    
    
    
    
    def multistep_scire_solver_third_update(self, x, model_prev_list, t_prev_list, t, solver_type='scire'):
        ns = self.noise_schedule
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        NSR_prev_2, NSR_prev_1, NSR_prev_0, NSR_t = ns.marginal_NSR(t_prev_2), ns.marginal_NSR(t_prev_1), ns.marginal_NSR(t_prev_0), ns.marginal_NSR(t)
        alpha_prev_0, alpha_t = ns.marginal_alpha_t(t_prev_0), ns.marginal_alpha_t(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)

        h_1 = NSR_prev_1 - NSR_prev_2
        h_0 = NSR_prev_0 - NSR_prev_1
        h = NSR_t - NSR_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1. / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1. / (r0 + r1)) * (D1_0 - D1_1)
        if self.algorithm_type == "scirer++":
            phi_1 = torch.expm1(-h)
            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5
            x_t = (
                (sigma_t / sigma_prev_0) * x
                - (alpha_t * phi_1) * model_prev_0
                + (alpha_t * phi_2) * D1
                - (alpha_t * phi_3) * D2
            )
        elif self.algorithm_type == "scirer":
            phi_1 = torch.expm1(h)
            phi_2 = phi_1 / h - 1.
            phi_3 = phi_2 / h - 0.5
            x_t = (
                (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                - (sigma_t * phi_1) * model_prev_0
                - (sigma_t * phi_2) * D1
                - (sigma_t * phi_3) * D2
            )
        else:
            raise ValueError("'algorithm_type' must be 'scire' or 'scire++', got {}".format(algorithm_type))            
        return x_t

    def singlestep_scire_solver_update(self, x, s, t, order, return_intermediate=False, solver_type='scire', r1=None, r2=None):
        """
        Singlestep SciRE-Solver with the order `order` from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of SciRE-Solver. We only support order == 1 or 2 or 3.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'scire' or 'ei'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'scire' type.
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

    def multistep_scire_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type='dpmsolver'):

        if order == 1:
            return self.scire_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_scire_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        # elif order == 3:
        #     return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def scire_solver_adaptive(self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-5, solver_type='dpmsolver'):
        ns = self.noise_schedule
        s = t_T * torch.ones((1,)).to(x)
        NSR_s = ns.marginal_NSR(s)
        NSR_0 = ns.marginal_NSR(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.scire_solver_first_update(x, s, t, return_intermediate=True)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_scire_solver_second_update(x, s, t, r1=r1, solver_type=solver_type, **kwargs)
        elif order == 3:
            r1, r2 = 1. / 3., 2. / 3.
            lower_update = lambda x, s, t: self.singlestep_scire_solver_second_update(x, s, t, r1=r1, return_intermediate=True, solver_type=solver_type)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_scire_solver_third_update(x, s, t, r1=r1, r2=r2, solver_type=solver_type, **kwargs)
        else:
            raise ValueError("For adaptive step size solver, order must be 2 or 3, got {}".format(order))
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_NSR(NSR_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.):
                x = x_higher
                s = t
                x_prev = x_lower
                NSR_s = ns.marginal_NSR(s)
            h = torch.min(theta * h * torch.float_power(E, -1. / order).float(), NSR_0 - NSR_s)
            nfe += order
        print('adaptive solver nfe', nfe)
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

    def inverse(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='scire',
        atol=0.0078, rtol=0.05, return_intermediate=False,
    ):
        """
        Inverse the sample `x` from time `t_start` to `t_end` by DPM-Solver.
        For discrete-time DPMs, we use `t_start=1/N`, where `N` is the total time steps during training.
        """
        t_0 = 1. / self.noise_schedule.total_N if t_start is None else t_start
        t_T = self.noise_schedule.T if t_end is None else t_end
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        return self.sample(x, steps=steps, t_start=t_0, t_end=t_T, order=order, skip_type=skip_type,
            method=method, lower_order_final=lower_order_final, denoise_to_zero=denoise_to_zero, solver_type=solver_type,
            atol=atol, rtol=rtol, return_intermediate=return_intermediate)

    def sample(self, x, steps=20, t_start=None, t_end=None, order=2, skip_type='time_uniform',
        method='multistep', lower_order_final=True, denoise_to_zero=False, solver_type='rde',
        atol=0.0078, rtol=0.05, return_intermediate=False,
    ):

        t_0 = 1. / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0, "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        if return_intermediate:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in ['multistep', 'singlestep', 'singlestep_fixed'], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device
        intermediates = []
        with torch.no_grad():
            if method == 'adaptive':
                x = self.scire_solver_adaptive(x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol, solver_type=solver_type)
            elif method == 'multistep':
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
                # Init the first `order` values by lower order multistep DPM-Solver.
                for step in range(1, order):
                    t = timesteps[step]
                    x = self.multistep_scire_solver_update(x, model_prev_list, t_prev_list, t, step, solver_type=solver_type)
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    t_prev_list.append(t)
                    model_prev_list.append(self.model_fn(x, t))
                # Compute the remaining values by `order`-th order multistep DPM-Solver.
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    # We only use lower order for steps < 10
                    if lower_order_final and steps < 10:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    x = self.multistep_scire_solver_update(x, model_prev_list, t_prev_list, t, step_order, solver_type=solver_type)
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
            elif method in ['singlestep', 'singlestep_fixed']:
                if method == 'singlestep':
                    timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(steps=steps, order=order, skip_type=skip_type, t_T=t_T, t_0=t_0, device=device)
                elif method == 'singlestep_fixed':
                    K = steps // order
                    orders = [order,] * K
                    timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
                for step, order in enumerate(orders):
                    s, t = timesteps_outer[step], timesteps_outer[step + 1]
                    # if self.algorithm_type == "scire++":
                    #     timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order, device=device)
                    #     NSR_inner = self.noise_schedule.marginal_NSR(timesteps_inner)
                    #     h = 1/NSR_inner[-1] - 1/NSR_inner[0]
                    #     r1 = None if order <= 1 else (1/NSR_inner[1] - 1/NSR_inner[0]) / h
                    #     r2 = None if order <= 2 else (1/NSR_inner[2] - 1/NSR_inner[0]) / h
                    #     x = self.singlestep_scire_solver_update(x, s, t, order, solver_type=solver_type, r1=r1, r2=r2)
                    # else:
                    #     timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order, device=device)
                    #     NSR_inner = self.noise_schedule.marginal_NSR(timesteps_inner)
                    #     h = NSR_inner[-1] - NSR_inner[0]
                    #     r1 = None if order <= 1 else (NSR_inner[1] - NSR_inner[0]) / h
                    #     r2 = None if order <= 2 else (NSR_inner[2] - NSR_inner[0]) / h
                    #     x = self.singlestep_scire_solver_update(x, s, t, order, solver_type=solver_type, r1=r1, r2=r2)
                    #     # x = self.singlestep_scire_solver_update(x, s, t, order, solver_type=solver_type)
                    timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order, device=device)
                    NSR_inner = self.noise_schedule.marginal_NSR(timesteps_inner)
                    h = NSR_inner[-1] - NSR_inner[0]
                    r1 = None if order <= 1 else (NSR_inner[1] - NSR_inner[0]) / h
                    r2 = None if order <= 2 else (NSR_inner[2] - NSR_inner[0]) / h
                    x = self.singlestep_scire_solver_update(x, s, t, order, solver_type=solver_type, r1=r1, r2=r2)
                    # x = self.singlestep_scire_solver_update(x, s, t, order, solver_type=solver_type)
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



#############################################################
# other utility functions
#############################################################

def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
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
