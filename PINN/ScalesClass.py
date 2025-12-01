from Utils import *

class Scales: #! ENTENDER
    def __init__(self, L0, E0, A0, rho0, f0_np, f0_th):
        self.L = float(L0)
        self.E_fun = E0
        self.A = float(A0)
        self.rho = float(rho0)
        self.f_func = f0_np   
        self.f_th_func = f0_th

        # Compute F0 = max |f(x)| for x ∈ [0, L]
        xs_np = np.linspace(0.0, self.L, 2001)
        # try numpy call, otherwise try torch call
        try:
            fvals = self.f_func(xs_np)
            if isinstance(fvals, torch.Tensor):
                fvals = fvals.detach().cpu().numpy()
            self.F0 = float(np.max(np.abs(np.array(fvals))) + 1e-12)
        except Exception:
            # try torch input
            try:
                xs_th = torch.from_numpy(xs_np).float()
                fvals_th = self.f_th_func(xs_th)
                if isinstance(fvals_th, torch.Tensor):
                    self.F0 = float(torch.max(torch.abs(fvals_th)).detach().cpu().numpy() + 1e-12)
                else:
                    self.F0 = float(np.max(np.abs(np.array(fvals_th))) + 1e-12)
            except Exception:
                # fallback to 1.0 to avoid div-by-zero
                self.F0 = 1.0 + 1e-12

        # Compute E0 = max |E(x)| for x ∈ [0, L]
        try:
            Evals = self.E_fun(xs_np)
            if isinstance(Evals, torch.Tensor):
                Evals = Evals.detach().cpu().numpy()
            self.E0 = float(np.max(np.abs(np.array(Evals))) + 1e-12)
        except Exception:
            try:
                xs_th = torch.from_numpy(xs_np).float()
                Evals_th = self.E_fun(xs_th)
                if isinstance(Evals_th, torch.Tensor):
                    self.E0 = float(torch.max(torch.abs(Evals_th)).detach().cpu().numpy() + 1e-12)
                else:
                    self.E0 = float(np.max(np.abs(np.array(Evals_th))) + 1e-12)
            except Exception:
                # fallback
                self.E0 = 1.0 + 1e-12


        # Time scale: T = L * sqrt(rho / E)
        self.T = float(np.sqrt(self.rho * self.L**2 / self.E0))

        # Displacement scale so the forcing coefficient becomes O(1)
        # U = F0 * L^2 / (E0 * A)
        self.U = float(self.F0 * self.L**2 / (self.E0 * self.A))

    # ==========================================================
    #   CONVERSION FUNCTIONS: PHYS → SCALED AND SCALED → PHYS
    # ==========================================================
    def x_phys_to_scaled(self, x_phys):
        return x_phys / self.L

    def t_phys_to_scaled(self, t_phys):
        return t_phys / self.T

    def u_phys_to_scaled(self, u_phys):
        return u_phys / self.U

    def E_phys_to_scaled(self, E_phys):
        return E_phys / self.E0

    def f_phys_to_scaled(self, x_phys):
        """
        Return f_scaled(x) = f(x) / F0
        x_phys can be np.array or torch.tensor
        """
        try:
            fvals = self.f_func(x_phys)

            if isinstance(fvals, torch.Tensor):
                return fvals / self.F0
            else:
                return np.array(fvals) / self.F0

        except:
            return np.array(self.f_func(x_phys)) / self.F0

    def scaled_to_u_phys(self, u_scaled):
        return u_scaled * self.U

    def scaled_to_E_phys(self, E_scaled):
        return E_scaled * self.E0
    

