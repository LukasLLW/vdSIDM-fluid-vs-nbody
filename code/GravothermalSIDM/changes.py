
# added imports:
from scipy.special import exp1
from scipy.interpolate import interp1d

self.halo_ini = {
    # as is...
    'sigma_smfp_path': "empty",
    "sigma_lmfp_path": "empty",  
    "alpha": 1}  

def initialize_elastic_scattering(self,model_name):

    if model_name in ('constant', 'const'):
        
        output = lambda x: 1.0        
    elif 'powerlaw' in model_name:
        index = float(model_name.split('_')[1])
        output = lambda x: np.pow(x,index)
    elif 'YukawaBornViscosityApprox' in model_name:
        index = int(model_name.split('_')[1].lstrip('K'))
        order = int(model_name.split('_')[2].lstrip('order'))
        flag_t_channel = False
        if 'Tchannel' in model_name:
            flag_t_channel = True
        output = average_YukawaBornViscosityApprox(index,order,flag_t_channel)

    elif 'Fischer' in model_name:
        phi = lambda x: 4 * x**2
        
        def fischer_function_smfp(v):
            try:
                p = phi(v)
                if p < 0.01:
                    return(1.0
                            -8.0*p
                            +60*p**2
                            -480*p**3
                            +4200.0*p**4)
                term1 = p - 2 - 1/p
                term2 = (3/p + 1/p**2) * np.exp(1/p) * exp1(1/p)
                res = 1 / (6 * p**3) * (term1 + term2)
                return res if np.isfinite(res) else 1.0
            except:
                return 1.0
        
        def fischer_function_lmfp(v):
            try:
                p = phi(v)
                if p < 1e-6: return 1.0 
                term1 = 1 + 1/p
                term2 = (2/p + 1/p**2) * np.exp(1/p) * exp1(1/p)
                res = 1 / (2 * p**2) * (term1 - term2) 
                return res if np.isfinite(res) else 1.0
            except:
                return 1.0
        
        from scipy.integrate import quad
        def fischer_function_7(v):
            phi_val = 4 * v**2
            
            if phi_val < 0.0005:
                return 1.0 - 5.0 * phi_val + 15.0 * phi_val**2
            
            if phi_val < 50.0:
                integrand = lambda x: (x**4 * np.exp(-x)) / (1 + phi_val * x)**2
                res, _ = quad(integrand, 0, np.inf)

                return res / 24.0

            return 0.0

                        
        nu_interp = np.logspace(-5, 3, 20000)


        model_lower = model_name.lower()
        if 'm7' in model_lower:
            print("K")
            y = np.array([fischer_function_7(x) for x in nu_interp])
            output = lambda x: np.interp(x, nu_interp, y)

        elif 'm5' in model_lower:
            y = np.array([fischer_function_lmfp(x) for x in nu_interp])
            output = lambda x: np.interp(x, nu_interp, y)

        elif 'm3' in model_lower:
            y = np.array([fischer_function_smfp(x) for x in nu_interp])
            output = lambda x: np.interp(x, nu_interp, y)

        elif 'viscosity' in model_lower:
            output = lambda x: 1/ ( (1+x*x) * (1+x*x) )

        else:
            raise IOError('Fischer model specified but regime (LMFP/SMFP) unknown in {}'.format(model_name))
    elif 'csv' in model_name:
        model_lower = model_name.lower()
        if 'smfp' in model_lower:
            path=self.sigma_smfp_path
            print('l')
        elif 'lmfp' in model_lower:
            path=self.sigma_lmfp_path
            print('s')
        data_smfp = np.loadtxt(path, delimiter=",", skiprows=1)

        x_smfp = data_smfp[:, 0]
        y_smfp = data_smfp[:, 1]

        output = interp1d(
            x_smfp,
            y_smfp,
            bounds_error=False,
            fill_value=(y_smfp[0], y_smfp[-1]),
        )
        
    else:
        raise IOError('Elastic scattering model {} unknown'.format(model_name))

    return output

def update_derived_parameters(self):

    self.u = (3./2.) * self.p/self.rho

    sigma_r_iso = np.sqrt(self.p / self.rho)

    self.v = sigma_r_iso
    


    ratio = self.v / self.w 
    F_lmfp = self.F_elastic_lmfp(ratio)
    F_smfp = self.F_elastic_smfp(ratio)

    self.Kinv_smfp = self.sigma_m * F_smfp / (self.b * self.v)
    self.Kinv_lmfp = 1. / (self.a * self.C * self.v * self.p * self.sigma_m * F_lmfp)
    Keff = (self.Kinv_lmfp**self.alpha + self.Kinv_smfp**self.alpha)**(-1.0 / self.alpha)                         

    self.L[1:-1] = -self.r[1:-1]*self.r[1:-1] * (Keff[1:-1]+Keff[2:])/2. * (self.u[2:]-self.u[1:-1])/((self.r[2:]-self.r[:-2])/2.)
    self.L[0] = -self.r[0]*self.r[0] * (Keff[0]+Keff[1])/2. * (self.u[1]-self.u[0])/(self.r[1]/2.)
    self.L[-1] = 0

    self.Kn = 1. / (np.sqrt(self.p) * self.sigma_m)

    return