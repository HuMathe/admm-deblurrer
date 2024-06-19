class AdmmBase:
    '''
    ADMM solver class for Inverse Problems.
    '''
    def __init__(self, input_shape):
        self.shape = input_shape

    def solve_inverse_problem(self, b, admm_max_iter=100, **kwargs):
        assert b.shape == self.shape

        x, z, u, b = self.admm_init_xzub(b)
        for _ in range(admm_max_iter):
            x = self.admm_x_step(x, z, u, b, **kwargs)
            z = self.admm_z_step(x, z, u, b, **kwargs)
            u = self.admm_u_step(x, z, u, b, **kwargs)
        x = self.admm_refine_x_step(x, z, u, b, **kwargs)
        return x.reshape(self.shape)
        
    def admm_init_xzub(self, b):
        raise NotImplementedError()

    def admm_refine_x_step(self, x, z, u, b, **kwargs):
        raise NotImplementedError()

    def admm_x_step(self, x, z, u, b, **kwargs):
        raise NotImplementedError()

    def admm_z_step(self, x, z, u, b, **kwargs):
        raise NotImplementedError()
    
    def admm_u_step(self, x, z, u, b, **kwargs):
        raise NotImplementedError()