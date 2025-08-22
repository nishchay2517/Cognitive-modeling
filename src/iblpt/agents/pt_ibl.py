from iblpt.agents.ibl import IBLAgent
import numpy as np


class PTIBLAgent(IBLAgent):
    def __init__(self, d, s, p, α, β, lam):
        super().__init__(d, s, p)
        self.α, self.β, self.lam = α, β, lam

    def prospect_value(self, x):
        return x**self.α if x >= 0 else -self.lam * ((-x)**self.β)

    def blended(self, opt):
        scale = self.s * np.sqrt(2)
        insts = [m for m in self.mem if m['opt']==opt]
        acts  = np.array([self.activation(i) for i in insts])
        w     = np.exp(acts / scale)
        if w.sum() > 0:
            w /= w.sum()
        else:
            w = np.ones_like(w) / len(w)
        vals  = np.array([self.prospect_value(i['out']) for i in insts])
        return (w * vals).sum()


