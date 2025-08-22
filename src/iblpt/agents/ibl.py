import random 
import numpy as np


class IBLAgent:
    def __init__(self, d, s, p):  # decay, noise, inertia
        self.d, self.s, self.p = d, s, p

    def reset(self):
        self.mem = [
            {'opt':'risky','out':30.,'t':0},
            {'opt':'safe', 'out':30.,'t':0}
        ]
        self.last, self.t = None, 1

    def activation(self, inst):
        # 1) Base-level: sum (t - ti)^(-d)
        ts = [m['t'] for m in self.mem
              if m['opt']==inst['opt'] and m['out']==inst['out']]
        B  = sum((self.t - ti)**(-self.d) for ti in ts)
        base = np.log(B + 1e-8)

        # 2) Noise: sigma * ln((1-u)/u), u ~ Uniform(0,1)
        u = random.random()
        noise = self.s * np.log((1 - u) / u)

        return base + noise

    def blended(self, opt):
        # apply spreadsheet’s weighting: t = sqrt(2)*s
        scale = self.s * np.sqrt(2)
        insts = [m for m in self.mem if m['opt']==opt]
        acts  = np.array([self.activation(i) for i in insts])
        w     = np.exp(acts / scale)
        if w.sum() > 0:
            w /= w.sum()
        else:
            w = np.ones_like(w) / len(w)
        outs = np.array([i['out'] for i in insts])
        return (w * outs).sum()

    def choose(self, row):
        if self.t==1:
            c = random.choice(['risky','safe'])
        elif random.random()<=self.p:
            c = self.last
        else:
            c = 'risky' if self.blended('risky')>self.blended('safe') else 'safe'
        self.last = c
        return c

    def update(self, c, o):
        self.mem.append({'opt':c,'out':o,'t':self.t})
        self.t += 1

    def run_n(self, row, N=100):
        """Play N repeated‐choices for one problem, return seq of 'risky'/'safe'."""
        self.reset()
        seq = []
        for _ in range(N):
            c = self.choose(row)
            if c == 'risky':
                o = row.val_high if random.random() < row.p_high else row.val_low
            else:
                o = row.val_safe
            self.update(c, o)
            seq.append(c)
        return seq


