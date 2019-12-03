import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import functional
from tqdm import tqdm
from lib.utils import mmread, trace, dist


class BISM(nn.Module):
    def __init__(self, R, args):
        super(BISM, self).__init__()
        RR = R.transpose() * R
        self.RR = torch.from_numpy(RR.todense().astype('float32')).to(args.device)
        # self.R = mmread(R).to(args.device)
        self.m, self.n = R.shape
        self.Sl = Parameter(torch.empty(self.n, self.n))
        nn.init.uniform_(self.Sl, 0, 1)
        self.Sl.data = self.Sl.data - torch.diag(torch.diag(self.Sl.data))
        self.Sg = Parameter(torch.empty(self.n, self.n))
        nn.init.uniform_(self.Sg, 0, 1)
        self.Sg.data = self.Sg.data - torch.diag(torch.diag(self.Sg.data))
        self.alpha = args.alpha
        self.beta = args.beta
        self.lamb = args.lamb
        self.c = args.c
        self.initer = args.initer
        self.F = Parameter(torch.rand(self.n, self.c))
        self.D = 1e-5

    def predict(self, R):
        return torch.matmul(R, self.Sl + self.Sg)

    def update_decouple(self):
        for i in tqdm(range(self.n)):
            sl = self.Sl.data[:, i]
            sg = self.Sg.data[:, i]
            Rr = self.RR[:, i]
            d = self.D[:, i]

            denominator = torch.matmul(self.RR, sl + sg) + d + self.beta * sl
            sl *= Rr
            sl /= denominator

            denominator = torch.matmul(self.RR, sl + sg) + self.alpha + self.beta * sg
            sg *= Rr
            sg /= denominator

    def update_S(self):
        Sl = self.Sl.data
        Sg = self.Sg.data

        denominator = torch.matmul(self.RR, Sl + Sg) + self.D + self.beta * Sl
        Sl *= self.RR
        Sl /= denominator

        denominator = torch.matmul(self.RR, Sl + Sg) + self.alpha + self.beta * Sg
        Sg *= self.RR
        Sg /= denominator

    def update_F(self):
        S = self.Sl.data
        S0 = (S + torch.t(S)) / 2
        L = torch.diag(torch.sum(S0, 0)) - S0
        L = (L + L.t()) / 2
        _, v = torch.symeig(L, True)
        F = v[:, 0:self.c]
        self.D = self.lamb * dist(F) + 1e-5
        self.F.data = F

    def object(self):
        Sl = self.Sl.data
        Sg = self.Sg.data
        S = Sl + Sg
        obj = torch.trace(self.RR) / 2
        obj -= trace(self.RR, S)
        obj += trace(S.t() @ self.RR, S) / 2
        obj += self.alpha * torch.sum(Sg)
        obj += trace(self.D, Sl) * self.lamb
        obj += (trace(Sl) + trace(Sg)) * self.beta / 2
        # obj += trace(torch.sum(S, 0) - 1) * self.gamma / 2
        return obj
