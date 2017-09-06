import torch.nn as nn
import torch.nn.functional as F

# D_in  = Input Dimension
# H = Hidden Dimension
# D_out = Output dimension

D_in, H, D_out = 14, 5, 14

# Our NeuralNet module with simple one layer encoder
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.Encode = nn.Linear(D_in, H)
        self.Decode = nn.Linear(H,D_out)
    def forward(self, input, future = 0):
        o_t = F.relu(self.Encode(input))
        o_t = F.sigmoid(self.Decode(o_t))
        return o_t

# Our NeuralNet module with multi layer encoder

H1 = 10
class NeuralNet_multilayerencode(nn.Module):
    def __init__(self):
        super(NeuralNet_multilayerencode, self).__init__()
        self.Encode_1 = nn.Linear(D_in, H)
        self.Encode_2 = nn.Linear(H,H1)
        self.Decode = nn.Linear(H1,D_out)
    def forward(self, input, future = 0):
        o_t = F.relu(self.Encode_1(input))
        o_t = F.relu(self.Encode_2(o_t))
        o_t = F.sigmoid(self.Decode(o_t))
        return o_t
