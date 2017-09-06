import numpy as np
import torch
import data_processing
from data_processing import next_batch
from network_module import NeuralNet,NeuralNet_multilayerencode
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


dataread = data_processing.Datareading(dataframe = 'dataset.csv')
Independent_data,Response = dataread.dataprocessed()
scaler,df  = data_processing.__Processing__(df = Independent_data,process='min-max')



trainx = df[:5000,:]
testX  = df[5000:,:]
net = NeuralNet_multilayerencode()
net.double()
criterion = nn.MSELoss(size_average=True)
optimizer = optim.Adadelta(net.parameters())

batch_size = 100
EPOCHS = 100
lr = 0.01
loss_summary = []
import time
start = time.time()
val_data = Variable(torch.from_numpy(testX))
for epoch in range(0, EPOCHS):
    optimizer.zero_grad()
    o=[]
    for x_batch in next_batch(trainx,batch_size):
        inputs = Variable(torch.from_numpy(x_batch))
        target = Variable(torch.from_numpy(x_batch))
        out = net(inputs)
        loss = criterion(out,target)
        net.zero_grad()
        loss.backward()
        for param in net.parameters():
             param.data -= lr * param.grad.data
    testout = net(val_data)
    val_loss_entire = criterion(testout, val_data)
    val_loss_entire = np.round(val_loss_entire.data.numpy()[0],5)
    training_loss = np.round(loss.data.numpy()[0],5)
    loss_summary.append([training_loss,val_loss_entire])
    print("Epoch: {}, trainingloss: {:.10f},entirevalidationloss: {:.10f}".format(epoch, training_loss,val_loss_entire))
end  = time.time()
tt = end - start
print("Timetaken:",tt)


