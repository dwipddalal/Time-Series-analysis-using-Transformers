import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np

import torch.nn as nn
import math
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import numpy as np
CUDA_LAUNCH_BLOCKING = "1"

# df = pd.read_csv('demo2.csv')
def baap(df): 
    
    df.set_index('Period', inplace=True)
    df.columns = [ 'Expected' ]
    df.head(50)

    # df.to_csv('demo2.csv')

    sugardaddy = list(df.index[292:])


    num_zeros = 0
    for i in range(df.shape[0]):
        if df.iloc[i]['Expected'] == 0:
            num_zeros += 1
    print(num_zeros)

    df['Expected'].replace(to_replace = 0, value = int(df['Expected'].mean()), inplace=True)

    df = df.loc[~(df==0).all(axis=1)]
    df.head(30)


    df['Expected'] = df['Expected'].astype('float32')
    df.dropna(inplace=True)
    df.shape

    data = df.values.astype('float32')


    def split_dataset_into_seq(dataset, start_index=0, end_index=None, history_size=13, step=1):
        data = []
        start_index = start_index + history_size
        if end_index is None:
            end_index = len(dataset)
        for i in range(start_index, end_index):
            indices = range(i - history_size, i, step)
            data.append(dataset[indices])
        return np.array(data)



    def split_dataset(data, TRAIN_SPLIT=0.9, VAL_SPLIT=0.6):
        # normalization
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0)
        data = (data - data_mean) / data_std
        stats = (data_mean, data_std)

        data_in_seq = split_dataset_into_seq(data, start_index=0, end_index=None, history_size=13, step=1)

        # split between validation dataset and test set:
        train_data, val_data = train_test_split(data_in_seq, train_size=TRAIN_SPLIT, shuffle=True, random_state=123)
        val_data, test_data = train_test_split(val_data, train_size=VAL_SPLIT, shuffle=True, random_state=123)

        return train_data, val_data, test_data

    def split_fn(spiltttt):
        inputs = torch.tensor(spiltttt[:, :-1, :], device=device)
        targets = torch.tensor(spiltttt[:, 1:, :], device=device)
        return inputs, targets

    def data_to_dataset(train_data, val_data, test_data, batch_size=32, target_features=list(range(1))):
        x_train, y_train = split_fn(train_data)
        x_val, y_val = split_fn(val_data)
        x_test, y_test = split_fn(test_data)

        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
        test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        return train_loader, val_loader, test_loader


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data, val_data, test_data = split_dataset(data, 0.9, 0.4)
    train_data.shape

    train_data.shape


    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data = (data - data_mean) / data_std
    inputs, targets = split_fn(split_dataset_into_seq(data))
    print(inputs.shape, targets.shape)
    test_dataset = torch.utils.data.TensorDataset(inputs, targets)
    test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=32)


    blah_blah_Blah = test_dataset


    train_dataset, val_dataset, test_dataset = data_to_dataset(train_data, val_data, test_data)




    class MultiHeadAttention(nn.Module):

        def __init__(self, D, H):
            super(MultiHeadAttention, self).__init__()
            self.H = H # number of heads
            self.D = D # dimension
            self.wq = nn.Linear(D, D*H)
            self.wk = nn.Linear(D, D*H)
            self.wv = nn.Linear(D, D*H)
            self.dense = nn.Linear(D*H, D)

        def concat_heads(self, x):
            
            B, H, S, D = x.shape
            x = x.permute((0, 2, 1, 3)).contiguous()  # (B, S, H, D)
            x = x.reshape((B, S, H*D))   # (B, S, D*H)
            return x

        def split_heads(self, x):
            
            B, S, D_H = x.shape
            x = x.reshape(B, S, self.H, self.D)    # (B, S, H, D)
            x = x.permute((0, 2, 1, 3))  # (B, H, S, D)
            return x

        def forward(self, x, mask):

            q = self.wq(x)  # (B, S, D*H)
            k = self.wk(x)  # (B, S, D*H)
            v = self.wv(x)  # (B, S, D*H)

            q = self.split_heads(q)  # (B, H, S, D)
            k = self.split_heads(k)  # (B, H, S, D)
            v = self.split_heads(v)  # (B, H, S, D)

            attention_scores = torch.matmul(q, k.transpose(-1, -2)) #(B,H,S,S)
            attention_scores = attention_scores / math.sqrt(self.D)

            if mask is not None:
                attention_scores += (mask * -1e9)
                
            attention_weights = nn.Softmax(dim=-1)(attention_scores)
            scaled_attention = torch.matmul(attention_weights, v)  # (B, H, S, D)
            concat_attention = self.concat_heads(scaled_attention) # (B, S, D*H)
            output = self.dense(concat_attention)  # (B, S, D)

            return output, attention_weights


    B, S, H, D = 9, 11, 5, 8
    mha = MultiHeadAttention(D, H)
    out, att = mha.forward(torch.zeros(B, S, D), mask=None)
    out.shape, att.shape

    def get_angles(pos, i, D):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(D))
        return pos * angle_rates

    def positional_encoding(D, position=20, dim=3, device=device):
        angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                                np.arange(D)[np.newaxis, :],
                                D)
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        if dim == 3:
            pos_encoding = angle_rads[np.newaxis, ...]
        elif dim == 4:
            pos_encoding = angle_rads[np.newaxis,np.newaxis,  ...]
        return torch.tensor(pos_encoding, device=device)


    # function that implement the look_ahead mask for masking future time steps. 
    def create_look_ahead_mask(size, device=device):
        mask = torch.ones((size, size), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask  # (size, size)

    create_look_ahead_mask(6)


    class TransformerLayer(nn.Module):
        def __init__(self, D, H, hidden_mlp_dim, dropout_rate):
            super(TransformerLayer, self).__init__()
            self.dropout_rate = dropout_rate
            self.mlp_hidden = nn.Linear(D, hidden_mlp_dim)
            self.mlp_out = nn.Linear(hidden_mlp_dim, D)
            self.layernorm1 = nn.LayerNorm(D, eps=1e-9)
            self.layernorm2 = nn.LayerNorm(D, eps=1e-9)
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)

            self.mha = MultiHeadAttention(D, H)


        def forward(self, x, look_ahead_mask):
            
            attn, attn_weights = self.mha(x, look_ahead_mask)  # (B, S, D)
            attn = self.dropout1(attn) # (B,S,D)
            attn = self.layernorm1(attn + x) # (B,S,D)

            mlp_act = torch.relu(self.mlp_hidden(attn))
            mlp_act = self.mlp_out(mlp_act)
            mlp_act = self.dropout2(mlp_act)
            
            output = self.layernorm2(mlp_act + attn)  # (B, S, D)

            return output, attn_weights

    dl = TransformerLayer(16, 3, 32, 0.1)
    out, attn = dl(x=torch.zeros(5, 7, 16), look_ahead_mask=None)
    out.shape, attn.shape

    class Transformer(nn.Module):

        def __init__(self, num_layers, D, H, hidden_mlp_dim, inp_features, out_features, dropout_rate):
            super(Transformer, self).__init__()
            self.sqrt_D = torch.tensor(math.sqrt(D))
            self.num_layers = num_layers
            self.input_projection = nn.Linear(inp_features, D) # multivariate input
            self.output_projection = nn.Linear(D, out_features) # multivariate output
            self.pos_encoding = positional_encoding(D)
            self.dec_layers = nn.ModuleList([TransformerLayer(D, H, hidden_mlp_dim, 
                                            dropout_rate=dropout_rate
                                        ) for _ in range(num_layers)])
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x, mask):
            B, S, D = x.shape
            attention_weights = {}
            x = self.input_projection(x)
            x *= self.sqrt_D
            
            x += self.pos_encoding[:, :S, :]

            x = self.dropout(x)

            for i in range(self.num_layers):
                x, block = self.dec_layers[i](x=x,
                                            look_ahead_mask=mask)
                attention_weights['decoder_layer{}'.format(i + 1)] = block
            
            x = self.output_projection(x)
            
            return x, attention_weights # (B,S,S)


    # Test Forward pass on the Transformer: 
    transformer = Transformer(num_layers=3, D=32, H=1, hidden_mlp_dim=32,
                                        inp_features=1, out_features=1, dropout_rate=0.1)
    transformer.to(device)
    (inputs, targets) = next(iter(train_dataset))
                            
    S = inputs.shape[1]
    mask = create_look_ahead_mask(S)
    out, attn = transformer (x=inputs.float(), mask=mask)
    out.shape, attn["decoder_layer1"].shape


    transformer = Transformer(num_layers=3, D=32, H=4, hidden_mlp_dim=32, inp_features=1, out_features=1, dropout_rate=0.1).to(device)


    filename = '/home/dwip.dalal/Time_series_Analysis/website blueswan/finalized_model.sav'
    # pickle.dump(transformer, open(filename, 'wb'))
    loaded_model = pickle.load(open(filename, 'rb'))

    test_dataset = blah_blah_Blah 

    data = df.values.astype('float32')

    test_losses, test_preds  = [], []
    loaded_model.eval()
    for (x, y) in test_dataset:
        
        S = x.shape[-2]
        
        y_pred, _ = loaded_model(x, mask=create_look_ahead_mask(S))
        y = y.cpu().detach().numpy()*data.std(axis=0)+data.mean(axis=0)
        y_pred = y_pred.cpu().detach().numpy()*data.std(axis=0)+data.mean(axis=0)
        
        y = torch.tensor(y)
        y_pred = torch.tensor(y_pred)
        
        loss_test = torch.nn.MSELoss()(torch.tensor(y_pred),torch.tensor(y))  # (B,S)
        test_losses.append(loss_test.item())
        test_preds.append(y_pred.detach().cpu().numpy())
    test_preds = np.vstack(test_preds)
    np.mean(test_losses)
    squared_loss = np.mean(test_losses)


    x_test, _ = test_dataset.dataset.tensors
    x_test = x_test.cpu().detach().numpy()*data.std(axis=0)+data.mean(axis=0)



    x_test_array = []
    for i in range(11):
        x_test_array.append(math.ceil(x_test[i, 0, 0]))
    for i in range(x_test.shape[0]):
        x_test_array.append(math.ceil(x_test[i,11]))

    test_preds_array = []
    for i in range(11):
        test_preds_array.append(math.ceil(test_preds[i, 0, 0]))
    for i in range(x_test.shape[0]):
        test_preds_array.append(math.ceil(test_preds[i,11]))

    len(test_preds_array)
    print(test_preds_array)

    len(x_test_array)
    print(x_test_array)

    expresso = pd.DataFrame({'Actual sales': x_test_array, 'Forecasted sales': test_preds_array})
    expresso.tail(50)

    limit = math.sqrt(squared_loss)
    print(limit)

    expresso['max_sales'] = expresso['Forecasted sales'] + math.ceil(limit)
    expresso['min_sales'] = expresso['Forecasted sales'] - math.ceil(limit)

    expresso.head()

    expresso.shape

    expresso = expresso[292:]

    
    # import seaborn as sns
    # sns.set()

    X = sugardaddy[:186]

    y = expresso['Actual sales'][:120]
    y1 = expresso['Forecasted sales']
    y2 = expresso['max_sales']
    y3 = expresso['min_sales']

    fig, ax = plt.subplots(figsize=(20,10))
    ax.plot(X[:120],y,color='b', label='Actual Sales')
    ax.plot(X[120:],y1[120:],color='r', label='Forecasted Sales')
    ax.plot(X[120:],y2[120:],color='g', label='Max Sales')
    ax.plot(X[120:],y3[120:],color='#06283D', label='Min Sales')
    plt.fill_between(X[120:],y2[120:], y3[120:], color='grey', alpha=0.5)
    # ax.plot(x2,y2,color='g', label='Model')
    ax.set_xlabel('Period', fontsize=28)
    ax.set_ylabel('Sales', fontsize=28)

    # x_trend=['01-02-2021', '24-05-2022']
    # y_trend=[7, 7]   
    # ax.plot(x_trend, y_trend, color='#06283D', label="Trend")
    ax.legend(loc='upper left', fontsize=16)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    # date_form = DateFormatter("%d/%m/%Y")
    # ax.xaxis.set_major_formatter(date_form)   
    ax.tick_params(axis='both', which='major', labelsize=15, colors='white')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=3))
    plt.grid()
    plt.savefig('/home/dwip.dalal/Time_series_Analysis/website blueswan/foo.png')


    # finding accuracy 
    p = 0
    pa = 0
    for a,b,c in zip(expresso['Actual sales'], expresso['max_sales'], expresso['min_sales']):
        if a >= c and a <= b:
            p += 1
        else:
            pa += 1

    accuracy = p/(p+pa)

    return accuracy


baap(pd.read_csv('/home/dwip.dalal/Time_series_Analysis/website blueswan/demo2.csv'))