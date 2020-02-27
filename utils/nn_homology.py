import numpy as np
import networkx as nx
import torch
import scipy.sparse

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
#     assert (H + 2 * padding - field_height) % stride == 0
#     assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

def conv_layer_as_matrix(X, X_names, W, stride, padding):
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    X_names_col = im2col_indices(X_names, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    out = W_col @ X_col
    out = out.reshape(n_filters, int(h_out), int(w_out), n_x)
    out = out.transpose(3, 0, 1, 2)

    return out, X_col, W_col, X_names_col

def inverse_abs(x):
    return np.abs(1/x)

def add_conv(G, input_size, p, name_this, name_next, stride, padding, weight_func=inverse_abs, next_linear=False):
    '''adds convolutional layer to graph and returns updated graph'''
    conv_format = '{}_{}_{}'
    input_channels = p.shape[1]
    X = np.ones(input_size)
    for c in range(input_channels):
        print('Channel: {}'.format(c))
        X_names = np.arange(X.shape[2]*X.shape[3]).reshape((1,1,X.shape[2],X.shape[3]))
        tx = X[:,c,:,:].reshape((X.shape[0],1,X.shape[2],X.shape[3]))
        # convert to matrix information
        mat, X_col, W_col, xnames = conv_layer_as_matrix(tx,X_names,p[:,c,:,:].reshape((p.shape[0],1,p.shape[2],p.shape[3])),stride,padding)
        for f in range(W_col.shape[0]):
            for row in range(X_col.shape[0]):
                for col in range(X_col.shape[1]):
                    v = W_col[f,row]
                    if v != 0:
                        if next_linear:
                            # next layer is linear
                            G.add_edge(conv_format.format(name_this,c,xnames[row,col]),conv_format.format(name_next,0,int((X_col.shape[1]*c) + (f*X_col.shape[1]) + col)), weight=weight_func(v))
                        else:
                            # next layer is conv
                            G.add_edge(conv_format.format(name_this,c,xnames[row,col]),conv_format.format(name_next,f,col), weight=weight_func(v))
    input_size = mat.shape
    return G, input_size

def add_mp(G, input_size, name_this, name_next, kernel_size, stride, padding, next_linear=False):
    '''adds max pooling layer to graph and returns updated graph'''
    conv_format = '{}_{}_{}'
    p = np.ones((input_size[0],input_size[1],kernel_size[0],kernel_size[1]))
    input_channels = p.shape[1]
    # next layer also conv
    X = np.ones(input_size)
    for c in range(input_channels):
        print('Channel: {}'.format(c))
        X_names = np.arange(X.shape[2]*X.shape[3]).reshape((1,1,X.shape[2],X.shape[3]))
        tx = X[:,c,:,:].reshape((X.shape[0],1,X.shape[2],X.shape[3]))
        # convert to matrix information
        mat, X_col, W_col, xnames = conv_layer_as_matrix(tx,X_names,p[:,c,:,:].reshape((p.shape[0],1,p.shape[2],p.shape[3])),stride,padding)
        for f in range(W_col.shape[0]):
            for row in range(X_col.shape[0]):
                for col in range(X_col.shape[1]):
                    node_name = conv_format.format(name_this,c,xnames[row,col])
                    ews = sorted(G.in_edges(node_name, data=True), key=lambda x: x[2]['weight'])
                    if len(ews) > 0:
                        v = ews[0][2]['weight']
                        if v != 0:
                            if next_linear:
                                G.add_edge(conv_format.format(name_this,c,xnames[row,col]),conv_format.format(name_next,0,int((X_col.shape[1]*c) + (f*X_col.shape[1]) + col)), weight=v)
                            else:
                                G.add_edge(conv_format.format(name_this,c,xnames[row,col]),conv_format.format(name_next,f,col), weight=v)
    input_size = [mat.shape[0], input_channels, mat.shape[2], mat.shape[3]]
    return G, input_size

def add_linear_linear(G, p, name_this, name_next, weight_func=inverse_abs):
    '''adds linear layer to graph and returns updated graph'''
    conv_format = '{}_{}_{}'
    for row in range(p.shape[1]):
        for col in range(p.shape[0]):
            v = p[col,row]
            if v != 0:
                G.add_edge(conv_format.format(name_this,0,row),conv_format.format(name_next,0,col), weight=weight_func(v))
    return G

def to_directed_networkx(params, input_size):
    # store all network info here
    G = nx.DiGraph()

    # assume last layer linear
    for l in range(len(params)-1):

        param = params[l]
        # need to look ahead at next layer to get naming correct
        param_next = params[l+1]

        print('Layer: {}'.format(param['name']))

        if param['layer_type'] == 'Conv2d':

            if param_next['layer_type'] == 'Conv2d' or param_next['layer_type'] == 'MaxPool2d':

                G, input_size = add_conv(G, input_size, param['param'], param['name'], param_next['name'], param['stride'], param['padding'], next_linear=False)

            elif param_next['layer_type'] == 'Linear':

                G, input_size = add_conv(G, input_size, param['param'], param['name'], param_next['name'], param['stride'], param['padding'], next_linear=True)

        elif param['layer_type'] == 'MaxPool2d':

            if param_next['layer_type'] == 'Conv2d':

                G, input_size = add_mp(G, input_size, param['name'], param_next['name'], param['kernel_size'], param['stride'], param['padding'], next_linear=False)

            if param_next['layer_type'] == 'Linear':

                G, input_size = add_mp(G, input_size, param['name'], param_next['name'], param['kernel_size'], param['stride'], param['padding'], next_linear=True)

        elif param['layer_type'] == 'Linear':
            # linear layer
            G = add_linear_linear(G, param['param'], param['name'], param_next['name'])

        else:
            raise ValueError('Layer type not implemented ')

    # add in last layer
    print('Layer: {}'.format(params[-1]['name']))
    G = add_linear_linear(G, params[-1]['param'], params[-1]['name'], 'Output')

    return G
