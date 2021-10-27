import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):

    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:,:, y, x,:,:] = img[:,:, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

def im2col_myown(input_data, filter_h, filter_w, stride=1, pad=0):
    
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N * out_h * out_w, C * filter_h * filter_w))
    
    for move_y in range(out_h):
        for move_x in range(out_w):
            start_x = move_x*stride
            start_y = move_y * stride
            row_number = move_x + move_y*out_w
            col[row_number::out_h * out_w,:] = \
                img[:,:, start_y:start_y + filter_h, start_x:start_x + filter_w].reshape(N, -1)

    return col

def col2im_myown(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    
    img = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    
    for column in range(C * filter_h * filter_w):
        c = column // (filter_h * filter_w)
        y = (column // filter_w) % (filter_h)
        x = column % filter_w
        y_max = y + stride * out_h
        x_max = x + stride * out_w
        img[:, c:c+1, y:y_max:stride, x:x_max:stride] += col[:, column].reshape(N, 1, out_h, out_w)
        
    return img[:, :, pad:H + pad, pad:W + pad]