from interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/
            #print(type(self.lr))
            return parameter - self.lr*parameter_grad
            # your code here /\

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/
            updater.inertia = updater.inertia*self.momentum + self.lr* parameter_grad
            return parameter - updater.inertia
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        #self.forward_inputs = inputs
        inputs_relu = inputs.clip(0, None)
        return inputs_relu
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        grad_inputs = grad_outputs * (self.forward_inputs >= 0).astype(grad_outputs.dtype)

        #grad_o_prev = grad_pi @ self.W.T
        return grad_inputs
        # your code here /\


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, d)), output values

            n - batch size
            d - number of units
        """
        # your code here \/
        exp = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
        mask = exp > 1e6
        exp[mask] = 1e6

        outputs = exp/np.sum(exp, axis=-1, keepdims=True)
        return outputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of units
        """
        # your code here \/
        y = self.forward_outputs
        #j = np.diag(y) - y*y.T
        grad_inputs = y*(grad_outputs - np.sum(y*grad_outputs, axis = -1, keepdims=True))
        return grad_inputs
        # your code here /\


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        (input_units,) = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name="weights",
            shape=(output_units, input_units),
            initializer=he_initializer(input_units),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_units,),
            initializer=np.zeros,
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, c)), output values

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        outputs = inputs@self.weights.T + self.biases
        return outputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        grad_inputs = grad_outputs @ self.weights # (n, d)
        self.weights_grad[:] = grad_outputs.T @ self.forward_inputs # (c, d)
        self.biases_grad[:] = np.sum(grad_outputs, axis = 0) #(c,1)
        return grad_inputs
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((1,)), mean Loss scalar for batch

            n - batch size
            d - number of units
        """
        # your code here \/
        eps = np.finfo(float).eps
        y_pred = np.clip(y_pred, eps, 1.0)
        loss = -np.sum(y_gt * np.log(y_pred))/y_gt.shape[0]
        return np.array([loss])
        # your code here /\

    def gradient_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((n, d)), dLoss/dY_pred

            n - batch size
            d - number of units
        """
        # your code here \/
        eps = np.finfo(float).eps
        y_pred = np.clip(y_pred, eps, 1.0)
        grad_output = -y_gt/(y_pred * y_gt.shape[0])
        return grad_output
        # your code here /\


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(loss = CategoricalCrossentropy(), optimizer = SGDMomentum(lr=0.01, momentum=0.9))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(256, input_shape = (784,)))
    model.add(ReLU())
    model.add(Dense(128))
    model.add(ReLU())
    model.add(Dense(10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=64, epochs=10, x_valid=x_valid, y_valid=y_valid)

    # your code here /\
    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get("USE_FAST_CONVOLVE", False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # your code here \/
    if padding > 0:
        inputs_padded = np.pad(inputs, ((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')
    else:
        inputs_padded = inputs

    n,d,ih,iw = inputs.shape
    c,kd,kh,kw = kernels.shape
    oh = ih + 2*padding - kh + 1
    ow = iw + 2*padding - kw + 1

    outputs = np.zeros((n,c,oh,ow), dtype=inputs.dtype)

    for i in range (oh):
        for j in range(ow):
            current_patch =inputs_padded[...,i:i+kh,j:j+kw]
            for k in range(c):
                outputs[...,k,i,j] = np.sum(current_patch*kernels[k,:, ::-1, ::-1], axis=(1,2,3))


    return outputs
    # your code here /\


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name="kernels",
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_channels,),
            initializer=np.zeros,
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, c, h, w)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        # your code here \/
        self.forward_inputs = inputs 
        output = convolve(inputs, self.kernels, padding = (self.kernel_size - 1)//2) 
        output += self.biases.reshape(1,-1,1,1)
        return output
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        # your code here \/
        n, d, h, w = self.forward_inputs.shape
        c, _, kh, kw = self.kernels.shape
        pad = (self.kernel_size - 1) // 2

        self.biases_grad[:] = np.sum(grad_outputs, axis=(0, 2, 3))

        inputs_padded = np.pad(self.forward_inputs, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        self.kernels_grad.fill(0)
        for i in range(kh):
            for j in range(kw):
                patch = inputs_padded[:, :, i:i+h, j:j+w]  # (n, d, h, w)
                self.kernels_grad[:, :, i, j] = np.tensordot(
                    grad_outputs, patch, axes=([0, 2, 3], [0, 2, 3])
                )

        self.kernels_grad[:] = self.kernels_grad[:, :, ::-1, ::-1]

        flipped = self.kernels[:, :, ::-1, ::-1]
        grad_inputs = convolve(grad_outputs, flipped.transpose(1, 0, 2, 3), padding=pad)

        return grad_inputs
        # your code here /\


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode="max", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {"avg", "max"}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, ih, iw)), input values

        :return: np.array((n, d, oh, ow)), output values

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        # your code here \/
        n, d, ih, iw = inputs.shape
        p = self.pool_size
        oh, ow = ih // p, iw // p

        blocks = inputs.reshape(n, d, oh, p, ow, p).transpose(0, 1, 2, 4, 3, 5)  # (n,d,oh,ow,p,p)
        block_flat = blocks.reshape(n, d, oh, ow, p*p)

        if self.pool_mode == "max":
            max_idx = np.argmax(block_flat, axis=-1)
            self.max_indices = max_idx
            outputs = np.take_along_axis(block_flat, max_idx[..., None], axis=-1).reshape(n, d, oh, ow)
        else:  
            outputs = block_flat.mean(axis=-1)

        self.forward_inputs = inputs
        self.forward_outputs = outputs
        return outputs.astype(inputs.dtype)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

        :return: np.array((n, d, ih, iw)), dLoss/dInputs

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        # your code here \/
        n, d, ih, iw = self.forward_inputs.shape
        p = self.pool_size
        oh, ow = grad_outputs.shape[2], grad_outputs.shape[3]

        grad_inputs = np.zeros_like(self.forward_inputs)

        if self.pool_mode == "max":
            grad_blocks = grad_inputs.reshape(n, d, oh, p, ow, p).transpose(0, 1, 2, 4, 3, 5)
            grad_blocks_flat = grad_blocks.reshape(n, d, oh, ow, p*p)
            grad_blocks_flat[np.arange(n)[:, None, None, None],
                             np.arange(d)[None, :, None, None],
                             np.arange(oh)[None, None, :, None],
                             np.arange(ow)[None, None, None, :],
                             self.max_indices] = grad_outputs
            grad_inputs = grad_blocks_flat.reshape(n, d, oh, ow, p, p).transpose(0, 1, 2, 4, 3, 5).reshape(n, d, ih, iw)
        else:  
            grad_block = grad_outputs / (p*p)
            grad_inputs = np.repeat(np.repeat(grad_block, p, axis=2), p, axis=3)

        return grad_inputs
        # your code here /\


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name="beta",
            shape=(input_channels,),
            initializer=np.zeros,
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name="gamma",
            shape=(input_channels,),
            initializer=np.ones,
        )

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, d, h, w)), output values

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/
        n, d, h, w = inputs.shape
        axis = (0, 2, 3) 
        eps = np.finfo(float).eps

        if self.is_training:
            batch_mean = inputs.mean(axis=axis)  # (d,)
            batch_var = inputs.var(axis=axis)    # (d,)

            x_centered = inputs - batch_mean[None, :, None, None]
            inv_std = 1.0 / np.sqrt(batch_var + eps)
            x_norm = x_centered * inv_std[None, :, None, None]

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            self.forward_centered_inputs = x_centered
            self.forward_inverse_std = inv_std
            self.forward_normalized_inputs = x_norm

            outputs = self.gamma[None, :, None, None] * x_norm + self.beta[None, :, None, None]

        else:
            x_centered = inputs - self.running_mean[None, :, None, None]
            x_norm = x_centered / np.sqrt(self.running_var + eps)[None, :, None, None]
            outputs = self.gamma[None, :, None, None] * x_norm + self.beta[None, :, None, None]

        return outputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/
        n, d, h, w = grad_outputs.shape
        m = n * h * w  

        x_norm = self.forward_normalized_inputs
        inv_std = self.forward_inverse_std
        x_centered = self.forward_centered_inputs

        self.gamma_grad[:] = np.sum(grad_outputs * x_norm, axis=(0, 2, 3))
        self.beta_grad[:] = np.sum(grad_outputs, axis=(0, 2, 3))

        dx_norm = grad_outputs * self.gamma[None, :, None, None]

        dx = (1.0 / m) * inv_std[None, :, None, None] * (
            m * dx_norm
            - np.sum(dx_norm, axis=(0, 2, 3))[None, :, None, None]
            - x_norm * np.sum(dx_norm * x_norm, axis=(0, 2, 3))[None, :, None, None]
        )

        return dx
        # your code here /\


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (int(np.prod(self.input_shape)),)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, (d * h * w))), output values

            n - batch size
            d - number of input channels
            (h, w) - image shape
        """
        # your code here \/
        n, d, h, w = inputs.shape
        return inputs.reshape(n, -1)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of units
            (h, w) - input image shape
        """
        # your code here \/
        n = grad_outputs.shape[0]
        d,h,w = self.input_shape
        return grad_outputs.reshape(n,d,h,w)
        # your code here /\


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        if self.is_training:
            self.forward_mask = (np.random.uniform(size=inputs.shape)>self.p)
            outputs = inputs * self.forward_mask
        else:
            outputs = inputs * (1-self.p)
        return outputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        if self.is_training:
            grad_inputs = grad_outputs * self.forward_mask
        else:
            grad_inputs = grad_outputs * (1 - self.p)
        return grad_inputs
        # your code here /\


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(loss = CategoricalCrossentropy(), optimizer = SGDMomentum(lr=0.01, momentum=0.9))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Conv2D(16, input_shape = (3,32,32)))
    model.add(ReLU())
    model.add(Pooling2D())
    model.add(Conv2D(32))
    model.add(ReLU())
    model.add(Pooling2D())
    model.add(Conv2D(64))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(ReLU())
    model.add(Dropout(0.4))
    model.add(Dense(64))
    model.add(ReLU())
    model.add(Dense(10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train, y_train, batch_size=64, epochs=20, x_valid=x_valid, y_valid=y_valid)

    # your code here /\
    return model


# ============================================================================
