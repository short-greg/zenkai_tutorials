import torch

def compute_conv_output_size(input_size, kernel_size, stride, padding):
    """
    Compute the output size after applying a convolution or deconvolution layer.
    This function does not modify any class state.
    """
    padding_height, padding_width = padding  # Unpack padding tuple

    output_height = (input_size[0] + 2 * padding_height - kernel_size[0]) // stride[0] + 1
    output_width = (input_size[1] + 2 * padding_width - kernel_size[1]) // stride[1] + 1

    return [output_height, output_width]


def compute_deconv_output_size(input_size, kernel_size, stride, padding):
    """
    Compute the output size after applying a convolution or deconvolution layer.
    This function does not modify any class state.
    """
    padding_height, padding_width = padding  # Unpack padding tuple

    output_height = (input_size[0] - 1) * stride[0] - 2 * padding_height + kernel_size[0]
    output_width = (input_size[1] - 1) * stride[1] - 2 * padding_width + kernel_size[1]

    return [output_height, output_width]


def calculate_conv_padding(input_size, kernel_size, stride):
    """
    Calculate the padding needed based on the input size, kernel size, and stride.
    Add padding to maintain the output size consistency with kernel_size > stride.
    """
    # Calculate required padding based on kernel size and stride
    # padding_height = max((stride - (input_size[0] % stride)) % stride, 0)
    # padding_width = max((stride - (input_size[1] % stride)) % stride, 0)

    # stride = 3
    # kw = 4
    # input_size % stride => 0
    # (8 - 6) ... 0, 3.. 8 % stride = 1.. 1.. If stride is 1.. don't need to worry
    # 8 % stride... 8 % stride = 2.. inputs - 2 = 6... kw - 

    # Ensure padding accounts for kernel size
    # 9 - 3 = 6 % 2 
    padding0 = (input_size[0] - kernel_size[0]) % stride[0]
    padding1 = (input_size[1] - kernel_size[1]) % stride[1]
    return (padding0, padding1)


def calculate_deconv_out_padding(deconv_out_size, output_size):
    """
    Calculate the padding needed based on the input size, kernel size, and stride.
    Add padding to maintain the output size consistency with kernel_size > stride.
    """
    padding0 = output_size[0] - deconv_out_size[0]
    padding1 = output_size[1] - deconv_out_size[1]
    return (padding0, padding1)


class AmplifyGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, by: float=1.0):
        """
        Amplify the gradient by a certain amount.
        """
        ctx.amplify_by = by
        return input

    @staticmethod
    def backward(ctx, grad_output):
        """
        
        """
        if ctx.amplify_by is None:
            return torch.zeros_like(grad_output), None
        return grad_output * ctx.amplify_by, None
