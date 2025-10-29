Please define a function or class for the 2D Attention Gate module, as described in the 'Attention U-Net' paper (https://arxiv.org/pdf/1804.03999)

This module should:

1. Be named AttentionGate.
2. Accept two 2D feature map inputs:
   • x: The feature map from the U-Net skip connection (e.g., from the encoder).
   • g: The gating signal from the coarser scale (e.g., from the decoder).
3. Define an internal number of filters, F_int.
4. Perform the following operations based on Figure 2:
   • Linearly transform g using a 2D 1x1 convolution (Conv2D(filters=F_int, strides=1, padding='same')) (W_g). Let's call this phi_g. This is at the coarse resolution.
   • Linearly transform x using a 2D convolution (W_x) with filters=F_int and a stride (e.g., strides=2) that downsamples it to match the spatial dimensions of phi_g. Let's call this theta_x_down.
   • Add phi_g and theta_x_down.
   • Pass the sum through a ReLU activation function (sigma_1).
   • Linearly transform the result using a 2D 1x1 convolution (Conv2D(filters=1, strides=1, padding='same')) (psi).
   • Pass this through a Sigmoid activation function (sigma_2) to get the coarse attention coefficients, alpha_coarse.
   • Resample alpha_coarse (e.g., using UpSampling2D with bilinear interpolation) to match the original spatial dimensions of x. Let's call this alpha_fine.
   • Multiply the original input x element-wise by alpha_fine (which may need to be repeated across the channel dimension) to get the final output x_hat.

5. Return the final output x_hat."
