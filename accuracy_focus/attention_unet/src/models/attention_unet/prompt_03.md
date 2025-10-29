"Let's build the complete 2D Attention U-Net model. Please create a function AttentionUNet that:

1. Takes input_shape (e.g., (256, 256, 1)) as an argument.
2. Defines the model using the Keras Functional API (or as a PyTorch nn.Module).
   Encoder (Down-sampling Path):
   • Block 1: Apply conv_block (e.g., 64 filters) to the input. Store the result as c1. Pass c1 to a MaxPooling2D (2x2) layer.
   • Block 2: Apply conv_block (e.g., 128 filters) to the pooled output. Store as c2. Pass c2 to MaxPooling2D.
   • Block 3: Apply conv_block (e.g., 256 filters). Store as c3. Pass to MaxPooling2D.
   • Block 4: Apply conv_block (e.g., 512 filters). Store as c4. Pass to MaxPooling2D.
   • Bottleneck: Apply conv_block (e.g., 1024 filters) to the pooled output.
   Decoder (Up-sampling Path with Attention Gates):
   • Decoder Block 4:
   o Up-sample the Bottleneck output (e.g., using UpSampling2D(2) or Conv2DTranspose). Let's call this g4.
   o Use the AttentionGate with x = c4 (from skip connection) and g = g4. Let's call this att4.
   o Concatenate att4 and g4.
   o Apply conv_block (512 filters).
   • Decoder Block 3:
   o Up-sample the output of Decoder Block 4 (UpSampling2D(2)). Let's call this g3.
   o Use the AttentionGate with x = c3 and g = g3. Let's call this att3.
   o Concatenate att3 and g3.
   o Apply conv_block (256 filters).
   • Decoder Block 2:
   o Up-sample the output of Decoder Block 3 (UpSampling2D(2)). Let's call this g2.
   o Use the AttentionGate with x = c2 and g = g2. Let's call this att2.
   o Concatenate att2 and g2.
   o Apply conv_block (128 filters).
   • Decoder Block 1:
   o Up-sample the output of Decoder Block 2 (UpSampling2D(2)). Let's call this g1.
   o Use the AttentionGate with x = c1 and g = g1. Let's call this att1.
   o Concatenate att1 and g1.
   o Apply conv_block (64 filters).
   Output Layer:
   • Apply a final 1x1 Conv2D to the output of Decoder Block 1.
   • The number of filters should equal the number of segmentation classes (e.g., 1 for binary segmentation).
   • Use a 'sigmoid' activation for binary segmentation or 'softmax' for multi-class.
