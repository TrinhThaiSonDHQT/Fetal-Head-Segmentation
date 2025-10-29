Now, define the Sorensen-Dice loss function:
Create a function dice_loss(y_true, y_pred):

1. Define a smooth factor (e.g., 1e-6) to prevent division by zero.
2. Flatten y_true and y_pred.
3. Calculate the intersection: tf.reduce_sum(y_true \* y_pred).
4. Calculate the Dice coefficient: (2. \* intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth).
5. The loss is 1 - dice_coefficient. Return the loss."
