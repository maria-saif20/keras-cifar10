# keras-cifar10
Let's break down the code:

1. Visualizing Data:
   - First, matplotlib is imported to visualize the data.
   - `x_train.shape` shows the shape of the training data, which is `(50000, 32, 32, 3)`, indicating there are 50,000 images of size 32x32 pixels with 3 color channels (RGB).
   - An example image from the training set is extracted and displayed. The image is a 32x32 RGB image represented as a NumPy array with values ranging from 0 to 255.
   - `y_train` and `y_test` are the labels corresponding to the training and testing images, respectively.

2. Data Preprocessing:
   - `to_categorical` from `tensorflow.keras.utils` is imported to convert the labels to one-hot encoded vectors.
   - The shape of `y_train` is `(50000, 1)`, indicating it's a 2D array with 50,000 rows and 1 column.
   - `to_categorical` is applied to `y_train`, resulting in `y_example`, which now has a shape of `(50000, 10)`, where each label is converted to a one-hot encoded vector.
   - Similarly, `to_categorical` is applied to `y_test` to get `y_cat_train` and `y_cat_test`.
   - The minimum and maximum pixel values of the image are computed and then the images are normalized by dividing by 216, bringing the pixel values to the range [0, 1].
   - After normalization, the maximum and minimum values of the scaled image are checked.

3. Reshaping Data:
   - The shape of `x_train` is `(50000, 32, 32, 3)`. To fit into the model, it's reshaped to `(50000, 32, 32, 3, 1)`, adding an extra dimension representing the channel.
   - Similar reshaping is done for `x_test`.

4. Model Building:
   - A convolutional neural network (CNN) model is built using Keras Sequential API.
   - Two convolutional layers with ReLU activation followed by max-pooling are added, along with a flatten layer to convert 2D output to 1D, and two dense layers.
   - The model is compiled with categorical cross-entropy loss, RMSprop optimizer, and accuracy metric.
   - `model.summary()` provides a summary of the model architecture.

5. Model Training:
   - The model is trained using `model.fit()` on the preprocessed training data (`x_train` and `y_cat_train`) for 20 epochs.

6. Model Evaluation:
   - The model is evaluated using `model.evaluate()` on the preprocessed testing data (`x_test` and `y_cat_test`).
   - The evaluation results in loss and accuracy values.

7. Model Prediction and Classification Report:
   - Predictions are made using `model.predict()` on the testing data.
   - `classification_report` from `sklearn.metrics` is used to generate a classification report comparing actual labels (`y_test`) with predicted labels.

Overall, this code demonstrates loading, preprocessing, visualizing, building, training, evaluating, and analyzing a convolutional neural network model for image classification on the CIFAR-10 dataset.
