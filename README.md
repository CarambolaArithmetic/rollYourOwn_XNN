# rollYourOwn_XNN
A toy library for neural network backpropogation written in Python. 
(Created for Arthur Redfern's CS6301 Convolutional Neural Networks special topics course.)

This project includes two demos, `demo1.py` and `demo2.py`. When run, each demo will begin training a
different neural network on the mnist data set. The first demo is a basic multi-level neural network
demonstrating the library's relu, softmax, and Matrix multiplication operations. The second is a
larger and more complex model which features convolutions and max pooling.

Since all computation is done on the CPU in basic Python, expect each demo to take around 2 hours
(possibly more) to complete. At the end of each demo, the script will report accuracy on the test
data set. The first demo (basic deep NN) always reports around 93% accuracy. The second demo (CNN)
on the other hand seems to be sensitive to training conditions, and only converges to a solution
roughly 44% of the time. If it has failed to learn, it rarely reports an accuracy over 15%. When it
does learn successfully, however, it usually obtains around 99% accuracy on the test data set.
