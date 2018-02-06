This is an example of lifted structured similarity softmax loss function which as presented in [Deep Metric Learning via Lifted Structured Feature Embedding](https://arxiv.org/pdf/1511.06452.pdf)

Some notes:
1. There is no classify output layer, the feature embedding layer(the last fully connected layer) should be followed by the cost layer with lifted structured similarity softmax loss.

2. As the lifted structured similarity softmax loss computes the loss between each class in the batch, you will need to arrange the batch data includes at least two classes. According to the paper, you should put as many as possible positive samples in the batch and set the number of negative samples equal to the number of positive samples.

For example, if the batch size = 128, you can put 64 positive samples and 64 negative samples, all positive samples have to belong to the same class, wether each negative sample does not necessarily arise from the same class.