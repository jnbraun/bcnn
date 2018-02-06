This is an example of lifted structure loss function which is presented in [Deep Metric Learning via Lifted Structured Feature Embedding](https://arxiv.org/pdf/1511.06452.pdf)

Here are some notes:
1. There is no classify output layer, feature embedded layer(the last fully connected layer) should be followed by cost layer

2. Because lifted structure loss computes the loss between each class in the batch, you need to arrange the batch data includes at least two classes. according to the paper, you should put as many as possibile positive sample in the batch and put the number of negative sample equal to the number of positive sample.

For example, the batch size = 128, you can put 64 positive samples and 64 negative samples, all positive samples are the same class, each negative sample does not has to be the same class.