import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from torch.autograd import Variable, Function

LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

# Unsupervised Domain Adaptation by Backpropagation.ICML 2015.
# Gradient Reversal Layer
class ReverseLayerF(Function):
    def forward(self, x):
        self.lambd = 1
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x):
    return ReverseLayerF()(x)

class CNN_Fusion(nn.Module):
    def __init__(self, args, W):
        super(CNN_Fusion, self).__init__()
        self.args = args
        self.event_num = args.event_num
        vocab_size = args.vocab_size
        emb_dim = args.embedding_dim
        # C = args.class_num
        self.lambd = args.lambd
        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embedding_dim
        # self.social_size = 19

        # TextCNN训练文本
        channel_in = 1
        filter_num = 20
        window_size = [1, 2, 3, 4]
        self.convs = nn.ModuleList([nn.Conv2d(channel_in, filter_num, (K, emb_dim)) for K in window_size])
        self.fc1 = nn.Linear(len(window_size) * filter_num, self.hidden_size)

        self.dropout = nn.Dropout(args.dropout)

        # TEXT RNN
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.embed.weight = nn.Parameter(torch.from_numpy(W))
        self.lstm = nn.LSTM(self.lstm_size, self.lstm_size)
        self.text_fc = nn.Linear(self.lstm_size, self.hidden_size)
        self.text_encoder = nn.Linear(emb_dim, self.hidden_size)

        # vgg训练图像
        vgg_19 = vgg19(pretrained=True)
        for param in vgg_19.parameters():
            param.requires_grad = False

        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.vgg = vgg_19
        self.image_fc1 = nn.Linear(num_ftrs, self.hidden_size)
        # self.image_fc2 = nn.Linear(512, self.hidden_size)
        self.image_adv = nn.Linear(self.hidden_size,  int(self.hidden_size))
        self.image_encoder = nn.Linear(self.hidden_size, self.hidden_size)

        #  社交上下文(待选)
        # self.social = nn.Linear(self.social_size, self.hidden_size)

        # attention?
        self.attention_layer = nn.Linear(self.hidden_size, emb_dim)

        # Class  Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(2 * self.hidden_size, 3))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm2d(100))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        # self.class_classifier.add_module('c_fc2', nn.Linear(self.hidden_size, 2))
        # self.class_classifier.add_module('c_bn2', nn.BatchNorm2d(self.hidden_size))
        # self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        # self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        # Event Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(2 * self.hidden_size, self.hidden_size))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.hidden_size, self.event_num))
        self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (to_var(torch.zeros(1, batch_size, self.lstm_size)),
                to_var(torch.zeros(1, batch_size, self.lstm_size)))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
        # x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, text, image, mask):
        # image
        image = self.vgg(image)
        image = F.leaky_relu(self.image_fc1(image))

        # CNN
        text = self.embed(text)
        text = text * mask.unsqueeze(2).expand_as(text)
        # print(torch.min(text))
        text = text.unsqueeze(1)
        text = [F.leaky_relu(conv(text)).squeeze(3) for conv in self.convs]  # [(N,hidden_dim,W), ...]*len(window_size)
        # text = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,hidden_dim), ...]*len(window_size)
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text = torch.cat(text, 1)
        text = F.leaky_relu(self.fc1(text))
        text_image = torch.cat((text, image), 1)

        # Fake or real
        class_output = self.class_classifier(text_image)
        # Domain (which Event )
        reverse_feature = grad_reverse(text_image)
        domain_output = self.domain_classifier(reverse_feature)

        # ### Multimodal
        # text_reverse_feature = grad_reverse(text)
        # image_reverse_feature = grad_reverse(image)
        # text_output = self.modal_classifier(text_reverse_feature)
        # image_output = self.modal_classifier(image_reverse_feature
        return class_output, domain_output

class TextCNN(nn.Module):
    def __init__(self, args, W):
        super(TextCNN, self).__init__()
        self.args = args
        self.event_num = args.event_num
        vocab_size = args.vocab_size
        emb_dim = args.embedding_dim
        # C = args.class_num
        self.lambd = args.lambd
        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embedding_dim

        # TextCNN训练文本
        channel_in = 1
        filter_num = 20
        window_size = [1, 2, 3, 4]
        self.convs = nn.ModuleList([nn.Conv2d(channel_in, filter_num, (K, emb_dim)) for K in window_size])
        self.fc1 = nn.Linear(len(window_size) * filter_num, self.hidden_size)

        self.dropout = nn.Dropout(args.dropout)

        # TEXT RNN
        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.embed.weight = nn.Parameter(torch.from_numpy(W))
        self.lstm = nn.LSTM(self.lstm_size, self.lstm_size)
        self.text_fc = nn.Linear(self.lstm_size, self.hidden_size)
        self.text_encoder = nn.Linear(emb_dim, self.hidden_size)

        # attention?
        self.attention_layer = nn.Linear(self.hidden_size, emb_dim)

        # Class  Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.hidden_size, 3))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        # Event Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.hidden_size, self.hidden_size))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.hidden_size, self.event_num))
        self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (to_var(torch.zeros(1, batch_size, self.lstm_size)),
                to_var(torch.zeros(1, batch_size, self.lstm_size)))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
        # x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, text, mask):
        # CNN
        text = self.embed(text)
        text = text * mask.unsqueeze(2).expand_as(text)
        # print(torch.min(text))
        text = text.unsqueeze(1)
        text = [F.leaky_relu(conv(text)).squeeze(3) for conv in self.convs]  # [(N,hidden_dim,W), ...]*len(window_size)
        # text = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in text]  # [(N,hidden_dim), ...]*len(window_size)
        text = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in text]
        text = torch.cat(text, 1)
        text = F.leaky_relu(self.fc1(text))

        # Fake or real
        class_output = self.class_classifier(text)
        # Domain (which Event )
        reverse_feature = grad_reverse(text)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

def to_var(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [Variable(y, volatile=volatile) for y in x]
    return Variable(x.cuda(), volatile=volatile)

def to_np(x):
    return x.data.cpu().numpy()