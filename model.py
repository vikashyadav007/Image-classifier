import time
from collections import OrderedDict

import torch
from torch import nn, optim
from torchvision import models
from tqdm import tqdm



class FlowerModel():
    def __init__(self, base_model='VGG19', hidden_units=4096,
                 learning_rate=0.002, use_gpu=False):
        self.base_model = base_model
        self.hidden_units = hidden_units
        self.use_gpu = use_gpu
        if not use_gpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")

        self._create_model(base_model, hidden_units, learning_rate)
        self.criterion = None

        
    def _create_model(self, base_model, hidden_units, learning_rate=0.005):
        supported_base_models = {
            'vgg13': models.vgg13,
            'vgg13_bn': models.vgg13_bn,
            'vgg16': models.vgg16,
            'vgg16_bn': models.vgg16_bn,
            'vgg19': models.vgg19,
            'vgg19_bn': models.vgg19_bn,
            'densenet121': models.densenet121,
            'densenet169': models.densenet169
        }
        input_features_dict = {
            'vgg13': 25088,
            'vgg13_bn': 25088,
            'vgg16': 25088,
            'vgg16_bn': 25088,
            'vgg19': 25088,
            'vgg19_bn': 25088,
            'densenet121': 1024,
            'densenet169': 1024
        }
        base_model_function = supported_base_models.get(base_model, None)

        if not base_model_function:
            print("Not a valid base_model. Try: {}".format(
                ','.join(supported_base_models.keys())))

        self.history = base_model_function(pretrained=True)

        input_features = input_features_dict[base_model]

        # Freeze weights of feature extractor.
        for param in self.history.parameters():
            param.requires_grad = False

        self.history.base_model = base_model
        self.history.hidden_units = hidden_units
        
        classifier = nn.Sequential(OrderedDict([
            ('input_1', nn.Linear(input_features, hidden_units,bias= True)),
            ('relu_1', nn.ReLU()),
            ('dropout_1', nn.Dropout(p=0.45)),
            ('layer_2', nn.Linear(hidden_units, 102,bias=True)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        self.history.classifier = classifier
        self.loss = nn.NLLLoss()
        self.optimizer = optim.Adam(
            self.history.classifier.parameters(), lr=learning_rate)

    @staticmethod
    def loading_checkpoint(self,path):
        # Load the saved file
        checkpoint = torch.load("checkpoint_1.pth")
    
        # Download pretrained model
        history = self.history.vgg19(pretrained=True)
    
        # Freeze parameters so we don't backprop through them
        for param in self.history.parameters(): 
            param.requires_grad = False
    
        # Load stuff from checkpoint
        history.class_to_idx = checkpoint['class_to_idx']
        history.classifier = checkpoint['classifier']
        history.load_state_dict(checkpoint['state_dict'])

        return history


    def predict(self,train_loader):
        correct_pred = 0
        total = 0
        self.history.to('cuda')
        with torch.no_grad():
            self.history.eval()
            for i in tqdm(train_loader):
                images, labels = i
                images, labels = images.to('cuda'), labels.to('cuda')
                opt = self.history(images)
                _, predicted = torch.max(opt.data, 1)
                total += labels.size(0)
                correct_pred += (predicted == labels).sum().item()

        print('Accuracy achieved on test images is: %d%%' % (100 * correct_pred / total))

        

    def save_model(self, filepath,train_datasets):
        self.history.class_to_idx = train_datasets.class_to_idx

        checkpoint = {
             'classifier': self.history.classifier,
             'class_to_idx': self.history.class_to_idx,
             'state_dict': self.history.state_dict()}

        torch.save(checkpoint, filepath)

    def valid_fun(self, test_loader):
        test_loss = 0
        accuracy = 0
    
        for i, (inputs, labels) in enumerate(test_loader):
        
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            opt = self.history.forward(inputs)
            test_loss += self.loss(opt, labels).item()
        
            temp = torch.exp(opt)
            equality = (labels.data == temp.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
    
        return test_loss, accuracy


    def train(self, save_dir, train_loader, valid_loader, test_loader,train_datasets, epochs):
        stp = 0
        self.history.to(self.device)
        
        for e in tqdm(range(epochs)):
            continuous_loss = 0
            self.history.train() 
    
            for i, (inputs, labels) in tqdm(enumerate(train_loader)):
                stp += 1
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                self.optimizer.zero_grad()
        
                opt = self.history.forward(inputs)
                tmp_loss = self.loss(opt, labels)
                tmp_loss.backward()
                self.optimizer.step()
        
                continuous_loss += tmp_loss.item()
        
                if (stp % 35 == 0):
                    self.history.eval()

                with torch.no_grad():
                    valid_loss, accuracy = self.valid_fun(valid_loader)
            
                print("Epoch: {}/{} | ".format(e+1, epochs),
                    "Training Loss: {:.4f} | ".format(continuous_loss/35),
                    "Validation Loss: {:.4f} | ".format(valid_loss/len(test_loader)),
                     "Validation Accuracy: {:.4f}".format(accuracy/len(test_loader)))

                continuous_loss = 0
                self.history.train()

        
        


        