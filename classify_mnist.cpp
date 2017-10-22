#include <utility>
#include <iostream>
#include <iomanip>
#include <limits>

#include "neural_net.cpp"
#include "mnist.cpp"

std::pair<double, double> runEpoch(Network<double> &net, MNistDataSet &set, bool train, double learningRate = 0.1, int batchSize = 100) {
  int numCorrect = 0;
  int numWrong = 0;
  double sumLoss = 0;
  double batchLoss = 0;
  int batchId = 0;
  for(uint32_t sample = 0; sample < set.getNumImages(); sample++) {
    std::vector<double> in = set.getImageDouble(sample);
    std::vector<double> out = net.forward(in);
    uint8_t estimatedLabel = std::distance(out.begin(), std::max_element(out.begin(), out.end()));
    bool isCorrect = estimatedLabel == set.getLabel(sample) ? true : false;
    if(isCorrect) {
      numCorrect++;
    }else {
      numWrong++;
    }
    std::vector<double> labelOneHot = set.getLabelDouble(sample);
    double sampleLoss = net.calcLoss(labelOneHot);
    sumLoss += sampleLoss;
    batchLoss += sampleLoss;
    if(train) {
      net.backward(labelOneHot);
      if(sample % batchSize == 0 || sample == set.getNumImages() - 1) {
	std::cout << std::fixed << std::setprecision(4) << "\rbatch loss[" << batchId << "]: " << batchLoss / batchSize;
	std::cout.flush();
	batchLoss = 0;
	batchId++;
	net.updateParam(learningRate);
      }
    }
  }
  double meanLoss = sumLoss / (numCorrect + numWrong);
  double errorRate = (double)numWrong / (numCorrect + numWrong);
  return std::make_pair(meanLoss, errorRate);
}

int main() {

  MNistDataSet trainSet("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte");
  MNistDataSet testSet("mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte");
  
  Network<double> net;
  net.addLayer(Layer<double>(trainSet.getNumRows() * trainSet.getNumColumns(), 300, Layer<double>::ActivationType::RELU));
  net.addLayer(Layer<double>(300, 10, Layer<double>::ActivationType::SOFTMAX));

  double prevLoss = std::numeric_limits<double>::max();
  double learningRate = 0.2;
  for(int epoch = 0; epoch < 50; epoch++) {
    std::cout << "running epoch " << epoch << std::endl;
    auto trainResult = runEpoch(net, trainSet, true, learningRate);
    std::cout << "epoch finished" << std::endl;
    std::cout << "train set mean loss: " << trainResult.first << std::endl;
    std::cout << "train set error rate: " << trainResult.second << std::endl;

    auto testResult = runEpoch(net, testSet, false);
    std::cout << "test set mean loss: " << testResult.first << std::endl;
    std::cout << "test set error rate: " << testResult.second << std::endl;

    if(prevLoss < trainResult.first) {
      learningRate *= 0.5;
      std::cout << "mean loss " << trainResult.first << " is worse than prev loss " << prevLoss << ": decaying learning rate to " << learningRate << std::endl;
    }
    prevLoss = trainResult.first;


  }
}
