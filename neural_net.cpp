#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <complex>

template <class S>
class RandomGenerator {
public:
  RandomGenerator(S amplitude);
  S rand();
};

template <>
class RandomGenerator<double> {
  std::random_device rnd;
  std::mt19937 mt;
  std::uniform_real_distribution<> dist;
public:
  RandomGenerator(double lb, double ub) :
    rnd(),
    mt(this->rnd()),
    dist(lb, ub)
  {
  }

  double rand() {
    return dist(mt);
  }
};

template <class S>
class Activation {
public:
  virtual std::vector<S> activation(std::vector<S> input) = 0;
  virtual std::vector<S> gradient(std::vector<S> input) = 0;
};

template <class S>
class ReLuActivation : public Activation<S> {
public:
  std::vector<S> activation(std::vector<S> input) {
    std::for_each(input.begin(), input.end(), [](S &x) {x = x > 0 ? x : 0;});
    return input;
  }

  std::vector<S> gradient(std::vector<S> input) {
    std::for_each(input.begin(), input.end(), [](S &x) {x = x > 0 ? 1 : 0;});
    return input;
  }
};

template <class S>
class SigmoidActivation : public Activation<S> {
public:
  static S sigmoid(S x) {
    return static_cast<S>(1.0) / (static_cast<S>(1.0) + std::exp(x));
  }

  std::vector<S> activation(std::vector<S> input) {
    std::for_each(input.begin(), input.end(), [](S &x) {x = sigmoid(x);});
    return input;
  }

  std::vector<S> gradient(std::vector<S> input) {
    std::for_each(input.begin(), input.end(), [](S &x) {x = sigmoid(x) / (static_cast<S>(1.0) - sigmoid(x));});
    return input;
  }
};

template <class S>
class SwishActivation : public Activation<S> {
public:
  static S sigmoid(S x) {
    return static_cast<S>(1.0) / (static_cast<S>(1.0) + std::exp(x));
  }

  static S swish (S x) {
    return x * sigmoid(x);
  }
  static S swishGradient(S x) {
    return swish(x) + sigmoid(x) * (1 - swish(x));
  }

  std::vector<S> activation(std::vector<S> input) {
    std::for_each(input.begin(), input.end(), [](S &x) {x = swish(x);});
    return input;
  }

  std::vector<S> gradient(std::vector<S> input) {
    std::for_each(input.begin(), input.end(), [](S &x) {x = swishGradient(x);});
    return input;
  }
};

template <class S>
class SoftmaxActivation : public Activation <S>{
public:
  std::vector<S> activation(std::vector<S> input) {
    std::for_each(input.begin(), input.end(), [](S &x) {x = std::exp(x);});
    S sum = std::accumulate(input.begin(), input.end(), static_cast<S>(0));
    std::for_each(input.begin(), input.end(), [sum](S &x) {x /= sum;});
    return input;
  }

  std::vector<S> gradient(std::vector<S> input) {
    return input; // dummy
  }
};

template <class S>
struct Layer {
  size_t _inSize, _outSize;
  size_t _sampleCount;
  std::vector<std::vector<S> > _w, _w_grad;
  std::vector<S> _input; /* size: inSize */
  std::vector<S> _u; /* size: outSize */
  std::vector<S> _output; /* size: outSize */
  std::shared_ptr<Activation<S>> _activation;
public:
  enum class ActivationType {
    RELU,
    SIGMOID,
    SWISH,
    SOFTMAX
  };

  Layer(size_t inSize, size_t outSize, ActivationType activationType) :
    _inSize(inSize + 1),
    _outSize(outSize),
    _input(_inSize, 0),
    _u(_outSize, 0),
    _sampleCount(0),
    _output(_outSize, 0)
  {
    RandomGenerator<S> rg(0.0, 1.0);
    for(int i = 0; i < this->_inSize; i++) {
      std::vector<S> row(_outSize);
      for(int j = 0; j < this->_outSize; j++) {
	row[j] = rg.rand() / this->_inSize;
      }
      _w.push_back(row);
      _w_grad.push_back(std::vector<S>(_outSize, 0));
    }

    switch(activationType) {
    case ActivationType::RELU:
      _activation = std::make_shared<ReLuActivation<S> >();
      break;
    case ActivationType::SIGMOID:
      _activation = std::make_shared<SigmoidActivation<S> >();
      break;
    case ActivationType::SWISH:
      _activation = std::make_shared<SwishActivation<S> >();
      break;
    case ActivationType::SOFTMAX:
      _activation = std::make_shared<SoftmaxActivation<S> >();
      break;
    }
  }

  std::vector<S> forward(const std::vector<S> &input) {
    _input = input;
    _input.push_back(static_cast<S>(1));
    std::fill(_u.begin(), _u.end(), 0);
    for(int i = 0; i < this->_inSize; i++) {
      for(int j = 0; j < this->_outSize; j++) {
	_u[j] += _input[i] * _w[i][j];
      }
    }
    _output = _activation->activation(_u);
    return _output;
  }

  std::vector<S> calcDelta(const std::vector<S> &nextDelta, const std::vector<std::vector<S> > &nextW) {
    std::vector<S> delta(this->_outSize, 0);
    std::vector<S> grad = _activation->gradient(_u);
    for(int j = 0; j < this->_outSize; j++) {
      for(int k = 0; k < nextW[j].size(); k++) {
	delta[j] += nextDelta[k] * nextW[j][k] * grad[j];
      }
    }
    return delta;
  }

  void updateGrad(const std::vector<S> &delta) {
    for(int i = 0; i < this->_inSize; i++) {
      for(int j = 0; j < this->_outSize; j++) {
	_w_grad[i][j] += _input[i] * delta[j];
      }
    }
    _sampleCount++;
  }

  void updateParam(S learningRate) {
    if(_sampleCount == 0) return;
    for(int i = 0; i < this->_inSize; i++) {
      for(int j = 0; j < this->_outSize; j++) {
	_w[i][j] -= _w_grad[i][j] * learningRate / _sampleCount;
	_w_grad[i][j] = 0;
      }
    }
    _sampleCount = 0;
  }
};

template<class S>
class Network {
  bool _verbose;
  std::vector<Layer<S>> _layers;
public:
  Network(bool verbose = false) :
    _verbose(verbose)
  {
  }

  void addLayer(Layer<S> layer) {
    _layers.push_back(layer);
  }

  std::vector<S> forward(const std::vector<S> &input) {
    std::vector<S> buffer = input;
    for(auto &layer : _layers) {
      std::vector<S> output = layer.forward(buffer);
      buffer = output;
    }
    return buffer;
  }

  void backward(const std::vector<S> &target) {
    Layer<S> &lastLayer = _layers[_layers.size() - 1];
    const std::vector<S> &y = lastLayer._output;
    std::vector<S> delta(target.size());
    for(int i = 0; i < y.size(); i++) {
      delta[i] = y[i] - target[i];
    }
    if(_verbose) {
      std::cout << "delta of layer " << _layers.size() - 1 << ": ";
      std::for_each(delta.begin(), delta.end(), [](const auto &x) {std::cout << " " << x;});
      std::cout << std::endl;
    }
    lastLayer.updateGrad(delta);
    for(int l = _layers.size() - 2; l >= 0; l--) {
      delta = _layers[l].calcDelta(delta, _layers[l+1]._w);
      _layers[l].updateGrad(delta);
      if(_verbose) {
	std::cout << "delta of layer " << l << ": ";
	std::for_each(delta.begin(), delta.end(), [](const auto &x) {std::cout << " " << x;});
	std::cout << std::endl;
      }
    }
  }
  
  S calcLoss(const std::vector<S> &target) {
    S loss = 0;
    for(int i = 0; i < target.size(); i++) {
      loss -= target[i] * std::log(_layers[_layers.size() - 1]._output[i]);
    }
    return loss;
  }

  void updateParam(S learningRate = static_cast<S>(0.1)) {
    for(auto & layer : _layers) {
      layer.updateParam(learningRate);
    }
  }
};

