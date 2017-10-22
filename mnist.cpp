#include <vector>
#include <iostream>
#include <fstream>

class MNistDataSet {
  uint32_t _numImages;
  uint32_t _numRows;
  uint32_t _numColumns;
  std::vector<std::vector<uint8_t> > _images;
  std::vector<uint8_t> _labels;

  static bool isLittleEndian() {
    volatile uint32_t i=0x01234567;
    return (*((uint8_t*)(&i))) == 0x67 ? true:false;
  }
  
  static void swapIfLittleEndian(char *buf, size_t width) {
    if(!isLittleEndian()) return;
    for(int pos = 0; 2 * pos < width; pos++) {
      int invPos = width - 1 - pos;
      char tmp = buf[pos];
      buf[pos] = buf[invPos];
      buf[invPos] = tmp;
    }
  }

  static unsigned int readUInt32(std::ifstream &ifs) {
    char buf[4];
    ifs.read(buf, 4);
    swapIfLittleEndian(buf, 4);
    uint32_t out = *reinterpret_cast<uint32_t *>(buf);
    return out;
  }
  
  std::vector<uint8_t> readArray(std::ifstream &ifs, int _size) {
    std::vector<uint8_t> buf(_size);
    ifs.read((char *)&buf[0], _size);
    return buf;
  }

public:
  MNistDataSet(const std::string imageFile, const std::string labelFile) {
    std::ifstream ifsLabel(labelFile, std::ios::in|std::ios::binary);
    uint32_t magicNumber;
    magicNumber = readUInt32(ifsLabel);
    _numImages = readUInt32(ifsLabel);
    _labels = readArray(ifsLabel, _numImages);
    
    std::ifstream ifsImage(imageFile, std::ios::in|std::ios::binary);
    magicNumber = readUInt32(ifsImage);
    _numImages = readUInt32(ifsImage);
    _numRows = readUInt32(ifsImage);
    _numColumns = readUInt32(ifsImage);
    for(int i = 0; i < _numImages; i++) {
      _images.push_back(readArray(ifsImage, _numRows*_numColumns));
    }
  }

  uint32_t getNumImages() {
    return _numImages;
  }

  uint32_t getNumRows() {
    return _numRows;
  }

  uint32_t getNumColumns() {
    return _numColumns;
  }

  uint8_t getLabel(int i) {
    return _labels[i];
  }

  std::vector<uint8_t> getImage(int i) {
    return _images[i];
  }

  std::vector<double> getImageDouble(int i) {
    std::vector<double> imageDouble(_numRows*_numColumns);
    for(int p = 0; p < _numRows*_numColumns; p++) {
      imageDouble[p] = ((double)_images[i][p]) / ((double)256);
    }
    return imageDouble;
  }

  std::vector<double> getLabelDouble(int i) {
    std::vector<double> labelDouble(10, 0.0);
    labelDouble[_labels[i]] = 1.0;
    return labelDouble;
  }
};

