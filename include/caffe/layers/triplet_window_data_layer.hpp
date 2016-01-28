#ifndef CAFFE_TRIPLET_WINDOW_DATA_LAYER_HPP_
#define CAFFE_TRIPLET_WINDOW_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>
#include <map>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class TripletWindowDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit TripletWindowDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~TripletWindowDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TripletWindowData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual unsigned int PrefetchRand();
  virtual void load_batch(Batch<Dtype>* batch);
  bool prepare_window(cv::Mat &warpimg, int &pad_w, int &pad_h, bool &do_mirror, const vector<float> &window, const bool mirror, const int context_pad, const bool use_square, const int crop_size);

  shared_ptr<Caffe::RNG> prefetch_rng_;
  vector<std::pair<std::string, vector<int> > > image_database_;
  enum WindowField { IMAGE_INDEX, BB_INDEX, LABEL, OVERLAP, X1, Y1, X2, Y2, NUM_WIN_FIELDS };
  enum PairField { PAIR_INDEX, IMAGE_INDEX_1, BB_INDEX_1, IMAGE_INDEX_2, BB_INDEX_2, IMAGE_INDEX_3, BB_INDEX_3, NUM_PAIR_FIELDS };
  map<int, map<int, vector<float> > > windows_; //map imidx => (bbidx => window)
  map<int, vector<vector<int> > > triplets_; //map class_label => triplets
  map<int, int > cls_idx_; //clas_label => sample idx
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
  bool has_mean_file_;
  bool has_mean_values_;
  bool cache_images_;
  vector<std::pair<std::string, Datum > > image_database_cache_;
  vector<int> cls_keys_;
};

}

#endif  // CAFFE_TRIPLET_WINDOW_DATA_LAYER_HPP_

