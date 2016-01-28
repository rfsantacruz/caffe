#ifndef CAFFE_TRIPLET_CLS_LOSS_LAYER_HPP_
#define CAFFE_TRIPLET_CLS_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {


template <typename Dtype>
class TripletClsLossLayer : public LossLayer<Dtype> {
 public:
  explicit TripletClsLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 3; }
   virtual inline const char* type() const { return "TripletClsLoss"; }
  /**
   * Unlike most loss layers, in the TripletLossLayer we can backpropagate
   * to the first three inputs.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 3;
  }

 protected:  
   virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> const_;  // cached for backward pass
  Blob<Dtype> diff_sample_target; // cached for backward pass
  Blob<Dtype> diff_sample_impostor; // cached for backward pass
  Blob<Dtype> diff_impostor_target;  // cached for backward pass
};
}  // namespace caffe

#endif  // CAFFE_TRIPLET_CLS_LOSS_LAYER_HPP_
