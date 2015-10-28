#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

//bottom 0 -> image
//bottom 1 -> target
//bottom 2 -> impostor

template <typename Dtype>
void TripletClsLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  //check channel size must be the same for all branches
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[1]->channels(), bottom[2]->channels());
  
  //check width and heights of each branch
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);

  //caches for backward pass
  diff_impostor_target.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sample_target.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sample_impostor.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);  
  const_.Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void TripletClsLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  //read image, target and impostor
  const Dtype* sample_data = bottom[0]->cpu_data();
  const Dtype* target_sample_data = bottom[1]->cpu_data();
  const Dtype* impostor_sample_data = bottom[2]->cpu_data();
    
  //compute sizes  
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int channels = bottom[0]->channels();

  //computing differences
  caffe_sub(count, impostor_sample_data, target_sample_data, diff_impostor_target.mutable_cpu_data());
  caffe_sub(count, target_sample_data, sample_data, diff_sample_target.mutable_cpu_data());
  caffe_sub(count, impostor_sample_data, sample_data, diff_sample_impostor.mutable_cpu_data());

  Dtype loss_hinge = 0;
  Dtype sample_target_distance;
  Dtype sample_impostor_distance;

  for (int i = 0; i < num; ++i) {

      sample_target_distance = caffe_cpu_dot(
          channels, diff_sample_target.cpu_data() + bottom[0]->offset(i),
          diff_sample_target.cpu_data() + bottom[0]->offset(i));

      sample_impostor_distance = caffe_cpu_dot(
          channels, diff_sample_impostor.cpu_data() + bottom[0]->offset(i),
          diff_sample_impostor.cpu_data() + bottom[0]->offset(i));

      Dtype violation = std::max(Dtype(0.0), sample_target_distance -
                  sample_impostor_distance + 1);

      loss_hinge += violation; const_.mutable_cpu_data()[i] = violation;	      
      
  }

  Dtype loss = 0.0;  
  loss = loss_hinge/static_cast<Dtype>(2.0*num);   
  top[0]->mutable_cpu_data()[0] = loss;
  
}

template <typename Dtype>
void TripletClsLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
         
  int num = bottom[0]->num();  
  int channels = bottom[0]->channels();

  for (int n = 0; n < num; ++n) {
    if (propagate_down[0]) {                                           
	    if(const_.cpu_data()[n] > Dtype(0.0)){
		
               //derivative in relation to image                
                 caffe_cpu_axpby(channels, Dtype( 2./num), diff_impostor_target.cpu_data() + bottom[0]->offset(n), Dtype(0.0),
                                bottom[0]->mutable_cpu_diff() + bottom[0]->offset(n));

		//derivative in relation to target                
                caffe_cpu_axpby(channels, Dtype( 2./num), diff_sample_target.cpu_data() + bottom[1]->offset(n), Dtype(0.0),
                                bottom[1]->mutable_cpu_diff() + bottom[1]->offset(n));                 

		//derivative in relation to impostor
                caffe_cpu_axpby(channels, Dtype( -2./num), diff_sample_impostor.cpu_data() + bottom[2]->offset(n), Dtype(0.0),
                                bottom[2]->mutable_cpu_diff() + bottom[2]->offset(n));     

		           
            }else{

		caffe_set(channels, Dtype(0.0), bottom[0]->mutable_cpu_diff() + bottom[0]->offset(n));
                caffe_set(channels, Dtype(0.0), bottom[1]->mutable_cpu_diff() + bottom[1]->offset(n));
                caffe_set(channels, Dtype(0.0), bottom[2]->mutable_cpu_diff() + bottom[2]->offset(n));

	    }
    }           
 }
}


#ifdef CPU_ONLY
STUB_GPU(TripletClsLossLayer);
#endif

INSTANTIATE_CLASS(TripletClsLossLayer);
REGISTER_LAYER_CLASS(TripletClsLoss);

}  // namespace caffe
