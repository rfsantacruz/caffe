#ifdef USE_OPENCV
#include <opencv2/highgui/highgui_c.h>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

#include <stdint.h>

#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "caffe/common.hpp"
#include "caffe/layers/triplet_window_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// caffe.proto > LayerParameter > TripletWindowDataParameter
//   'source' field specifies the window_file
//   'crop_size' indicates the desired warped size

namespace caffe {

template <typename Dtype>
TripletWindowDataLayer<Dtype>::~TripletWindowDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void TripletWindowDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // LayerSetUp runs through the window_file and creates two structures
  // that hold windows: one for foreground (object) windows and one
  // for background (non-object) windows. We use an overlap threshold
  // to decide which is which.

  // window_file format
  // repeated:
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    class_index overlap x1 y1 x2 y2

  LOG(INFO) << "Triplet Window data layer:" << std::endl
      << "  foreground (object) overlap threshold: "
      << this->layer_param_.triplet_window_data_param().fg_threshold() << std::endl
      << "  background (non-object) overlap threshold: "
      << this->layer_param_.triplet_window_data_param().bg_threshold() << std::endl      
      << "  cache_images: "
      << this->layer_param_.triplet_window_data_param().cache_images() << std::endl
      << "  root_folder: "
      << this->layer_param_.triplet_window_data_param().root_folder();

  cache_images_ = this->layer_param_.triplet_window_data_param().cache_images();
  string root_folder = this->layer_param_.triplet_window_data_param().root_folder();

  const bool prefetch_needs_rand =
      this->transform_param_.mirror() ||
      this->transform_param_.crop_size();
  if (prefetch_needs_rand) {
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
  } else {
    prefetch_rng_.reset();
  }

  std::ifstream infile(this->layer_param_.triplet_window_data_param().source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.triplet_window_data_param().source() << std::endl;

  map<int, int> label_hist, sample_hist, tg_hist, imp_hist;
  label_hist.insert(std::make_pair(0, 0)); sample_hist.insert(std::make_pair(0, 0)); tg_hist.insert(std::make_pair(0, 0)); imp_hist.insert(std::make_pair(0, 0));

  //pairs statistics
  int count_same_img = 0;
  boost::accumulators::accumulator_set<float, boost::accumulators::features<boost::accumulators::tag::count, boost::accumulators::tag::mean, boost::accumulators::tag::variance > > pair_stat_acc;

  string hashtag, type;
  int index=0, image_index=0, channels=0, num_pairs=0;
  if (!(infile >> hashtag >> type >> index)) {
    LOG(FATAL) << "Window file is empty";
  }
  do {
    CHECK_EQ(hashtag, "#");

    if(boost::iequals(type, "FULL_IMAGES")){
            image_index = index;
	    // read image path
	    string image_path;
	    infile >> image_path;
	    image_path = root_folder + image_path;
	    // read image dimensions
	    vector<int> image_size(3);
	    infile >> image_size[0] >> image_size[1] >> image_size[2];
	    channels = image_size[0];
	    image_database_.push_back(std::make_pair(image_path, image_size));

	    if (cache_images_) {
	      Datum datum;
	      if (!ReadFileToDatum(image_path, &datum)) {
		LOG(ERROR) << "Could not open or find file " << image_path;
		return;
	      }
	      image_database_cache_.push_back(std::make_pair(image_path, datum));
	    }
	    // read each box
	    int num_windows;
	    infile >> num_windows;
	    const float fg_threshold =
		this->layer_param_.triplet_window_data_param().fg_threshold();
	    const float bg_threshold =
		this->layer_param_.triplet_window_data_param().bg_threshold();
	    for (int i = 0; i < num_windows; ++i) {
	      int bb_index, label, x1, y1, x2, y2;
	      float overlap;
	      infile >> bb_index >> label >> overlap >> x1 >> y1 >> x2 >> y2;

	      vector<float> window(TripletWindowDataLayer::NUM_WIN_FIELDS);
	      window[TripletWindowDataLayer::IMAGE_INDEX] = image_index;
	      window[TripletWindowDataLayer::BB_INDEX] = bb_index;
	      window[TripletWindowDataLayer::LABEL] = label;
	      window[TripletWindowDataLayer::OVERLAP] = overlap;
	      window[TripletWindowDataLayer::X1] = x1;
	      window[TripletWindowDataLayer::Y1] = y1;
	      window[TripletWindowDataLayer::X2] = x2;
	      window[TripletWindowDataLayer::Y2] = y2;

	      // add window to foreground list or background list
	      if (overlap >= fg_threshold) {
		int label = window[TripletWindowDataLayer::LABEL];
		CHECK_GT(label, 0);				
		label_hist.insert(std::make_pair(label, 0));
		label_hist[label]++;
		
		map<int,vector<float> > inner;
                pair<map<int, map<int, vector<float> > >::iterator,bool> ret;
		ret = windows_.insert(std::make_pair(image_index, inner));
		ret.first->second.insert(std::make_pair(bb_index, window));
		

	      } else if (overlap < bg_threshold) {
		// background window, force label and overlap to 0
		window[TripletWindowDataLayer::LABEL] = 0;
		window[TripletWindowDataLayer::OVERLAP] = 0;		
		label_hist[0]++;

		map<int,vector<float> > inner;
                pair<map<int, map<int, vector<float> > >::iterator,bool> ret;
		ret = windows_.insert(std::make_pair(image_index, inner));
		ret.first->second.insert(std::make_pair(bb_index, window));

	      }
	    }

	    if (image_index % 100 == 0) {
	      LOG(INFO) << "num: " << image_index << " "
		  << image_path << " "
		  << image_size[0] << " "
		  << image_size[1] << " "
		  << image_size[2] << " "
		  << "windows to process: " << num_windows;
	    }
    }else if(boost::iequals(type, "PAIRS")){
    	
         num_pairs = num_pairs + index;
	 for (int i = 0; i < index; ++i) {
	    int pair_index, image_index_1, bb_index_1, image_index_2, bb_index_2, image_index_3, bb_index_3;
            infile >> pair_index >> image_index_1 >> bb_index_1 >> image_index_2 >> bb_index_2 >> image_index_3 >> bb_index_3;
         
            vector<int> pair(TripletWindowDataLayer::NUM_PAIR_FIELDS);
	    pair[TripletWindowDataLayer::PAIR_INDEX] = pair_index;
	    pair[TripletWindowDataLayer::IMAGE_INDEX_1] = image_index_1;
            pair[TripletWindowDataLayer::BB_INDEX_1] = bb_index_1;
            pair[TripletWindowDataLayer::IMAGE_INDEX_2] = image_index_2;
            pair[TripletWindowDataLayer::BB_INDEX_2] = bb_index_2;
	    pair[TripletWindowDataLayer::IMAGE_INDEX_3] = image_index_3;
            pair[TripletWindowDataLayer::BB_INDEX_3] = bb_index_3;	    	    
            int cls = windows_[image_index_1][bb_index_1][TripletWindowDataLayer::LABEL];

            //cache triplets
	    vector<vector<int> > inner;
            std::pair<std::map<int, vector<vector<int> > >::iterator, bool> ret;
	    ret = triplets_.insert(std::make_pair(cls, inner));
	    ret.first->second.push_back(pair);

	    //compute distribution of classes in the triplets
            int sample_label, tg_label, imp_label;
            sample_label = windows_[image_index_1][bb_index_1][TripletWindowDataLayer::LABEL];
	    sample_hist.insert(std::make_pair(sample_label, 0));
            sample_hist[sample_label]++;

            tg_label = windows_[image_index_2][bb_index_2][TripletWindowDataLayer::LABEL];
	    tg_hist.insert(std::make_pair(tg_label, 0));
            tg_hist[tg_label]++;

            imp_label = windows_[image_index_3][bb_index_3][TripletWindowDataLayer::LABEL];
            imp_hist.insert(std::make_pair(imp_label, 0));
            imp_hist[imp_label]++;
	    
	    //compute overlap between target and sample
	    if(image_index_1 == image_index_2){
               count_same_img++;

               vector<float> window_a = windows_[image_index_1][bb_index_1];
               vector<float> window_b = windows_[image_index_2][bb_index_2];
               float XA1 = window_a[TripletWindowDataLayer<Dtype>::X1]; float YA1 = window_a[TripletWindowDataLayer<Dtype>::Y1]; 
               float XA2 = window_a[TripletWindowDataLayer<Dtype>::X2]; float YA2 = window_a[TripletWindowDataLayer<Dtype>::Y2];

               float XB1 = window_b[TripletWindowDataLayer<Dtype>::X1]; float YB1 = window_b[TripletWindowDataLayer<Dtype>::Y1]; 
               float XB2 = window_b[TripletWindowDataLayer<Dtype>::X2]; float YB2 = window_b[TripletWindowDataLayer<Dtype>::Y2];

               float sa = (XA2 - XA1) * (YA2 - YA1);
               float sb = (XB2 - XB1) * (YB2 - YB1); 
               float si = std::max(0.0f, std::min(XA2,XB2) - std::max(XA1,XB1)) * std::max(0.0f, std::min(YA2,YB2) - std::max(YA1,YB1));
               if(si > 0.0f){
                   pair_stat_acc(si/(sa+sb-si));
               }
            }
	 }         	
    }
    
  } while (infile >> hashtag >> type >> index);

  //shuffle triplets and create vector of keys
  LOG(INFO) << "Shuffling triplets";
  for(std::map<int, vector<vector<int> > >::iterator it = triplets_.begin(); it != triplets_.end(); it++) {
     std::random_shuffle(it->second.begin(), it->second.end());
     cls_keys_.push_back(it->first);
     cls_idx_.insert(std::make_pair(it->first, 0)); 
  }  


  LOG(INFO) << "Number of images: " << image_index+1;
  LOG(INFO) << "Number of classes with instances: " << cls_keys_.size();
  for (map<int, int>::iterator it = label_hist.begin();
      it != label_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << label_hist[it->first]
              << " images";
  }
  
  LOG(INFO) << "Number of triplets: " << num_pairs;

  LOG(INFO) << "Distribution of sample class in triplets: ";
  for (map<int, int>::iterator it = sample_hist.begin();
      it != sample_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " is the sample of " << sample_hist[it->first]
              << " ("<< (sample_hist[it->first] * 100.0)/num_pairs << ") triplets";
  }

  LOG(INFO) << "Distribution of targets class in triplets: ";
  for (map<int, int>::iterator it = tg_hist.begin();
      it != tg_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " is the target of " << tg_hist[it->first]
              << " ("<< (tg_hist[it->first] * 100.0)/num_pairs << ") triplets";
  }

  LOG(INFO) << "Distribution of impostors class in triplets: ";
  for (map<int, int>::iterator it = imp_hist.begin();
      it != imp_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " is the impostor of " << imp_hist[it->first]
              << "("<< (imp_hist[it->first] * 100.0)/num_pairs << ") triplets";
  }

  LOG(INFO) << "Percentage of triplets with target from the same image: "
      << count_same_img/((float)num_pairs);

  LOG(INFO) << "Percentage of triplets with overlaped target: "
      << boost::accumulators::count(pair_stat_acc)/((float)num_pairs);

  LOG(INFO) << "Average overlap ratio in triplets: "
      << boost::accumulators::mean(pair_stat_acc);
  
  LOG(INFO) << "STD overlap ratio in triplets: "
      << std::sqrt(boost::accumulators::variance(pair_stat_acc));

  LOG(INFO) << "Amount of context padding: "
      << this->layer_param_.triplet_window_data_param().context_pad();

  LOG(INFO) << "Crop mode: "
      << this->layer_param_.triplet_window_data_param().crop_mode();


  //configure top blobs (sample image, target image, impostor image)
  // image
  const int crop_size = this->transform_param_.crop_size();
  CHECK_GT(crop_size, 0);
  const int batch_size = this->layer_param_.triplet_window_data_param().batch_size();
  top[0]->Reshape(batch_size, 3*channels, crop_size, crop_size);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i)
    this->prefetch_[i].data_.Reshape(
        batch_size, 3*channels, crop_size, crop_size);

  LOG(INFO) << "output data size: images grouped as " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  
 // label
  vector<int> label_shape(3, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
    
  // data mean
  has_mean_file_ = this->transform_param_.has_mean_file();
  has_mean_values_ = this->transform_param_.mean_value_size() > 0;
  if (has_mean_file_) {
    const string& mean_file =
          this->transform_param_.mean_file();
    LOG(INFO) << "Loading mean file from: " << mean_file;
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }

  if (has_mean_values_) {
    CHECK(has_mean_file_ == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < this->transform_param_.mean_value_size(); ++c) {
      mean_values_.push_back(this->transform_param_.mean_value(c));
    }
    CHECK(mean_values_.size() == 1 || mean_values_.size() == channels) <<
     "Specify either 1 mean_value or as many as channels: " << channels;
    if (channels > 1 && mean_values_.size() == 1) {
      // Replicate the mean_value for simplicity
      for (int c = 1; c < channels; ++c) {
        mean_values_.push_back(mean_values_[0]);
      }
    }
  }
}

template <typename Dtype>
unsigned int TripletWindowDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

// This function is called on prefetch thread
template <typename Dtype>
void TripletWindowDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  // At each iteration, uniformally sample N triplets
  CPUTimer batch_timer;
  batch_timer.Start();
  double trans_time = 0;
  CPUTimer timer;
  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = batch->label_.mutable_cpu_data();
  const Dtype scale = this->layer_param_.triplet_window_data_param().scale();
  const int batch_size = this->layer_param_.triplet_window_data_param().batch_size();
  const int context_pad = this->layer_param_.triplet_window_data_param().context_pad();
  const int crop_size = this->transform_param_.crop_size();
  const bool mirror = this->transform_param_.mirror();

  Dtype* mean = NULL;
  int mean_off = 0;
  int mean_width = 0;
  int mean_height = 0;
  if (this->has_mean_file_) {
    mean = this->data_mean_.mutable_cpu_data();
    mean_off = (this->data_mean_.width() - crop_size) / 2;
    mean_width = this->data_mean_.width();
    mean_height = this->data_mean_.height();
  }
  //cv::Size cv_crop_size(crop_size, crop_size);
  const string& crop_mode = this->layer_param_.triplet_window_data_param().crop_mode();

  bool use_square = (crop_mode == "square") ? true : false;

  // zero out batch
  caffe_set(batch->data_.count(), Dtype(0), top_data);


  //configure to sample triplets uniformly
  map<int, int> cls_numSamp; // cls -> qtd
  for(int cls = 0; cls < cls_keys_.size(); cls++) {
     cls_numSamp.insert(std::make_pair(cls_keys_[cls], batch_size/cls_keys_.size()));
  }
  const int rest = batch_size % triplets_.size();
  for(int r=0; r < rest; r++) {
     const unsigned int rand_index = PrefetchRand();
     int sort_key = cls_keys_[rand_index % cls_keys_.size()];
     cls_numSamp[sort_key]++;
  }

  //check if the sampling is correct
  int total_samp = 0;
  for(int cls = 0; cls < cls_keys_.size(); cls++) {
     total_samp += cls_numSamp[cls_keys_[cls]];
  }  
  CHECK_EQ(batch_size, total_samp);

  int item_id = 0;
  //sample in each class
  for (int cls = 0; cls < cls_keys_.size(); cls++) {
    int cls_key = cls_keys_[cls];
    for (int dummy = 0; dummy < cls_numSamp[cls_key]; ++dummy) {
      // sample a triplet
      timer.Start();
      //const unsigned int rand_index = PrefetchRand();
      vector<int> pair = triplets_[cls_key][cls_idx_[cls_key] % triplets_[cls_key].size()]; cls_idx_[cls_key]++;

      //a - sample, b - target, c - impostor
      vector<float> window_a = windows_[pair[TripletWindowDataLayer<Dtype>::IMAGE_INDEX_1]][pair[TripletWindowDataLayer<Dtype>::BB_INDEX_1]];
      vector<float> window_b = windows_[pair[TripletWindowDataLayer<Dtype>::IMAGE_INDEX_2]][pair[TripletWindowDataLayer<Dtype>::BB_INDEX_2]];
      vector<float> window_c = windows_[pair[TripletWindowDataLayer<Dtype>::IMAGE_INDEX_3]][pair[TripletWindowDataLayer<Dtype>::BB_INDEX_3]];

      //check target and impostor roles-- uncomented for unsuperv exps
      CHECK_EQ(window_a[TripletWindowDataLayer::LABEL] , window_b[TripletWindowDataLayer::LABEL]);        
      CHECK_NE(window_a[TripletWindowDataLayer::LABEL], window_c[TripletWindowDataLayer::LABEL]);
      

      //prepare windows
      int pad_wa=0, pad_ha=0, pad_wb=0, pad_hb=0, pad_wc=0, pad_hc=0;
      bool do_mirrora = false, do_mirrorb = false, do_mirrorc = false;
      cv::Mat cv_cropped_img_a, cv_cropped_img_b, cv_cropped_img_c; 
      bool isImage_aPrep = this->prepare_window(cv_cropped_img_a, pad_wa, pad_ha, do_mirrora, window_a, mirror, context_pad, use_square, crop_size);
      bool isImage_bPrep = this->prepare_window(cv_cropped_img_b, pad_wb, pad_hb, do_mirrorb, window_b, mirror, context_pad, use_square, crop_size);
      bool isImage_cPrep = this->prepare_window(cv_cropped_img_c, pad_wc, pad_hc, do_mirrorc, window_c, mirror, context_pad, use_square, crop_size);
      if(!isImage_aPrep || !isImage_bPrep || !isImage_cPrep)
	return;

      CHECK_EQ(cv_cropped_img_a.channels(), cv_cropped_img_b.channels());
      CHECK_EQ(cv_cropped_img_a.channels(), cv_cropped_img_c.channels());
      const int channels = cv_cropped_img_a.channels();

      // copy the warped window a into top_data
      for (int h = 0; h < cv_cropped_img_a.rows; ++h) {
        const uchar* ptr = cv_cropped_img_a.ptr<uchar>(h);
        int img_index = 0;
        for (int w = 0; w < cv_cropped_img_a.cols; ++w) {
          for (int c = 0; c < channels; ++c) {
            int top_index = ((item_id * (3*channels) + c) * crop_size + h + pad_ha)
                     * crop_size + w + pad_wa;
            // int top_index = (c * height + h) * width + w;
            Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
            if (this->has_mean_file_) {
              int mean_index = (c * mean_height + h + mean_off + pad_ha)
                           * mean_width + w + mean_off + pad_wa;
              top_data[top_index] = (pixel - mean[mean_index]) * scale;
            } else {
              if (this->has_mean_values_) {
                top_data[top_index] = (pixel - this->mean_values_[c]) * scale;
              } else {
                top_data[top_index] = pixel * scale;
              }
            }
          }
        }
      }

      // copy the warped window b into top_data
      for (int h = 0; h < cv_cropped_img_b.rows; ++h) {
        const uchar* ptr = cv_cropped_img_b.ptr<uchar>(h);
        int img_index = 0;
        for (int w = 0; w < cv_cropped_img_b.cols; ++w) {
          for (int c = channels; c < 2*channels; ++c) {
            int top_index = ((item_id * (3*channels) + c) * crop_size + h + pad_hb)
                     * crop_size + w + pad_wb;
            // int top_index = (c * height + h) * width + w;
            Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
            int mean_c = c % channels;
            if (this->has_mean_file_) {
              int mean_index = (mean_c * mean_height + h + mean_off + pad_hb)
                           * mean_width + w + mean_off + pad_wb;
              top_data[top_index] = (pixel - mean[mean_index]) * scale;
            } else {
              if (this->has_mean_values_) {
                top_data[top_index] = (pixel - this->mean_values_[mean_c]) * scale;
              } else {
                top_data[top_index] = pixel * scale;
              }
            }
          }
        }
      }

      // copy the warped window c into top_data
      for (int h = 0; h < cv_cropped_img_c.rows; ++h) {
        const uchar* ptr = cv_cropped_img_c.ptr<uchar>(h);
        int img_index = 0;
        for (int w = 0; w < cv_cropped_img_c.cols; ++w) {
          for (int c = 2*channels; c < 3*channels; ++c) {
            int top_index = ((item_id * (3*channels) + c) * crop_size + h + pad_hc)
                     * crop_size + w + pad_wc;
            // int top_index = (c * height + h) * width + w;
            Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
            int mean_c = c % channels;
            if (this->has_mean_file_) {
              int mean_index = (mean_c * mean_height + h + mean_off + pad_hc)
                           * mean_width + w + mean_off + pad_wc;
              top_data[top_index] = (pixel - mean[mean_index]) * scale;
            } else {
              if (this->has_mean_values_) {
                top_data[top_index] = (pixel - this->mean_values_[mean_c]) * scale;
              } else {
                top_data[top_index] = pixel * scale;
              }
            }
          }
        }
      }

      // get window labels      
      trans_time += timer.MicroSeconds();
      top_label[item_id * 3 + 0] = window_a[TripletWindowDataLayer<Dtype>::LABEL];
      top_label[item_id * 3 + 1] = window_b[TripletWindowDataLayer<Dtype>::LABEL];
      top_label[item_id * 3 + 2] = window_c[TripletWindowDataLayer<Dtype>::LABEL];
      

     
      // useful debugging code for dumping transformed windows to disk
      #if 0
      std::pair<std::string, vector<int> > image_debug_a = image_database_[window_a[TripletWindowDataLayer<Dtype>::IMAGE_INDEX]];

      std::pair<std::string, vector<int> > image_debug_b = image_database_[window_b[TripletWindowDataLayer<Dtype>::IMAGE_INDEX]];

      std::pair<std::string, vector<int> > image_debug_c = image_database_[window_c[TripletWindowDataLayer<Dtype>::IMAGE_INDEX]];

      string file_id;
      std::stringstream ss;
      ss << PrefetchRand();
      ss >> file_id;
      std::ofstream inf((string("dump/") + file_id +
          string("_info.txt")).c_str(), std::ofstream::out);
      inf << "Pair_ID = " << pair[TripletWindowDataLayer<Dtype>::PAIR_INDEX] << std::endl;
      
      inf << "Sample" << std::endl;
      inf << image_debug_a.first << std::endl;
      inf << window_a[TripletWindowDataLayer<Dtype>::X1]+1 << std::endl;
      inf << window_a[TripletWindowDataLayer<Dtype>::Y1]+1 << std::endl;
      inf << window_a[TripletWindowDataLayer<Dtype>::X2]+1 << std::endl;
      inf << window_a[TripletWindowDataLayer<Dtype>::Y2]+1 << std::endl;
      inf << window_a[TripletWindowDataLayer<Dtype>::LABEL] << std::endl;
      inf << do_mirrora << std::endl;
      inf << top_label[item_id * 3 + 0] << std::endl;      

      inf << "Target" << std::endl;
      inf << image_debug_b.first << std::endl;
      inf << window_b[TripletWindowDataLayer<Dtype>::X1]+1 << std::endl;
      inf << window_b[TripletWindowDataLayer<Dtype>::Y1]+1 << std::endl;
      inf << window_b[TripletWindowDataLayer<Dtype>::X2]+1 << std::endl;
      inf << window_b[TripletWindowDataLayer<Dtype>::Y2]+1 << std::endl;
      inf << window_b[TripletWindowDataLayer<Dtype>::LABEL] << std::endl;
      inf << do_mirrorb << std::endl;
      inf << top_label[item_id * 3 + 1] << std::endl;

      inf << "Imposto" << std::endl;
      inf << image_debug_c.first << std::endl;
      inf << window_c[TripletWindowDataLayer<Dtype>::X1]+1 << std::endl;
      inf << window_c[TripletWindowDataLayer<Dtype>::Y1]+1 << std::endl;
      inf << window_c[TripletWindowDataLayer<Dtype>::X2]+1 << std::endl;
      inf << window_c[TripletWindowDataLayer<Dtype>::Y2]+1 << std::endl;
      inf << window_c[TripletWindowDataLayer<Dtype>::LABEL] << std::endl;
      inf << do_mirrorc << std::endl;          
      inf << top_label[item_id * 3 + 2] << std::endl;      
      inf.close();

      std::ofstream top_data_file_a((string("dump/") + file_id +
          string("_data.txt")).c_str(),
          std::ofstream::out | std::ofstream::binary);
      for (int c = 0; c < 3*channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            top_data_file_a.write(reinterpret_cast<char*>(
                &top_data[((item_id * (3*channels) + c) * crop_size + h)
                          * crop_size + w]),
                sizeof(Dtype));
          }
        }
      }
      top_data_file_a.close();
      #endif

      item_id++;
    }
  }  
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";  
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
bool TripletWindowDataLayer<Dtype>::prepare_window(cv::Mat &warpimg, int &pad_w, int &pad_h, bool &do_mirror, const vector<float> &window, const bool mirror, const int context_pad, const bool use_square, const int crop_size){

      cv::Size cv_crop_size(crop_size, crop_size);
      do_mirror = mirror && PrefetchRand() % 2;

      // load the image containing the window
      pair<std::string, vector<int> > image =
          image_database_[window[TripletWindowDataLayer<Dtype>::IMAGE_INDEX]];   

      cv::Mat cv_img;
      if (this->cache_images_) {
        pair<std::string, Datum> image_cached =
          image_database_cache_[window[TripletWindowDataLayer<Dtype>::IMAGE_INDEX]];
        cv_img = DecodeDatumToCVMat(image_cached.second, true);
      } else {
        cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
        if (!cv_img.data) {
          LOG(ERROR) << "Could not open or find file " << image.first;
          return false;
        }
      }

      // crop window out of image and warp it
      int x1 = window[TripletWindowDataLayer<Dtype>::X1];
      int y1 = window[TripletWindowDataLayer<Dtype>::Y1];
      int x2 = window[TripletWindowDataLayer<Dtype>::X2];
      int y2 = window[TripletWindowDataLayer<Dtype>::Y2];

      pad_w = 0;
      pad_h = 0;
      if (context_pad > 0 || use_square) {
        // scale factor by which to expand the original region
        // such that after warping the expanded region to crop_size x crop_size
        // there's exactly context_pad amount of padding on each side
        Dtype context_scale = static_cast<Dtype>(crop_size) /
            static_cast<Dtype>(crop_size - 2*context_pad);

        // compute the expanded region
        Dtype half_height = static_cast<Dtype>(y2-y1+1)/2.0;
        Dtype half_width = static_cast<Dtype>(x2-x1+1)/2.0;
        Dtype center_x = static_cast<Dtype>(x1) + half_width;
        Dtype center_y = static_cast<Dtype>(y1) + half_height;
        if (use_square) {
          if (half_height > half_width) {
            half_width = half_height;
          } else {
            half_height = half_width;
          }
        }
        x1 = static_cast<int>(round(center_x - half_width*context_scale));
        x2 = static_cast<int>(round(center_x + half_width*context_scale));
        y1 = static_cast<int>(round(center_y - half_height*context_scale));
        y2 = static_cast<int>(round(center_y + half_height*context_scale));

        // the expanded region may go outside of the image
        // so we compute the clipped (expanded) region and keep track of
        // the extent beyond the image
        int unclipped_height = y2-y1+1;
        int unclipped_width = x2-x1+1;
        int pad_x1 = std::max(0, -x1);
        int pad_y1 = std::max(0, -y1);
        int pad_x2 = std::max(0, x2 - cv_img.cols + 1);
        int pad_y2 = std::max(0, y2 - cv_img.rows + 1);
        // clip bounds
        x1 = x1 + pad_x1;
        x2 = x2 - pad_x2;
        y1 = y1 + pad_y1;
        y2 = y2 - pad_y2;
        CHECK_GT(x1, -1);
        CHECK_GT(y1, -1);
        CHECK_LT(x2, cv_img.cols);
        CHECK_LT(y2, cv_img.rows);

        int clipped_height = y2-y1+1;
        int clipped_width = x2-x1+1;

        // scale factors that would be used to warp the unclipped
        // expanded region
        Dtype scale_x =
            static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_width);
        Dtype scale_y =
            static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_height);

        // size to warp the clipped expanded region to
        cv_crop_size.width =
            static_cast<int>(round(static_cast<Dtype>(clipped_width)*scale_x));
        cv_crop_size.height =
            static_cast<int>(round(static_cast<Dtype>(clipped_height)*scale_y));
        pad_x1 = static_cast<int>(round(static_cast<Dtype>(pad_x1)*scale_x));
        pad_x2 = static_cast<int>(round(static_cast<Dtype>(pad_x2)*scale_x));
        pad_y1 = static_cast<int>(round(static_cast<Dtype>(pad_y1)*scale_y));
        pad_y2 = static_cast<int>(round(static_cast<Dtype>(pad_y2)*scale_y));

        pad_h = pad_y1;
        // if we're mirroring, we mirror the padding too (to be pedantic)
        if (do_mirror) {
          pad_w = pad_x2;
        } else {
          pad_w = pad_x1;
        }

        // ensure that the warped, clipped region plus the padding fits in the
        // crop_size x crop_size image (it might not due to rounding)
        if (pad_h + cv_crop_size.height > crop_size) {
          cv_crop_size.height = crop_size - pad_h;
        }
        if (pad_w + cv_crop_size.width > crop_size) {
          cv_crop_size.width = crop_size - pad_w;
        }
      }

      cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
      cv::Mat cv_cropped_img = cv_img(roi);
      cv::resize(cv_cropped_img, cv_cropped_img,
          cv_crop_size, 0, 0, cv::INTER_LINEAR);
     
      // horizontal flip at random
      if (do_mirror) {
        cv::flip(cv_cropped_img, cv_cropped_img, 1);
      }
      warpimg = cv_cropped_img;      	    

      return true;
}

INSTANTIATE_CLASS(TripletWindowDataLayer);
REGISTER_LAYER_CLASS(TripletWindowData);

}  // namespace caffe
#endif  // USE_OPENCV
