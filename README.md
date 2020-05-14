# A Generic Approach to Predict Bird’s-Eye-View from Multi-View Scenes through Self-Supervised Monocular Depth Estimation

### Prof. Yann LeCun's "Deep Learning" course project at New York University, USA

### Contributors: Vikas Patidar, Mohith Damarapati, Alfred Ajay Aureate Rajakumar

## Run command 
python *main_file*.py --data_dir "data-dir" --depth_dir "depth-data-dir" --annotation_dir "data-dir/annotation.csv" --model_dir "save-models" --cuda --num_train_epochs 2 --per_gpu_batch_size 2 --depth_avail True --loss bce --siamese True --use_orient_net True 

# Abstract

This work talks about an easy and effective approach to address a very challenging and interesting task of road layout estimation in complex driving environment. From six camera images
encompassing the whole 360&deg; view, we try to predict the bird’s-eye-view of the road and surrounding objects of the ego car. We present an effective generic approach to handle both the tasks
by reducing each of them to an Instance segmentation problem. Further by leveraging the unlabeled
dataset and data augmentation techniques, we estimate depth in an unsupervised manner. Finally
using pretrained depth and novel architectures, we
accurately generate bird’s-eye-view of a scene.

