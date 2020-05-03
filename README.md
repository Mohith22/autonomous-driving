# autonomous-driving
Autonomous Driving - Yann LeCun's "Deep Learning" course challenge

Road Map Detection 

python main.py --data_dir "../../data" --depth_dir "../../data_dir" --annotation_dir "../../data/annotation.csv" --model_dir "Mini_Enc_Dec" --cuda --num_train_epochs 50 --per_gpu_batch_size 2 --depth_avail True --loss bce --siamese True --use_orient_net True  
