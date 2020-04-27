import argparse
import json
import torch

def parse_args():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", 
						type=str,
						help="Data file path",
						required=True,
						)
	parser.add_argument("--annotation_dir", 
						type=str,
						required=True,
						help="Annotations file path"
						)
	parser.add_argument("--model_dir", 
						type=str,
						required=True,
						help="Models to be saved - file path"
						)
	parser.add_argument("--per_gpu_batch_size", 
						default=8, 
						type=int, 
						help="Batch size per GPU",
						)
	parser.add_argument("--cuda", 
						action="store_true",
						)
	parser.add_argument("--num_train_epochs",
						default=1,
						type=int,
						)
	parser.add_argument("--seed",
						default=0,
						type=int,
						)
	parser.add_argument("--use_bce",
						default=False,
						type=bool,
						)


	args = parser.parse_args()

	if args.cuda:
		args.device = torch.device("cuda")
		args.n_gpu = torch.cuda.device_count()
	else:
		args.device = torch.device("cpu")
		args.n_gpu = 0
	return args