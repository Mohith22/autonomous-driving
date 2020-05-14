# -- Imports -- #
import argparse
import json
import torch

# -- Argument Parameters -- #
def parse_args():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", 
						type=str,
						help="Data file path",
						required=True,
						)
	parser.add_argument("--depth_dir", 
						type=str,
						help="Depth file path",
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
	parser.add_argument("--loss",
						default="dice",
						type=str,
						)
	parser.add_argument("--thres",
						default=0.5,
						type=float,
						)
	parser.add_argument("--depth_avail",
						default=False,
						type=bool,
						)
	parser.add_argument("--siamese",
						default=False,
						type=bool,
						)
	parser.add_argument("--use_orient_net",
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