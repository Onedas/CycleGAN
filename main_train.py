import os
from config import get_arguments
from cycleGAN import CycleGANModel
from dataloader import load_horse2zebra

if __name__ == "__main__":
	parser = get_arguments()
	opt = parser.parse_args()
	print(opt)

	os.makedirs(opt.save_path, exist_ok=True)
	
	# model
	model = CycleGANModel(opt)

	# data load
	train_loader, valid_loader = load_horse2zebra(opt)

	model.train(opt, train_loader, valid_loader)
	print('train done')
