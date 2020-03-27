if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

if __name__ == "__main__":
	model = C3D_model.C3D()
	model.to(device)

	model.eval()
	