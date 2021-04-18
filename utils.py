import tarfile
import pickle

def untar(tar_path, out_path='./data'):
	if tar_path.endswith("tar.gz") or tar_path.endswith(".tgz"):
		tar = tarfile.open(tar_path, "r:gz")
		tar.extractall(out_path)
		tar.close()
	print("File is not .tar.gz")

def load_pkl(fpath):
	with open(fpath, "rb", encoding="utf-8") as f:
		print(f)
	f.close()
	return data

if __name__ == "__main__":
  print("Running utils.py")

  tar_path = "./data/im2text.tgz"
  untar(tar_path)

  # formula_lst = "./data/im2latex_formulas.lst"
  # print(load_lst(formula_lst))

  print("Finished.")
