import tarfile

def untar(tar_path, out_path='.'):
	if tar_path.endswith("tar.gz"):
		tar = tarfile.open(tar_path, "r:gz")
		tar.extractall(out_path)
		tar.close()
	print("File is not .tar.gz")


if __name__ == "__main__":
  print("Running utils.py")

  tar_path = "formula_images.tar.gz"
  untar(tar_path)

  print("Finished.")
