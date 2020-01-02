import nbformat 
from nbconvert.preprocessors import ExecutePreprocessor,CellExecutionError
import sys
import os

# ------- Specific to this repository -----

result_path = "test_results"

# --------- Generic ---------

def ListNotebooks(dir=None):
	filenames_extensions = [os.path.splitext(f) for f in os.listdir(dir)]
	filenames = [filename for filename,extension in filenames_extensions if extension==".ipynb"]
	subdirectories = [filename for filename,extension in filenames_extensions 
	if extension=="" and filename.startswith("Notebooks_") and filename!="Notebooks_Repro"]
	subfilenames = [os.path.join(subdir,file) for subdir in subdirectories for file in ListNotebooks(subdir)]
	return filenames+subfilenames


def TestNotebook(notebook_filename, result_path):
	print("Testing notebook " + notebook_filename)
	filename,extension = os.path.splitext(notebook_filename)
	if extension=='': extension='.ipynb'
	filename_out = filename+"_out"
	with open(filename+extension, encoding='utf8') as f:
		nb = nbformat.read(f,as_version=4) # alternatively nbformat.NO_CONVERT
	ep = ExecutePreprocessor(timeout=600,kernel_name='python3')
	success = True
	try:
		out = ep.preprocess(nb,{}) #, {'metadata': {'path': run_path}}
	except CellExecutionError as e:
		msg = 'Error executing the notebook "%s".\n\n' % notebook_filename
		msg += 'See notebook "%s" for the traceback.' % filename_out+extension
		print(msg)
		print(str(e))
		success=False
	finally:
		subdir,file = os.path.split(filename_out)
		with open(os.path.join(subdir,result_path,file)+extension, mode='wt') as f:
			nbformat.write(nb, f)
		return success

if __name__ == '__main__':
#	if not os.path.exists(result_path): os.mkdir(result_path)
	notebook_filenames = sys.argv[1:] if len(sys.argv)>=2 else ListNotebooks()
	notebooks_failed = []
	for notebook_filename in notebook_filenames:
		if not TestNotebook(notebook_filename,result_path):
			notebooks_failed.append(notebook_filename)

	if len(notebooks_failed)>0:
		print("!!! Failure !!! The following notebooks raised errors:\n"+" ".join(notebooks_failed))
	else:
		print("Success ! All notebooks completed without errors.")
