import numpy as np
import importlib

from .Grid import GetCorners,Rect,GetAxes,GetGrid,PointFromIndex,IndexFromPoint,CenteredLinspace
from .LibraryCall import GetBinaryDir
from .run_detail import RunRaw,RunSmart,Cache

def reload_submodules():
	from importlib import reload
	import sys
	hfm = sys.modules['agd.HFMUtils']

	global GetCorners,Rect,GetAxes,GetGrid,PointFromIndex,IndexFromPoint,CenteredLinspace
	hfm.Grid = reload(hfm.Grid)
	GetCorners,Rect,GetAxes,GetGrid,PointFromIndex,IndexFromPoint,CenteredLinspace = (
		Grid.GetCorners,Grid.Rect,Grid.GetAxes,Grid.GetGrid,Grid.PointFromIndex,Grid.IndexFromPoint,Grid.CenteredLinspace)

	global GetBinaryDir
	hfm.LibraryCall = reload(hfm.LibraryCall)
	GetBinaryDir =  LibraryCall.GetBinaryDir

	global RunRaw,RunSmart
	hfm.run_detail = reload(hfm.run_detail)
	RunSmart =  run_detail.RunSmart
	RunRaw = run_detail.RunRaw

def Run(hfmIn,smart=False,**kwargs):
	"""
	Calls to the HFM library, returns output and prints log.

	Parameters
	----------
	smart : bool  
		Choose between a smart and raw run
	**kwargs
		Passed to RunRaw or RunSmart
	"""
	return RunSmart(hfmIn,**kwargs) if smart else RunRaw(hfmIn,**kwargs)

def VoronoiDecomposition(arr):
	"""
	Calls the FileVDQ library to decompose the provided quadratic form(s),
	as based on Voronoi's first reduction of quadratic forms.
	"""
	from ..Metrics import misc
	from . import FileIO
	bin_dir = GetBinaryDir("FileVDQ",None)
	vdqIn ={'tensors':np.moveaxis(misc.flatten_symmetric_matrix(arr),0,-1)}
	vdqOut = FileIO.WriteCallRead(vdqIn, "FileVDQ", bin_dir)
	return np.moveaxis(vdqOut['weights'],-1,0),np.moveaxis(vdqOut['offsets'],[-1,-2],[0,1])


# ----- Basic utilities for HFM input and output -----

def GetGeodesics(output,suffix=''): 
	if suffix != '' and not suffix.startswith('_'): suffix='_'+suffix
	return np.vsplit(output['geodesicPoints'+suffix],
					 output['geodesicLengths'+suffix].cumsum()[:-1].astype(int))

# ----------- Helper class ----------

class dictIn(dict):
	"""
	A very shallow subclass of a python dictionnary, intended for storing the inputs to the HFM library.
	Usage: a number of the free functions of HFMUtils are provided as methods, for convenience.
	"""

	# Coordinates related methods
	@property
	def Corners(self):
		return GetCorners(self)
	def SetRect(self,*args,**kwargs):
		self.update(Rect(*args,**kwargs))

	def copy(self):
		return dictIn(dict.copy(self))

	Axes=GetAxes
	Grid=GetGrid
	PointFromIndex=PointFromIndex
	IndexFromPoint=IndexFromPoint

	# Running
	Run = Run
	RunRaw = RunRaw
	RunSmart = RunSmart






