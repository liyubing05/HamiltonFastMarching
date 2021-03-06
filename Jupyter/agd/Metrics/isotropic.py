from .base import Base
from .. import AutomaticDifferentiation as ad

class Isotropic(Base):
	"""
	An isotropic metric, defined through a cost function.
	"""

	def __init__(self,cost,vdim=None):
		self.cost = ad.toarray(cost)
		if vdim is not None:
			self.vdim = vdim

	@classmethod
	def from_speed(cls,speed):
		return cls(1./speed)

	def dual(self):
		other = self.from_speed(self.cost)
		if hasattr(self,'_vdim'): other._vdim = self._vdim
		return other

	def norm(self,v):
		return self.cost*ad.Optimization.norm(v,ord=2,axis=0)

	def is_definite(self):
		return self.cost>0.

	def anisotropy(self):
		return 1.

	@property
	def vdim(self): 
		if self.cost.ndim>0:
			return self.cost.ndim
		elif hasattr(self,'_vdim'):
			return self._vdim
		else:
			raise ValueError("Could not determine dimension of isotropic metric")

	@vdim.setter
	def vdim(self,vdim):
		if self.cost.ndim>0:
			assert(self.cost.ndim==vdim)
		else:
			self._vdim = vdim

	@property
	def shape(self): return self.cost.shape
	
	def rotate(self,a):
		return self

	def flatten(self):		return self.cost
	@classmethod
	def expand(cls,arr):	return cls(arr)

	def to_HFM(self):		return self.cost
	@classmethod
	def from_HFM(cls,arr):	return cls(arr)

	def model_HFM(self):
		return "Isotropic"+str(self.vdim)

	@classmethod
	def from_cast(cls,metric):
		if isinstance(metric,cls): return metric
		raise ValueError("Isotropic.from_cast error : cannot cast an isotropic metric from ",type(metric))

	def __iter__(self):
		yield self.cost

	# TODO : upwind gradient from HFM AD info (with grid)