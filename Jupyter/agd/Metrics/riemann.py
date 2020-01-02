import numpy as np

from . import misc
from .base import Base
from .isotropic import Isotropic
from .. import LinearParallel as lp
from .. import AutomaticDifferentiation as ad
from .. import FiniteDifferences as fd

class Riemann(Base):

	def __init__(self,m):
		self.m=m

	def norm(self,v):
		v,m = fd.common_field((v,self.m),(1,2))
		return np.sqrt(lp.dot_VAV(v,m,v))

	def dual(self):
		return Riemann(lp.inverse(self.m))

	@property
	def vdim(self): return len(self.m)

	@property
	def shape(self): return self.shape.m[2:]	

	def eigvals(self):
		return np.moveaxis(np.linalg.eigvals(np.moveaxis(self.m,(0,1),(-2,-1))),-1,0)
	def is_definite(self):
		return self.eigvals().min(axis=0)>0
	def anisotropy(self):
		ev = self.eigvals()
		return np.sqrt(ev.max(axis=0)/ev.min(axis=0))

	def inv_transform(self,a):
		return Riemann(lp.dot_AA(lp.transpose(a),lp.dot_AA(self.m,a)))

	def flatten(self):
		return misc.flatten_symmetric_matrix(self.m)

	@classmethod
	def expand(cls,arr):
		return cls(misc.expand_symmetric_matrix(arr))

	def model_HFM(self):
		return "Riemann"+str(self.vdim)

	@classmethod
	def needle(cls,u,cost_parallel,cost_orthogonal,ret_u=False):
		"""
		Defines a Riemannian metric, with 
		- eigenvector u
		- eigenvalue cost_parallel**2 in the eigenspace spanned by u
		- eigenvalue cost_orthogonal**2 in the eigenspace orthogonal with u

		The metric is 
		- needle-like if cost_parallel < cost_orthogonal
		- plate-like otherwise

		Optional argument:
		- ret_u : wether to return the (normalized) vector u
		"""
		u,cost_parallel,cost_orthogonal = (ad.toarray(e) for e in (u,cost_parallel,cost_orthogonal))
		u,cost_parallel,cost_orthogonal = fd.common_field((u.copy(),cost_parallel,cost_orthogonal),(1,0,0))
		
		# Eigenvector normalization
		nu = ad.Optimization.norm(u,ord=2,axis=0)
		mask = nu>0
		if not u.flags.writeable: u = u.copy()
		u[:,mask] /= nu[mask]

		ident = fd.as_field(np.eye(len(u)),cost_parallel.shape,conditional=False)

		m = (cost_parallel**2-cost_orthogonal**2) * lp.outer_self(u) + cost_orthogonal**2 * ident
		return (cls(m),u) if ret_u else cls(m)

	@classmethod
	def from_diagonal(cls,*args):
		args = ad.array(args)
		z = np.zeros(args[0].shape)
		vdim = len(args)
		arr = ad.array([[z if i!=j else args[i] for i in range(vdim)] for j in range(vdim)])
		return cls(arr)

	@classmethod
	def from_cast(cls,metric):
		if isinstance(metric,cls): return metric
		isotropic = Isotropic.from_cast(metric)
		return Riemann.from_diagonal( *(isotropic.cost**2,)*isotropic.vdim )

	def __iter__(self):
		yield self.m
