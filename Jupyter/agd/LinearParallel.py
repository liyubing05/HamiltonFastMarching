import numpy as np
from . import AutomaticDifferentiation as ad
from . import FiniteDifferences as fd

def identity(shape):
	dim = len(shape)
	a = np.full((dim,dim)+shape,0.)
	for i in range(dim):
		a[i,i]=1.
	return a

def rotation(theta,axis=None):
	"""
	Dimension 2 : by a given angle.
	Dimension 3 : by a given angle, along a given axis.
	Three dimensional rotation matrix, with given axis and angle.
	Adapted from https://stackoverflow.com/a/6802723
	"""
	if axis is None:
		c,s=np.cos(theta),np.sin(theta)
		return ad.array([[c,-s],[s,c]])
	else:
		theta,axis = (ad.toarray(e) for e in (theta,axis))
		axis = axis / np.linalg.norm(axis,axis=0)
		theta,axis=fd.common_field((theta,axis),(0,1))
		a = np.cos(theta / 2.0)
		b, c, d = -axis * np.sin(theta / 2.0)
		aa, bb, cc, dd = a * a, b * b, c * c, d * d
		bc, ad_, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
		return ad.array([
			[aa + bb - cc - dd, 2 * (bc + ad_), 2 * (bd - ac)],
			[2 * (bc - ad_), aa + cc - bb - dd, 2 * (cd + ab)],
			[2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
#	return scipy.linalg.expm(np.cross(np.eye(3), axis/scipy.linalg.norm(axis)*theta)) # Alternative

# Dot product (vector-vector, matrix-vector and matrix-matrix) in parallel
def dot_VV(v,w):
	if v.shape[0]!=w.shape[0]: raise ValueError('dot_VV : Incompatible shapes')
	return (v*w).sum(0)

def dot_AV(a,v):
	if a.shape[1]!=v.shape[0]: raise ValueError("dot_AV : Incompatible shapes")
	return (a*np.expand_dims(v,axis=0)).sum(1)

def dot_VA(v,a):
	m,n = a.shape[:2]
	bounds = a.shape[2:]
	if v.shape != (m,)+bounds:
		raise ValueError("dot_VA : Incompatible shapes")

	return (v.reshape((m,1)+bounds)*a).sum(0)


def dot_AA(a,b):
	m,n=a.shape[:2]
	bounds = a.shape[2:]
	k = b.shape[1]
	if b.shape!=(n,k,)+bounds:
		raise ValueError("dot_AA error : Incompatible shapes")
	return (a.reshape((m,n,1)+bounds)*b.reshape((1,n,k)+bounds)).sum(1)

def dot_VAV(v,a,w):
	return dot_VV(v,dot_AV(a,w))
	
# Multiplication by scalar, of a vector or matrix
def mult(k,x):
	bounds = k.shape
	dim = x.ndim-k.ndim
	if x.shape[dim:]!=bounds:
		raise ValueError("mult error : incompatible shapes")
	return k.reshape((1,)*dim+bounds)*x
	

def perp(v):
	if v.shape[0]!=2:
		raise ValueError("perp error : Incompatible dimension")		
	return ad.array( (-v[1],v[0]) )
	
def cross(v,w):
	if v.shape[0]!=3 or v.shape!=w.shape:
		raise ValueError("cross error : Incompatible dimensions")
	return ad.array( (v[1]*w[2]-v[2]*w[1], \
	v[2]*w[0]-v[0]*w[2], v[0]*w[1]-v[1]*w[0]) )
	
def outer(v,w):
	if v.shape[1:] != w.shape[1:]:
		raise ValueError("outer error : Incompatible dimensions")
	m,n=v.shape[0],w.shape[0]
	bounds = v.shape[1:]
	return v.reshape((m,1)+bounds)*w.reshape((1,n)+bounds)

def outer_self(v):
	return outer(v,v)

def transpose(a):
	return a.transpose( (1,0,)+tuple(range(2,a.ndim)) )
	
def trace(a):
	dim = a.shape[0]
	if a.shape[1]!=dim:
		raise ValueError("trace error : incompatible dimensions")
	return a[(range(dim),range(dim))].sum(0)

# Low dimensional special cases

def det(a):
	dim = a.shape[0]
	if a.shape[1]!=dim:
		raise ValueError("inverse error : incompatible dimensions")
	if dim==1:
		return a[0,0]
	elif dim==2:
		return a[0,0]*a[1,1]-a[0,1]*a[1,0]
	elif dim==3:
		return a[0,0]*a[1,1]*a[2,2]+a[0,1]*a[1,2]*a[2,0]+a[0,2]*a[1,0]*a[2,1] \
		- a[0,2]*a[1,1]*a[2,0] - a[1,2]*a[2,1]*a[0,0]- a[2,2]*a[0,1]*a[1,0]
	else:
		raise ValueError("det error : unsupported dimension") 

def inverse(a):
	if isinstance(a,ad.Dense.denseAD):
		b = inverse(a.value)
		b_ = fd.as_field(b,(a.size_ad,),conditional=False) #np.expand_dims(b,axis=-1)
		h = a.coef
		return ad.Dense.denseAD(b,-dot_AA(b_,dot_AA(h,b_)))
	elif isinstance(a,ad.Dense2.denseAD2):
		b = inverse(a.value)
		b1 = fd.as_field(b,(a.size_ad,),conditional=False)
		h = a.coef1
		h2 = a.coef2

		bh = dot_AA(b1,h)
		bhb = dot_AA(bh,b1)
		bhbhb = dot_AA(np.broadcast_to(np.expand_dims(bh,-1),h2.shape),
			np.broadcast_to(np.expand_dims(bhb,-2),h2.shape))

		b2 = fd.as_field(b,(a.size_ad,a.size_ad),conditional=False)
		bh2b = dot_AA(b2,dot_AA(h2,b2))
		return ad.Dense2.denseAD2(b,-bhb,bhbhb+np.swapaxes(bhbhb,-1,-2)-bh2b)
	elif ad.is_ad(a):
		d=len(a)
		return ad.apply(inverse,a,shape_bound=a.shape[2:])
	else:
		return np.moveaxis(np.linalg.inv(np.moveaxis(a,(0,1),(-2,-1))),(-2,-1),(0,1))

def solve_AV(a,v):
	if ad.is_ad(v): return dot_AV(inverse(a),v) # Inefficient, but compatible with ndarray subclasses
	return np.moveaxis(np.linalg.solve(np.moveaxis(a,(0,1),(-2,-1)),np.moveaxis(v,0,-1)),-1,0)			