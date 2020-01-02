import numpy as np
from . import misc

_add_dim = misc._add_dim; _add_coef=misc._add_coef

class denseAD(np.ndarray):
	"""
	A class for dense forward automatic differentiation
	"""

	# Construction
	# See : https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
	def __new__(cls,value,coef=None,broadcast_ad=False):
		if isinstance(value,denseAD):
			assert coef is None
			return value
		obj = np.asarray(value).view(denseAD)
		shape = obj.shape
		shape2 = shape+(0,)
		obj.coef  = (np.full(shape2,0.) if coef is None 
			else misc._test_or_broadcast_ad(coef,shape,broadcast_ad) )
		return obj

#	def __array_finalize__(self,obj): pass

	def copy(self,order='C'):
		return denseAD(self.value.copy(order=order),self.coef.copy(order=order))

	# Representation 
	def __iter__(self):
		for value,coef in zip(self.value,self.coef):
			yield denseAD(value,coef)

	def __str__(self):
		return "denseAD("+str(self.value)+","+misc._prep_nl(str(self.coef))+")"
#		return "denseAD"+str((self.value,self.coef))
	def __repr__(self):
		return "denseAD("+repr(self.value)+","+misc._prep_nl(repr(self.coef))+")"
#		return "denseAD"+repr((self.value,self.coef))	

	# Operators
	def __add__(self,other):
		if _is_constant(other): return self.__add__(other.view(np.ndarray))
		if isinstance(other,denseAD):
			return denseAD(self.value+other.value, _add_coef(self.coef,other.coef))
		else:
			return denseAD(self.value+other, self.coef, broadcast_ad=True)

	def __sub__(self,other):
		if _is_constant(other): return self.__sub__(other.view(np.ndarray))
		if isinstance(other,denseAD):
			return denseAD(self.value-other.value, _add_coef(self.coef,-other.coef))
		else:
			return denseAD(self.value-other, self.coef, broadcast_ad=True)

	def __mul__(self,other):
		if _is_constant(other): return self.__mul__(other.view(np.ndarray))
		if isinstance(other,denseAD):
			return denseAD(self.value*other.value,_add_coef(_add_dim(other.value)*self.coef,_add_dim(self.value)*other.coef))
		elif isinstance(other,np.ndarray):
			return denseAD(self.value*other,_add_dim(other)*self.coef)
		else:
			return denseAD(self.value*other,other*self.coef)

	def __truediv__(self,other):		
		if _is_constant(other): return self.__truediv__(other.view(np.ndarray))
		if isinstance(other,denseAD):
			return denseAD(self.value/other.value,
				_add_coef(_add_dim(1/other.value)*self.coef,_add_dim(-self.value/other.value**2)*other.coef))
		elif isinstance(other,np.ndarray):
			return denseAD(self.value/other,_add_dim(1./other)*self.coef)
		else:
			return denseAD(self.value/other,(1./other)*self.coef) 

	__rmul__ = __mul__
	__radd__ = __add__
	def __rsub__(self,other): 		return -(self-other)
	def __rtruediv__(self,other): 	return denseAD(other/self.value,_add_dim(-other/self.value**2)*self.coef)

	def __neg__(self):		return denseAD(-self.value,-self.coef)

	# Math functions
	def _math_helper(self,deriv):
		a,b=deriv
		return denseAD(a,_add_dim(b)*self.coef)

	def sqrt(self):			return self**0.5
	def __pow__(self,n):	return self._math_helper(misc.pow1(self.value,n))
	def log(self):			return self._math_helper(misc.log1(self.value))
	def exp(self):			return self._math_helper(misc.exp1(self.value))
	def abs(self):			return self._math_helper(misc.abs1(self.value))
	def sin(self):			return self._math_helper(misc.sin1(self.value))
	def cos(self):			return self._math_helper(misc.cos1(self.value))
	def tan(self):			return self._math_helper(misc.tan1(self.value))
	def arcsin(self):		return self._math_helper(misc.arcsin1(self.value))
	def arccos(self):		return self._math_helper(misc.arccos1(self.value))
	def arctan(self):		return self._math_helper(misc._arctan1(self.value))
	def sinh(self):			return self._math_helper(misc.sinh1(self.value))
	def cosh(self):			return self._math_helper(misc.cosh1(self.value))
	def tanh(self):			return self._math_helper(misc.tanh1(self.value))
	def arcsinh(self):		return self._math_helper(misc.arcsinh1(self.value))
	def arccosh(self):		return self._math_helper(misc.arccosh1(self.value))
	def arctanh(self):		return self._math_helper(misc._arctanh1(self.value))

	@staticmethod
	def compose(a,t):
		assert isinstance(a,denseAD) and all(isinstance(b,denseAD) for b in t)
		b = np.moveaxis(denseAD.concatenate(t,axis=0),0,-1)
		coef = (_add_dim(a.coef)*b.coef).sum(axis=-2)
		return denseAD(a.value,coef)

	#Indexing
	@property
	def value(self): return self.view(np.ndarray)
	@property
	def size_ad(self):  return self.coef.shape[-1]

	def gradient(self,i=None): 
		"""Returns the gradient, or the i-th component of the gradient if specified."""
		return np.moveaxis(self.coef,-1,0) if i is None else self.coef[...,i]

	def __getitem__(self,key):
		ekey = misc.key_expand(key)
		return denseAD(self.value[key], self.coef[ekey])

	def __setitem__(self,key,other):
		ekey = misc.key_expand(key)
		if isinstance(other,denseAD):
			if other.size_ad==0: return self.__setitem__(key,other.view(np.ndarray))
			elif self.size_ad==0: self.coef=np.zeros(self.coef.shape[:-1]+(other.size_ad,))
			self.value[key] = other.value
			self.coef[ekey] =  other.coef
		else:
			self.value[key] = other
			self.coef[ekey] = 0.

	def reshape(self,shape,order='C'):
		shape2 = (shape if isinstance(shape,tuple) else (shape,))+(self.size_ad,)
		return denseAD(self.value.reshape(shape,order=order),self.coef.reshape(shape2,order=order))

	def flatten(self):	return self.reshape( (self.size,) )
	def squeeze(self,axis=None): return self.reshape(self.value.squeeze(axis).shape)

	def broadcast_to(self,shape):
		shape2 = shape+(self.size_ad,)
		return denseAD(np.broadcast_to(self.value,shape), np.broadcast_to(self.coef,shape2) )

	@property
	def T(self):	return self if self.ndim<2 else self.transpose()
	
	def transpose(self,axes=None):
		if axes is None: axes = tuple(reversed(range(self.ndim)))
		axes2 = tuple(axes) +(self.ndim,)
		return denseAD(self.value.transpose(axes),self.coef.transpose(axes2))

	# Reductions
	def sum(self,axis=None,out=None,**kwargs):
		if axis is None: return self.flatten().sum(axis=0,out=out,**kwargs)
		out = denseAD(self.value.sum(axis,**kwargs), self.coef.sum(axis,**kwargs))
		return out

	prod = misc.prod

	def min(self,*args,**kwargs): return misc.min(self,*args,**kwargs)
	def max(self,*args,**kwargs): return misc.max(self,*args,**kwargs)
	def argmin(self,*args,**kwargs): return self.value.argmin(*args,**kwargs)
	def argmax(self,*args,**kwargs): return self.value.argmax(*args,**kwargs)

	def sort(self,*varargs,**kwargs):
		from . import sort
		self=sort(self,*varargs,**kwargs)


	# See https://docs.scipy.org/doc/numpy/reference/ufuncs.html
	def __array_ufunc__(self,ufunc,method,*inputs,**kwargs):

		# Return an np.ndarray for piecewise constant functions
		if ufunc in [
		# Comparison functions
		np.greater,np.greater_equal,
		np.less,np.less_equal,
		np.equal,np.not_equal,

		# Math
		np.floor_divide,np.rint,np.sign,np.heaviside,

		# 'Floating' functions
		np.isfinite,np.isinf,np.isnan,np.isnat,
		np.signbit,np.floor,np.ceil,np.trunc
		]:
			inputs_ = (a.value if isinstance(a,denseAD) else a for a in inputs)
			return super(denseAD,self).__array_ufunc__(ufunc,method,*inputs_,**kwargs)


		if method=="__call__":

			# Reimplemented
			if ufunc==np.maximum: return misc.maximum(*inputs,**kwargs)
			if ufunc==np.minimum: return misc.minimum(*inputs,**kwargs)

			# Math functions
			if ufunc==np.sqrt:		return self.sqrt()
			if ufunc==np.log:		return self.log()
			if ufunc==np.exp:		return self.exp()
			if ufunc==np.abs:		return self.abs()
			if ufunc==np.sin:		return self.sin()
			if ufunc==np.cos:		return self.cos()
			if ufunc==np.tan:		return self.tan()
			if ufunc==np.arcsin:	return self.arcsin()
			if ufunc==np.arccos:	return self.arccos()
			if ufunc==np.arctan:	return self.arctan()
			if ufunc==np.sinh:		return self.sinh()
			if ufunc==np.cosh:		return self.cosh()
			if ufunc==np.tanh:		return self.tanh()
			if ufunc==np.arcsinh:	return self.arcsinh()
			if ufunc==np.arccosh:	return self.arccosh()
			if ufunc==np.arctanh:	return self.arctanh()


			# Operators
			if ufunc==np.add: return self.add(*inputs,**kwargs)
			if ufunc==np.subtract: return self.subtract(*inputs,**kwargs)
			if ufunc==np.multiply: return self.multiply(*inputs,**kwargs)
			if ufunc==np.true_divide: return self.true_divide(*inputs,**kwargs)

		return NotImplemented


	# Numerical 
	def solve(self,shape_free=None,shape_bound=None):
		shape_free,shape_bound = misc._set_shape_free_bound(self.shape,shape_free,shape_bound)
		assert np.prod(shape_free)==self.size_ad
		v = np.moveaxis(np.reshape(self.value,(self.size_ad,)+shape_bound),0,-1)
		a = np.moveaxis(np.reshape(self.coef,(self.size_ad,)+shape_bound+(self.size_ad,)),0,-2)
		return -np.reshape(np.moveaxis(np.linalg.solve(a,v),-1,0),self.shape)

	# Static methods

	# Support for +=, -=, *=, /= 
	@staticmethod
	def add(*args,**kwargs): return misc.add(*args,**kwargs)
	@staticmethod
	def subtract(*args,**kwargs): return misc.subtract(*args,**kwargs)
	@staticmethod
	def multiply(*args,**kwargs): return misc.multiply(*args,**kwargs)
	@staticmethod
	def true_divide(*args,**kwargs): return misc.true_divide(*args,**kwargs)

	@staticmethod
	def stack(elems,axis=0):
		return denseAD.concatenate(tuple(np.expand_dims(e,axis=axis) for e in elems),axis)

	@staticmethod
	def concatenate(elems,axis=0):
		axis1 = axis if axis>=0 else axis-1
		elems2 = tuple(denseAD(e) for e in elems)
		size_ad = max(e.size_ad for e in elems2)
		assert all((e.size_ad==size_ad or e.size_ad==0) for e in elems2)
		return denseAD( 
		np.concatenate(tuple(e.value for e in elems2), axis=axis), 
		np.concatenate(tuple(e.coef if e.size_ad==size_ad else np.zeros(e.shape+(size_ad,)) for e in elems2),axis=axis1))

	def associate(self,squeeze_free_dims=-1,squeeze_bound_dims=-1):
		from . import associate
		sq_free = squeeze_free_dims
		sq_free1= sq_free if sq_free>=0 else (sq_free-1)
		value = associate(self.value,sq_free, squeeze_bound_dims)
		coef  = associate(self.coef, sq_free1,squeeze_bound_dims)
		coef = np.moveaxis(coef,self.ndim if sq_free is None else (self.ndim-1),-1)
		return denseAD(value,coef)

	def apply_linear_operator(self,op):
		return denseAD(op(self.value),misc.apply_linear_operator(op,self.coef,flatten_ndim=1))

# -------- End of class denseAD -------

# -------- Some utility functions, for internal use -------

def _is_constant(a):	return isinstance(a,denseAD) and a.size_ad==0

# -------- Factory method -----

def identity(shape=None,shape_free=None,shape_bound=None,constant=None,shift=(0,0)):
	"""
	Creates a dense AD variable with independent symbolic perturbations for each coordinate
	(unless some are tied together as specified by shape_free and shape_bound)
	"""
	shape,constant = misc._set_shape_constant(shape,constant)
	shape_free,shape_bound = misc._set_shape_free_bound(shape,shape_free,shape_bound)

	ndim_elem = len(shape)-len(shape_bound)
	shape_elem = shape[:ndim_elem]
	size_elem = int(np.prod(shape_elem))
	size_ad = shift[0]+size_elem+shift[1]
	coef1 = np.full((size_elem,size_ad),0.)
	for i in range(size_elem):
		coef1[i,shift[0]+i]=1.
	coef1 = coef1.reshape(shape_elem+(1,)*len(shape_bound)+(size_ad,))
	if coef1.shape[:-1]!=constant.shape: 
		coef1 = np.broadcast_to(coef1,shape+(size_ad,))
	return denseAD(constant,coef1)

def register(inputs,iterables=None,shape_bound=None,shift=(0,0),ident=identity,considered=None):
	"""
	Creates a series of dense AD variables with independent symbolic perturbations for each coordinate,
	and adequate intermediate shifts.
	"""
	if iterables is None:
		iterables = (tuple,)
	boundsize = 1 if shape_bound is None else np.prod(shape_bound,dtype=int)
	def is_considered(a):
		return considered is None or a in considered

	start=shift[0]
	starts = []
	def setstart(a):
		nonlocal start,starts
		if considered is None or any(a is val for val in considered):
			a,to_ad = misc.ready_ad(a)
			if to_ad: 
				starts.append(start)
				start += a.size//boundsize
				return a
		starts.append(None)
		return a
	inputs = misc.map_iterables(setstart,inputs,iterables)

	end = start+shift[1]

	starts_it = iter(starts)
	def setad(a):
		start = next(starts_it)
		if start is None:
			return a
		else:
			return ident(constant=a,shift=(start,end-start-a.size//boundsize),
				shape_bound=shape_bound)
	return misc.map_iterables(setad,inputs,iterables)

