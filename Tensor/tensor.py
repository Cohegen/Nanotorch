import numpy as np

#constants for memory calculations
BYTES_PER_FLOAT32 = 4 #memory to be allocated to varibles of type float32

#conversion metric to be used in conversion of KB to bytes
KB_TO_BYTES = 1024

#conversion metric to be used in conversion of MB to bytes
MB_TO_BYTES = 1024*1024


#class the performs the core ML operations
class Tensor():
    def __init__(self,data, requires_grad=False):
        """Creating a new tensor from data"""

        #1.Converting data to Numpy array with dtype=float32
        # Handle list of tensors
        if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], Tensor):
            data = [t.data if isinstance(t, Tensor) else t for t in data]
        elif isinstance(data, Tensor):
            requires_grad = data.requires_grad or requires_grad
            data = data.data

        self.data = np.array(data,dtype=np.float32)
        #2. Setting self.shape from the array's shape
        self.shape = self.data.shape
        #3. Setting self.size_val from the array's size (renamed from size to not conflict with size())
        self.size_val = self.data.size
        #4. Setting self.dtype from the array's size
        self.dtype= self.data.dtype
        
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
        self.device = "cpu"

    @property
    def num_elements(self):
        """Returns the total number of elements in the tensor"""
        return self.size_val

    def size(self, dim=None):
        """Returns the size of the tensor along a given dimension"""
        if dim is None:
            return self.shape
        return self.shape[dim]

    def view(self, *shape):
        """Alias for reshape to match PyTorch API"""
        return self.reshape(*shape)

    def contiguous(self):
        """Returns the tensor itself (since we're always using NumPy arrays)"""
        return self

    def split(self, split_size, dim=0):
        """Splits the tensor into chunks along a given dimension"""
        n = self.shape[dim]
        num_splits = n // split_size
        indices = [split_size * i for i in range(1, num_splits)]
        splits = np.split(self.data, indices, axis=dim)
        return [Tensor(s) for s in splits]

    def masked_fill(self, mask, value):
        """Fills elements of the tensor with value where mask is True"""
        mask_data = mask.data if isinstance(mask, Tensor) else mask
        # Using np.where handles broadcasting correctly
        new_data = np.where(mask_data.astype(bool), value, self.data)
        return Tensor(new_data)

    def numel(self):
        """Returns the total number of elements in the tensor"""
        return self.size_val

    @property
    def ndim(self):
        """Returns the number of dimensions"""
        return self.data.ndim

    def dim(self):
        """Returns the number of dimensions (PyTorch style)"""
        return self.ndim

    def __array__(self, dtype=None):
        """Allowing NumPy to treat Tensor as an array-like object"""
        if dtype:
            return self.data.astype(dtype)
        return self.data

    def __repr__(self):
        """String representation of a tensor for debugging"""
        return f"Tensor(data={self.data}),shape={self.shape}"

    def __str__(self):
        """Human readable string representation"""
        return f"Tensor({self.data})"

    def numpy(self):
        """Return the underlying Numpy array"""
        return self.data

    def tolist(self):
        """Return the tensor data as a list"""
        return self.data.tolist()

    def __len__(self):
        """Returns the length of the first dimension of the tensor"""
        if len(self.shape) == 0:
            return 1
        return self.shape[0]

    def __eq__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data == other_data)

    def __lt__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data < other_data)

    def __gt__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data > other_data)

    def __le__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data <= other_data)

    def __ge__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data >= other_data)

    def __ne__(self, other):
        other_data = other.data if isinstance(other, Tensor) else other
        return Tensor(self.data != other_data)

    def __add__(self,other):
        """Add two tensors element-wise with broadcasting supporting"""

        #checking is other is a tensor
        if isinstance(other,Tensor):
            return Tensor(self.data + other.data)
        ##applying broadcasting
        else:
            return Tensor(self.data + other)


    def __sub__(self,other):
        """Subtract two tensors elementwise"""

        #checking if other is a tensor
        if isinstance(other,Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __mul__(self,other):
        """Multiplying two tensors elemetwise which not the same as matrix multiplication"""

        ##checking if other is a tensor
        if isinstance(other,Tensor):
            return Tensor(self.data*other.data)
        else:
            return Tensor(self.data*other)

    def __truediv__(self,other):
        """Divide two tensors element-wise"""

        if isinstance(other,Tensor):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)

    def matmul(self,other):
        """Performing matrix multiplication of two tensors"""
        if  not isinstance(other,Tensor):
            raise TypeError(f"Expected Tensor for matrix multiplication, got{type(other)}")
        #checking for scalar cases
        if self.shape ==() or other.shape ==():
            return Tensor(self.data*other.data)
        if len(self.shape) == 0 or len(other.shape) == 0:
            return Tensor(self.data*other.data)
        
         ##checking for 2D+ matrices
        if len(self.shape) >= 2 and len(other.shape) >= 2:
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(
                    f"Cannot perform matrix multiplication:{self.shape} @ {other.shape}"
                    f"Inner dimensions must match: {self.shape[-1]} not equal to {other.shape[-2]}"

                )

        a = self.data
        b = other.data

        result_data = np.matmul(a,b)

        return Tensor(result_data)

    def __matmul__(self,other):
        """Enabling @ operator for matmul"""
        return self.matmul(other)

    def __getitem__(self,key):
        """Enabling indexing and slicing operations of tensors"""

        result_data = self.data[key]

        if not isinstance(result_data,np.ndarray):
            result_data = np.array(result_data)

        return Tensor(result_data)


    def reshape(self,*shape):
        """Reshaping tensor to new dimensions"""

        if len(shape) == 1 and isinstance(shape[0],(tuple,list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = shape
        if -1 in new_shape:
            if list(new_shape).count(-1) > 1:
                raise ValueError("Can only specify one unknown dimension with -1")
            known_size=1
            unknown_idx = new_shape.index(-1)
            for i,dim in enumerate(new_shape):
                if i != unknown_idx:
                    known_size *= dim
            unknown_dim = self.size_val // known_size
            new_shape = list(new_shape)
            new_shape[unknown_idx] = unknown_dim
            new_shape = tuple(new_shape)

        if np.prod(new_shape) != self.size_val:
            target_size = int(np.prod(new_shape))
            raise ValueError(
                f"Total elements must match: {self.size_val} not equal to {target_size}"

            )
        reshaped_data =np.reshape(self.data,new_shape)
        return Tensor(reshaped_data) 

    def transpose(self,dim0=None,dim1=None):
        """Transpose tensor dimensions."""

        if dim0 is None and dim1 is None:
            ## returning a copy of the tensor
            ##since data has one dimension
            if len(self.shape) < 2:
                return Tensor(self.data.copy())

            ##swapping the specified dimensions
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                transposed_data = np.transpose(self.data, axes)
        else:
            if dim0 is None or dim1 is None:
                raise ValueError("Both dim0 and dim1 must be specified")
            axes = list(range(len(self.shape)))
            axes[dim0],axes[dim1] = axes[dim1],axes[dim0]
            transposed_data = np.transpose(self.data,axes)
        return Tensor(transposed_data)


    def sum(self,axis=None,keepdims=False):
        """Summing a tensor along a specified axis"""

        result =np.sum(self.data,axis=axis,keepdims=keepdims)
        return Tensor(result)

    def mean(self,axis=None,keepdims=False):
        """Calculating mean along a specified axis"""
        result = np.mean(self.data,axis=axis,keepdims=keepdims)
        return Tensor(result)

    def max(self,axis=None,keepdims=False):
        """Finding maximum values along a specified axis"""
        result = np.max(self.data,axis=axis,keepdims=keepdims)
        return Tensor(result)
    


##testing 
def  tensor_creation():
    """Testing tensor creating with various data types."""

    #testing scalar scalar creation
    scalar = Tensor(5.0)
    assert scalar.data == 5.0
    assert scalar.shape == ()
    assert scalar.num_elements == 1 
    assert scalar.dtype == np.float32

    #testing vector creation

    vector = Tensor([1,2,3])
    assert np.array_equal(vector.data,np.array([1,2,3],dtype=np.float32))
    assert vector.shape == (3,)
    assert vector.num_elements == 3


    ##testing matrix creation
    matrix = Tensor([[1,2],[3,5]])
    assert np.array_equal(matrix.data,np.array([[1,2],[3,5]],dtype=np.float32))
    assert matrix.shape == (2,2)
    assert matrix.num_elements == 4

    #test 3D tensor creation
    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert tensor_3d.shape == (2,2,2)
    assert tensor_3d.num_elements == 8

    print("Tensor creation works correctly")

if __name__ == "__main__":
    tensor_creation()


