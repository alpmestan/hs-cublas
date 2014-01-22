{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}

module Foreign.CUDA.BLAS.Level1 
 ( 
 -- * Dot products
   sdot
 , ddot

 -- * Absolute sum of elements
 , sasum
 , dasum

 -- * Scalar multiplication and vector addition
 , saxpy
 , daxpy

 -- * Norm2 of vectors
 , snrm2
 , dnrm2

 -- * Scaling
 , sscal
 , dscal

 -- * Raw bindings
 , cublasSdot
 , cublasDdot
 , cublasSasum
 , cublasDasum
 , cublasSaxpy
 , cublasDaxpy
 , cublasSnrm2
 , cublasDnrm2
 , cublasSscal
 , cublasDscal
 ) where

import Foreign.CUDA.BLAS.Internal.C2HS
import Foreign.CUDA.BLAS.Error
import Foreign.CUDA.BLAS.Helper

-- from the 'cuda' package
import Foreign.CUDA.Ptr

import Foreign
import Foreign.C
import Foreign.Storable

#include <cublas_wrap.h>
{# context lib="cublas" #}

{# fun unsafe cublasSdot_v2 as cublasSdot
  { useHandle `Handle' 
  ,           `Int'
  , useDev    `DevicePtr Float'
  ,           `Int'
  , useDev    `DevicePtr Float'
  ,           `Int'
  , useDev    `DevicePtr Float' } -> `Status' cToEnum #}


{# fun unsafe cublasDdot_v2 as cublasDdot 
  { useHandle `Handle' 
  ,           `Int'
  , useDev    `DevicePtr Double'
  ,           `Int'
  , useDev    `DevicePtr Double'
  ,           `Int'
  , useDev    `DevicePtr Double' } -> `Status' cToEnum #}

useDev      = useDevicePtr . castDevPtr

-- | CUBLAS dot product for 'Float's
sdot :: Handle -- ^ CUBLAS context
     -> Int    -- ^ Number of elements in the two input vectors
     -> DevicePtr Float -> Int -- ^ first input vector and its stride
     -> DevicePtr Float -> Int -- ^ second input vector and its stride
     -> DevicePtr Float        -- ^ 0-dimensional output vector
     -> IO ()
sdot h n v1 stride1 v2 stride2 oPtr =
  nothingIfOk =<< cublasSdot h n v1 stride1 v2 stride2 oPtr

-- | CUBLAS dot product for 'Double's
ddot :: Handle -- ^ CUBLAS context
     -> Int    -- ^ Number of elements in the two input vectors
     -> DevicePtr Double -> Int -- ^ first input vector and its stride
     -> DevicePtr Double -> Int -- ^ second input vector and its stride
     -> DevicePtr Double        -- ^ 0-dimensional output vector
     -> IO ()
ddot h n v1 stride1 v2 stride2 oPtr =
  nothingIfOk =<< cublasDdot h n v1 stride1 v2 stride2 oPtr

{# fun unsafe cublasSasum_v2 as cublasSasum
  { useHandle `Handle'
  ,           `Int'
  , useDev    `DevicePtr Float' 
  ,           `Int'
  , useDev    `DevicePtr Float' } -> `Status' cToEnum #}

{# fun unsafe cublasDasum_v2 as cublasDasum
  { useHandle `Handle'
  ,           `Int'
  , useDev    `DevicePtr Double' 
  ,           `Int'
  , useDev    `DevicePtr Double' } -> `Status' cToEnum #}

-- | Absolute sum of vector elements for 'Float's
sasum :: Handle -- ^ CUBLAS context
      -> Int    -- ^ Number of elements in the input vector
      -> DevicePtr Float -> Int -- ^ first input vector and its stride
      -> DevicePtr Float        -- ^ 0-dim output vector
      -> IO ()
sasum h n v stride outPtr = nothingIfOk =<< cublasSasum h n v stride outPtr

-- | Absolute sum of vector elements for 'Double's
dasum :: Handle -- ^ CUBLAS context
      -> Int    -- ^ Number of elements in the input vector
      -> DevicePtr Double -> Int -- ^ input vector and its stride
      -> DevicePtr Double        -- ^ 0-dim output vector
      -> IO ()
dasum h n v stride outPtr = nothingIfOk =<< cublasDasum h n v stride outPtr

{# fun unsafe cublasSaxpy_v2 as cublasSaxpy 
  { useHandle `Handle'
  ,           `Int'
  , useDev    `DevicePtr Float'
  , useDev    `DevicePtr Float' 
  ,           `Int'
  , useDev    `DevicePtr Float' 
  ,           `Int' } -> `Status' cToEnum #}

{# fun unsafe cublasDaxpy_v2 as cublasDaxpy
  { useHandle `Handle' 
  ,           `Int'
  , useDev    `DevicePtr Double'
  , useDev    `DevicePtr Double' 
  ,           `Int'
  , useDev    `DevicePtr Double' 
  ,           `Int' } -> `Status' cToEnum #}

-- | Vector scaling and addition all at once, for 'Float's
saxpy :: Handle          -- ^ CUBLAS context
      -> Int             -- ^ Number of elements in the vector
      -> DevicePtr Float -- ^ Scalar /alpha/, by which we scale the first vector
      -> DevicePtr Float -- ^ first vector, /x/
      -> Int             -- ^ stride for /x/
      -> DevicePtr Float -- ^ second vector, /y/, which gets overwritten by the result of /alpha*x + y/
      -> Int             -- ^ stride for /y/
      -> IO ()
saxpy h n alpha x strX y strY = nothingIfOk =<< cublasSaxpy h n alpha x strX y strY

-- | Vector scaling and addition all at once, for 'Double's
daxpy :: Handle           -- ^ CUBLAS context
      -> Int              -- ^ Number of elements in the vector
      -> DevicePtr Double -- ^ Scalar /alpha/, by which we scale the first vector
      -> DevicePtr Double -- ^ first vector, /x/
      -> Int              -- ^ stride for /x/
      -> DevicePtr Double -- ^ second vector, /y/, which gets overwritten by the result of /alpha.x + y/
      -> Int              -- ^ stride for /y/
      -> IO ()
daxpy h n alpha x strX y strY = nothingIfOk =<< cublasDaxpy h n alpha x strX y strY

{# fun unsafe cublasSnrm2_v2 as cublasSnrm2
  { useHandle `Handle'
  ,           `Int'
  , useDev    `DevicePtr Float'
  ,           `Int'
  , useDev    `DevicePtr Float' } -> `Status' cToEnum #}

{# fun unsafe cublasDnrm2_v2 as cublasDnrm2
  { useHandle `Handle'
  ,           `Int'
  , useDev    `DevicePtr Double'
  ,           `Int'
  , useDev    `DevicePtr Double' } -> `Status' cToEnum #}

-- | Norm2 of a vector, for 'Float' elements
snrm2 :: Handle           -- ^ CUBLAS context
      -> Int              -- ^ number of element in /x/
      -> DevicePtr Float  -- ^ vector, /x/
      -> Int              -- ^ stride for /x/
      -> DevicePtr Float  -- ^ output scalar, equal to the norm2 of /x/
      -> IO ()
snrm2 h n x strX out = nothingIfOk =<< cublasSnrm2 h n x strX out

-- | Norm2 of a vector, for 'Double' elements
dnrm2 :: Handle           -- ^ CUBLAS context
      -> Int              -- ^ number of element in /x/
      -> DevicePtr Double -- ^ vector, /x/
      -> Int              -- ^ stride for /x/
      -> DevicePtr Double -- ^ output scalar, equal to the norm2 of /x/
      -> IO ()
dnrm2 h n x strX out = nothingIfOk =<< cublasDnrm2 h n x strX out

{# fun unsafe cublasSscal_v2 as cublasSscal 
  { useHandle `Handle'
  ,           `Int'
  , useDev    `DevicePtr Float'
  , useDev    `DevicePtr Float'
  ,           `Int' } -> `Status' cToEnum #}

{# fun unsafe cublasDscal_v2 as cublasDscal 
  { useHandle `Handle'
  ,           `Int'
  , useDev    `DevicePtr Double'
  , useDev    `DevicePtr Double'
  ,           `Int' } -> `Status' cToEnum #}

-- | Scales a vector in-place, for 'Float's
sscal :: Handle           -- ^ CUBLAS context
      -> Int              -- ^ number of elements in /x/
      -> DevicePtr Float  -- ^ scalar /alpha/ by which we scale /x/
      -> DevicePtr Float  -- ^ vector, /x/, which gets overwritten by the result of /alpha.x/
      -> Int              -- ^ stride for /x/
      -> IO ()
sscal h n alpha x strX = nothingIfOk =<< cublasSscal h n alpha x strX

-- | Scales a vector in-place, for 'Double's
dscal :: Handle           -- ^ CUBLAS context
      -> Int              -- ^ number of elements in /x/
      -> DevicePtr Double -- ^ scalar /alpha/ by which we scale /x/
      -> DevicePtr Double -- ^ vector, /x/, which gets overwritten by the result of /alpha.x/
      -> Int              -- ^ stride for /x/
      -> IO ()
dscal h n alpha x strX = nothingIfOk =<< cublasDscal h n alpha x strX