{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}

module Foreign.CUDA.BLAS.Level1 
 ( 
 -- * Wrappers (preferred)
   sdot
 , ddot

 -- * Raw bindings
 , cublasSdot
 , cublasDdot
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
  where
    useDev      = useDevicePtr . castDevPtr

{# fun unsafe cublasDdot_v2 as cublasDdot 
  { useHandle `Handle' 
  ,           `Int'
  , useDev    `DevicePtr Double'
  ,           `Int'
  , useDev    `DevicePtr Double'
  ,           `Int'
  , useDev    `DevicePtr Double' } -> `Status' cToEnum #}
  where
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