{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}

module Foreign.CUDA.BLAS.Internal.Level1 
 ( cublasSdot
 , cublasDdot
 ) where

import Foreign.CUDA.BLAS.Internal.C2HS
import Foreign.CUDA.BLAS.Internal.Error
import Foreign.CUDA.BLAS.Internal.Helper

-- from the 'cuda' package
import Foreign.CUDA.Ptr

import Foreign
import Foreign.C
import Foreign.Storable

#include <cublas_wrap.h>
{# context lib="cublas" #}

{# fun unsafe cublasSdot_v2 
  { useHandle `Handle' 
  ,           `Int'
  , useDev    `DevicePtr Float'
  ,           `Int'
  , useDev    `DevicePtr Float'
  ,           `Int'
  , useDev    `DevicePtr Float' } -> `Status' cToEnum #}
  where
    useDev      = useDevicePtr . castDevPtr

{# fun unsafe cublasDdot_v2 
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
cublasSdot :: Handle -- ^ CUBLAS context
           -> Int    -- ^ Number of elements in the two input vectors
           -> DevicePtr Float -> Int -- ^ first input vector and its stride
           -> DevicePtr Float -> Int -- ^ second input vector and its stride
           -> DevicePtr Float        -- ^ output vector (one element, scalar)
           -> IO ()
cublasSdot h n v1 stride1 v2 stride2 outVector =
  nothingIfOk =<< cublasSdot_v2 h n v1 stride1 v2 stride2 outVector

-- | CUBLAS dot product for 'Double's
cublasDdot :: Handle -- ^ CUBLAS context
           -> Int    -- ^ Number of elements in the two input vectors
           -> DevicePtr Double -> Int -- ^ first input vector and its stride
           -> DevicePtr Double -> Int -- ^ second input vector and its stride
           -> DevicePtr Double        -- ^ output vector (one element, scalar)
           -> IO ()
cublasDdot h n v1 stride1 v2 stride2 outVector =
  nothingIfOk =<< cublasDdot_v2 h n v1 stride1 v2 stride2 outVector