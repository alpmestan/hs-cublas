{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}

module Foreign.CUDA.BLAS.Helper
  ( Handle (..)
  , create
  , destroy
  , version
  ) where

import Foreign.CUDA.BLAS.Internal.C2HS
import Foreign.CUDA.BLAS.Error

import Control.Monad (liftM)
import Foreign
import Foreign.C

#include <cublas_wrap.h>
{# context lib="cublas" #}

-- | Handle type
--   Pointer to an opaque context
newtype Handle = Handle { useHandle :: {# type cublasHandle_t #}}

-- | Create a CUBLAS context. 
--   Must be called before calling any other CUBLAS function
create :: IO Handle
create = statusIfOk =<< cublasCreate

{# fun unsafe cublasCreate_v2 as cublasCreate
  { alloca- `Handle' peekHdl* } -> `Status' cToEnum #}
  where 
  	peekHdl = liftM Handle . peek

-- | Destroy a CUBLAS context created with 'cublasCreate'
destroy :: Handle -> IO ()
destroy h = nothingIfOk =<< cublasDestroy h

{# fun unsafe cublasDestroy_v2 as cublasDestroy
  { useHandle `Handle' } -> `Status' cToEnum #}

-- | Get the version of CUBLAS
version :: Handle -> IO CInt
version h = statusIfOk =<< cublasGetVersion h

{# fun unsafe cublasGetVersion_v2 as cublasGetVersion 
  { useHandle `Handle' 
  , alloca- `CInt' peek* } -> `Status' cToEnum #}