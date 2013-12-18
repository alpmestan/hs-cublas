{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}

module Foreign.CUDA.BLAS.Internal.Helper
  ( Handle (..)
  , cublasCreate
  , cublasDestroy
  , cublasVersion
  ) where

import Foreign.CUDA.BLAS.Internal.C2HS
import Foreign.CUDA.BLAS.Internal.Error

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
cublasCreate :: IO Handle
cublasCreate = statusIfOk =<< cublasCreate_v2

{# fun unsafe cublasCreate_v2 
  { alloca- `Handle' peekHdl* } -> `Status' cToEnum #}
  where 
  	peekHdl = liftM Handle . peek

-- | Destroy a CUBLAS context created with 'cublasCreate'
cublasDestroy :: Handle -> IO ()
cublasDestroy h = nothingIfOk =<< cublasDestroy_v2 h

{# fun unsafe cublasDestroy_v2
  { useHandle `Handle' } -> `Status' cToEnum #}

-- | Get the version of CUBLAS
cublasVersion :: Handle -> IO CInt
cublasVersion h = statusIfOk =<< cublasGetVersion_v2 h

{# fun unsafe cublasGetVersion_v2 
  { useHandle `Handle' 
  , alloca- `CInt' peek* } -> `Status' cToEnum #}