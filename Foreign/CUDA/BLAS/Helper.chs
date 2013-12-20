{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}

module Foreign.CUDA.BLAS.Helper
  ( -- * CUBLAS context
    Handle (..)
  , create
  , destroy
    -- * Pointer mode
  , PointerMode(..)  
  , getPointerMode
  , setPointerMode
    -- * Misc
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

-- | Pointer mode type.
--   'PointerModeHost' to require the scalars to be passed /by reference on the host/
--   'PointerModeDevice' to require the scalars to be passed /by reference on the device/
{# enum cublasPointerMode_t as PointerMode
  { underscoreToCase }
  with prefix="CUBLAS" deriving (Eq, Show) #}


-- | Set the pointer mode
setPointerMode :: Handle -> PointerMode -> IO ()
setPointerMode h p = nothingIfOk =<< cublasSetPointerMode h p

-- | Get the pointer mode
getPointerMode :: Handle -> IO PointerMode
getPointerMode h = statusIfOk =<< cublasGetPointerMode h

{# fun unsafe cublasSetPointerMode_v2 as cublasSetPointerMode 
  { useHandle `Handle'
  , cFromEnum `PointerMode' } -> `Status' cToEnum #}

{# fun unsafe cublasGetPointerMode_v2 as cublasGetPointerMode 
  { useHandle `Handle'
  , alloca-   `PointerMode' peekPM* } -> `Status' cToEnum #}
  where peekPM = liftM cToEnum . peek
