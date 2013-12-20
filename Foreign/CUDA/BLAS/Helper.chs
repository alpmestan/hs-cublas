{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}

module Foreign.CUDA.BLAS.Helper
  ( -- * CUBLAS context
    Handle (..)
  , create
  , destroy
  , withCublas
    -- * Pointer mode
  , PointerMode(..)  
  , getPointerMode
  , setPointerMode
    -- * Stream related
  , module CS
  , setStream
  , getStream
    -- * Atomics mode
  , AtomicsMode (..)
  , setAtomicsMode
  , getAtomicsMode
    -- * Misc
  , version
  ) where

import Foreign.CUDA.BLAS.Internal.C2HS
import Foreign.CUDA.BLAS.Error

import Control.Exception (bracket)
import Control.Monad     (liftM)
import Foreign
import Foreign.C
import qualified Foreign.CUDA.Runtime.Stream as CS

#include <cublas_wrap.h>
{# context lib="cublas" #}

-- | 'Handle' type.
--   Pointer to an opaque context.
-- 
--   See <http://docs.nvidia.com/cuda/cublas/#cublashandle_t>
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
--
--   See <http://docs.nvidia.com/cuda/cublas/#cublaspointermode_t>
{# enum cublasPointerMode_t as PointerMode
  { underscoreToCase }
  with prefix="CUBLAS" deriving (Eq, Show) #}

-- | Give it something to do with a CUBLAS context, and it takes care
--   of initialising and cleaning up, e.g:
--  
--   >>> withCublas version
-- 
--   /Note/: it also calls 'setPointerMode' to set the PM to 'PointerModeDevice'
--           otherwise, we can't pass scalars as 0-dimensional arrays to the FFI
--           see <https://groups.google.com/forum/#!topic/accelerate-haskell/4-HO34MsA8g>
withCublas :: (Handle -> IO a) -> IO a
withCublas act = bracket create' destroy act
  where create' = create >>= (\h -> setPointerMode h PointerModeDevice >> return h)

-- | Same as 'withCublas' but lets you specify the 'PointerMode' you want to use
withCublasUsing :: PointerMode -> (Handle -> IO a) -> IO a
withCublasUsing pm act = bracket create' destroy act
  where create' = create >>= (\h -> setPointerMode h pm >> return h)

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

-- | Get the 'CS.Stream' used by your CUBLAS context
--   
--   For more details about 'CS.Stream', see the cuda package, 
--   in the "Foreign.CUDA.Runtime.Stream" module.
getStream :: Handle -> IO CS.Stream
getStream h = statusIfOk =<< cublasGetStream h

-- | Set the 'CS.Stream' used by your CUBLAS context
--   
--   For more details about 'CS.Stream', see the cuda package, 
--   in the "Foreign.CUDA.Runtime.Stream" module.
setStream :: Handle -> CS.Stream -> IO ()
setStream h s = nothingIfOk =<< cublasSetStream h s

{# fun unsafe cublasSetStream_v2 as cublasSetStream 
  { useHandle    `Handle'
  , useStream `CS.Stream' } -> `Status' cToEnum #}
  where useStream = CS.useStream

{# fun unsafe cublasGetStream_v2 as cublasGetStream
  { useHandle `Handle'
  , alloca-   `CS.Stream' peekStream* } -> `Status' cToEnum #}
  where peekStream = liftM CS.Stream . peek

-- | 'AtomicsMode' type.
-- 
--   This lets you control whether CUBLAS routines which have 
--   an alternate implementation using atomics can be used.
--  
--   See <http://docs.nvidia.com/cuda/cublas/#cublasatomicsmode_t>
{# enum cublasAtomicsMode_t as AtomicsMode
  { underscoreToCase }
  with prefix="CUBLAS" deriving (Eq, Show) #}

-- | Set the 'AtomicsMode' for your CUBLAS context
setAtomicsMode :: Handle -> AtomicsMode -> IO ()
setAtomicsMode h m = nothingIfOk =<< cublasSetAtomicsMode h m

-- | Get the 'AtomicsMode' of your CUBLAS context
getAtomicsMode :: Handle -> IO AtomicsMode
getAtomicsMode h = statusIfOk =<< cublasGetAtomicsMode h

{# fun unsafe cublasSetAtomicsMode as cublasSetAtomicsMode
  { useHandle `Handle'
  , cFromEnum `AtomicsMode' } -> `Status' cToEnum #}

{# fun unsafe cublasGetAtomicsMode as cublasGetAtomicsMode
  { useHandle `Handle'
  , alloca-   `AtomicsMode' peekAM* } -> `Status' cToEnum #}
  where peekAM = liftM cToEnum . peek
