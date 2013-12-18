{-# LANGUAGE CPP #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE ForeignFunctionInterface #-}

module Foreign.CUDA.BLAS.Internal.Error 
  ( Status (..)
  , describe
  , CUBLASException (..)
  , cublasError
  , statusIfOk
  , nothingIfOk
  ) where

import Control.Exception
import Data.Typeable

#include <cublas_wrap.h>
{# context lib="cublas" #}

{# enum cublasStatus_t as Status
  { underscoreToCase }
  with prefix="CUBLAS" deriving (Eq, Show) #}

describe :: Status -> String
describe StatusSuccess = "the operation completed successfully"
describe StatusNotInitialized = "the cubas library was not initialized"
describe StatusAllocFailed = "resource allocation failed"
describe StatusInvalidValue = "an unsupported/invalid value was passed to a function (e.g negative vector size)"
describe StatusArchMismatch = "the function requires a feature absent from your architecture"
describe StatusMappingError = "an access to GPU memory space failed"
describe StatusExecutionFailed = "the gpu program failed to execute" 
describe StatusInternalError = "an internal CUBLAS operation failed, probably because of a cudaMemcpyAsync() failure"

data CUBLASException
  = ExitCode  Status
  | UserError String
  deriving Typeable

instance Exception CUBLASException

instance Show CUBLASException where
	showsPrec _ (ExitCode s)  = showString ("CUBLAS Exception: " ++ describe s)
	showsPrec _ (UserError e) = showString ("CUBLAS Exception: " ++ e)

-- | Throw a CUBLAS exception in IO, using 'throwIO'
cublasError :: String -> IO a
cublasError s = throwIO (UserError s)

-- | Return the results of a function on successful execution, otherwise throw
--   an exception with an error string associated with the return code
--
statusIfOk :: (Status, a) -> IO a
statusIfOk (status, res) = case status of
	StatusSuccess -> return res
	_             -> throwIO (ExitCode status)

-- | Do nothing if successful, throw the appropriate exception otherwise
nothingIfOk :: Status -> IO ()
nothingIfOk StatusSuccess = return ()
nothingIfOk s       = throwIO (ExitCode s)