module Foreign.CUDA.BLAS
  ( -- * Level 1 functions
    sdot
  , ddot
  , cublasSdot
  , cublasDdot
  , -- * Helper functions
    create
  , destroy
  , version
  ) where

import Foreign.CUDA.BLAS.Helper
import Foreign.CUDA.BLAS.Level1
import Foreign.CUDA.BLAS.Error