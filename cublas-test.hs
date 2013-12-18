module Main where

import Foreign.CUDA.BLAS
import Foreign.C

main :: IO ()
main = do
  h <- cublasCreate
  i <- cublasVersion h
  putStrLn $ "CUBLAS version: " ++ show i
  cublasDestroy h