module Main where

import Foreign.CUDA.BLAS
import Foreign.C

main :: IO ()
main = do
  h <- create
  i <- version h
  putStrLn $ "CUBLAS version: " ++ show i
  destroy h