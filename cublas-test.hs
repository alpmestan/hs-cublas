module Main where

import Foreign.CUDA.BLAS.Helper
import Foreign.C

main :: IO ()
main = do
  h  <- create
  i  <- version h
  pm <- getPointerMode h
  putStrLn $ "CUBLAS version: " ++ show i
  putStrLn $ "Cublas pointer mode: " ++ show pm
  putStrLn "[...changing pointer mode to 'on device'...]"
  setPointerMode h PointerModeDevice
  pm' <- getPointerMode h
  putStrLn $ "Cublas pointer mode: " ++ show pm'
  destroy h