module Main where

import Foreign.CUDA.BLAS.Helper
import Foreign.C

main :: IO ()
main = do
  h  <- create
  i  <- version h
  pm <- getPointerMode h
  s  <- getStream h
  am <- getAtomicsMode h
  
  putStrLn $ "CUBLAS version: " ++ show i
  putStrLn $ "CUBLAS stream: " ++ show s
  putStrLn $ "CUBLAS atomics mode: " ++ show am
  putStrLn $ "Cublas pointer mode: " ++ show pm

  putStrLn "[...changing pointer mode to 'on device'...]"
  setPointerMode h PointerModeDevice
  pm' <- getPointerMode h
  putStrLn $ "Cublas pointer mode: " ++ show pm'
  
  putStrLn "[...changing atomics mode to 'allowed'...]"
  setAtomicsMode h AtomicsAllowed
  am' <- getAtomicsMode h
  putStrLn $ "Cublas atomics mode: " ++ show am'

  destroy h