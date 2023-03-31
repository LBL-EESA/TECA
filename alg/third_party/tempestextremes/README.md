## TELite

TELite library only includes the TempestExtremes sources related to the DetectNodes function.
Some modifications/simplifications were introduced in the original codes for porting DetectNodes to TECA.

### Compiling TELite

~~~~~~~~
   mkdir build
   cd build/
   cmake .. -DCMAKE_INSTALL_PREFIX=<install dir>
   make
   make install
~~~~~~~

