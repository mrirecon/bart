#ifndef BART_API_H_INCLUDED
#define BART_API_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif
     //! BART's current debug level
     extern int debug_level;

     //! Load the content of some in-memory CFL
     /*!
      *  This function will load the data from some named in-memory CFL and returns 
      *  its data.
      *  The dimensions array will get modified to match those from the CFL
      *
      *  \param name       Name used to refer to in-memory CFL
      *  \param D          Size of the dimensions array (should be < 16)
      *  \param dimensions Array holding the dimensions of the data
      *                    (will get modified)
      *
      *  \return Pointer to the data or NULL if no matching in-memory CFL file
      *  was found
      */
     void* load_mem_cfl(const char* name, unsigned int D, long dimensions[]);
     
     //! Register some memory into the list of in-memory CFL files
     /*! 
      *  This function handles data that was allocated using the C malloc(...) function.
      *  It takes *ownership* of the data and will free it using free(...)
      *
      *  \param name       Name which will be used to refer to the created in-mem CFL
      *  \param D          Size of the dimensions array (should be < 16)
      *  \param dimensions Array holding the dimensions of the data
      *  \param ptr        Pointer to the data
      *
      *  \note The underlying data type of ptr is assumed to be complex floats
      *  (complex float or _Complex float)
      *
      *  \warning Calling this function on data allocated with new[] will result
      *  in undefined behaviour!
      *
      *  \warning Be aware that if MEMONLY_CFL is not defined, names that do not
      *  end with the '.mem' extension will be unreachable by user code
      */
     void register_mem_cfl_malloc(const char* name, unsigned int D, const long dimensions[], void* ptr);

     //! Register some memory into the list of in-memory CFL files
     /*! 
      *  This function handles data that was allocated using the C++ new[] operator
      *  It takes *ownership* of the data and will free it using delete[]
      *
      *  \param name       Name which will be used to refer to the created in-mem CFL
      *  \param D          Size of the dimensions array (should be < 16)
      *  \param dimensions Array holding the dimensions of the data
      *  \param ptr        Pointer to the data
      *
      *  \note The underlying data type of ptr is assumed to be complex floats
      *  (complex float or _Complex float)
      *
      *  \warning Calling this function on data allocated with malloc will
      *  result in undefined behaviour!
      *
      *  \warning Be aware that if MEMONLY_CFL is not defined, names that do not
      *  end with the '.mem' extension will be unreachable by user code
      */
     void register_mem_cfl_new(const char* name, unsigned int D, const long dimensions[], void* ptr);

     //! Register some memory into the list of in-memory CFL files
     /*! 
      *  This function handles data that was allocated by the user and of which
      *  the user wishes to retain control of its lifetime.
      *  It does *not* takes ownership of the data
      *
      *  \param name       Name which will be used to refer to the created in-mem CFL
      *  \param D          Size of the dimensions array (should be < 16)
      *  \param dimensions Array holding the dimensions of the data
      *  \param ptr        Pointer to the data
      *
      *  \note The underlying data type of ptr is assumed to be complex floats
      *  (complex float or _Complex float)
      *
      *  \warning Be aware that if MEMONLY_CFL is not defined, names that do not
      *  end with the '.mem' extension will be unreachable by user code
      */
     void register_mem_cfl_non_managed(const char* name, unsigned int D, const long dims[], void* ptr);
     
     //! BART's main function
     /*!
      *  This function will execute the BART command specified in argv[0]
      *  
      *  If applicable, the output of the BART command will be returned into
      *  out. This applies to:
      *    - bitmask
      *    - estdims
      *    - estvar
      *    - nrmse
      *    - sdot
      *    - show
      *    - version
      *
      *  If out is not NULL, outputs of the above commands are redirected to out
      *
      *  \param len  Size of the out buffer
      *  \param out  Should be either NULL or point to a valid array of characters
      *  \param argc Same as for the main function
      *  \param argv Same as for the main function
      *
      *  \warning Be aware that if MEMONLY_CFL is not defined, names that do not
      *  end with the '.mem' extension will be unreachable by user code
      */
     int bart_command(int len, char* out, int argc, char* argv[]);
     
     //! Deallocate any memory CFLs
     /*!
      * \note It is safe to call this function multiple times.
      */
     void deallocate_all_mem_cfl();
#ifdef __cplusplus
}
#endif

#endif //BART_API_H_INCLUDED
