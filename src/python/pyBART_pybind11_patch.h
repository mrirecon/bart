#define MYPYBIND11_PLUGIN_IMPL(name)				\
     extern "C" PYBIND11_EXPORT PyObject *PyInit_##name()
#define MYPYBIND11_CHECK_PYTHON_VERSION					\
     {									\
	  const char *compiled_ver = PYBIND11_TOSTRING(PY_MAJOR_VERSION) \
	       "." PYBIND11_TOSTRING(PY_MINOR_VERSION);			\
	  const char *runtime_ver = Py_GetVersion();			\
	  size_t len = std::strlen(compiled_ver);			\
	  if (std::strncmp(runtime_ver, compiled_ver, len) != 0		\
	      || (runtime_ver[len] >= '0' && runtime_ver[len] <= '9')) { \
	       PyErr_Format(PyExc_ImportError,				\
			    "Python version mismatch: module was compiled for Python %s, " \
			    "but the interpreter version is incompatible: %s.",	\
			    compiled_ver, runtime_ver);			\
	       return nullptr;						\
	  }								\
     }

#define MYPYBIND11_CATCH_INIT_EXCEPTIONS		\
     catch (pybind11::error_already_set &e) {		\
	  PyErr_SetString(PyExc_ImportError, e.what());	\
	  return nullptr;				\
     } catch (const std::exception &e) {		\
	  PyErr_SetString(PyExc_ImportError, e.what());	\
	  return nullptr;				\
     }							\

#define MYPYBIND11_MODULE(name, variable)				\
     static void PYBIND11_CONCAT(pybind11_init_, name)(pybind11::mymodule &); \
     MYPYBIND11_PLUGIN_IMPL(name) {					\
	  MYPYBIND11_CHECK_PYTHON_VERSION;				\
	  auto m = pybind11::mymodule(PYBIND11_TOSTRING(name));		\
	  try {								\
	       PYBIND11_CONCAT(pybind11_init_, name)(m);		\
	       return m.ptr();						\
	  } MYPYBIND11_CATCH_INIT_EXCEPTIONS;				\
     }									\
     void PYBIND11_CONCAT(pybind11_init_, name)(pybind11::mymodule &variable)

#if PY_MAJOR_VERSION < 3
class A {}; // dummy class
#define MYPYBIND11_MODULE_DESTRUCTOR_IMPLEMENT				\
     py::class_<A>(m, "BaseClass");					\
     py::cpp_function cleanup_callback(					\
	  [](py::handle weakref) {					\
	       cleanup_memory();					\
	       weakref.dec_ref();					\
	  }								\
	  );								\
     (void) py::weakref(m.attr("BaseClass"), cleanup_callback).release()
#else
#  define MYPYBIND11_MODULE_DESTRUCTOR_IMPLEMENT
static int pyBART_clear(PyObject *m);
static int pyBART_traverse(PyObject *m, visitproc visit, void *arg);
#endif /* PY_MAJOR_VERSION < 3 */
     
     
NAMESPACE_BEGIN(PYBIND11_NAMESPACE)


/// Wrapper for Python extension modules
class mymodule : public module
{
public:
     PYBIND11_OBJECT_DEFAULT(mymodule, module, PyModule_Check)

     /// Create a new top-level Python module with the given name and docstring
     explicit mymodule(const char *name, const char *doc = nullptr) {
	  if (!options::show_user_defined_docstrings()) doc = nullptr;
#if PY_MAJOR_VERSION >= 3
	  PyModuleDef *def = new PyModuleDef();
	  std::memset(def, 0, sizeof(PyModuleDef));
	  def->m_name = name;
	  def->m_doc = doc;
	  def->m_size = -1;
	  def->m_traverse = pyBART_traverse;
	  def->m_clear = pyBART_clear;
	  Py_INCREF(def);
	  m_ptr = PyModule_Create(def);
#else
	  m_ptr = Py_InitModule3(name, nullptr, doc);
#endif
	  if (m_ptr == nullptr)
	       pybind11_fail("Internal error in mymodule::mymodule()");
	  inc_ref();
     }
};

NAMESPACE_END(PYBIND11_NAMESPACE)
