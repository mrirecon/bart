/* Copyright 2017-2018. Damien Nguyen.
 * Copyright 2017-2018. Francesco Santini
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2017-2018 Nguyen Damien <damien.nguyen@alumni.epfl.ch>
 * 2017-2018 Francesco Santini <francesco.santini@unibas.ch>
 */

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif /* !_GNU_SOURCE */

#ifdef BART_WITH_PYTHON
#  include <Python.h>
#  define PY_ARRAY_UNIQUE_SYMBOL bart_numpy_identifier
#  define NO_IMPORT_ARRAY
#  define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#  include <numpy/arrayobject.h>
#endif /* BART_WITH_PYTHON */

#ifndef SUPER_DEBUG_OUT
#define SUPER_DEBUG_OUT(...)	((void)0)
#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdbool.h>

#include <algorithm>
#include <string>
#include <vector>

#include "num/multind.h"

#include "misc/mri.h"
#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/io.h"

#include "mmiocc.hh"

#define DIMS_MAX 32U // instead of 16 because of squeeze.c

typedef _Complex float cx_float;

// ========================================================================

#ifdef BART_WITH_PYTHON
extern void* load_mem_cfl_python(const char* name, unsigned int D, long dimensions[]);
#endif /* BART_WITH_PYTHON */

// ========================================================================

namespace internal_ {
     enum DATA_DIRECTION {
	  INPUT,
	  OUTPUT
     };
     
     struct Node
     {
	  Node(const std::string& name)
	       : name_(name)
	       , dirty_(name.empty() ? true : false)
	       {}
	  
	  virtual ~Node() {}

	  virtual void reset() {
	       name_.clear();
	       dirty_ = false;
	  }
	  virtual std::string name() const { return name_; }
	  virtual const long* dims() { return dims_; }
	  virtual void* data() const = 0;

	  virtual DATA_DIRECTION& data_dir() { return direction_; }
	  virtual bool& dirty() { return dirty_; }

	  virtual void clear_flags()
	       {
		    dirty_ = false;
		    direction_ = OUTPUT;
	       }
	  
	  std::string name_;
	  bool dirty_;
	  DATA_DIRECTION direction_;
	  long dims_[DIMS_MAX];
     };

     // ------------------------------------------------------------------------

     struct cpp_deleter_t
     {
	  template <typename T>
	  static void deallocate(T* data) { delete[] data; }
     };

     struct c_deleter_t
     {
	  template <typename T>
	  static void deallocate(T* data) { xfree(data); }
     };
     
     struct noop_deleter_t
     {
	  template <typename T>
	  static void deallocate(T*) {}
     };
     
     // ------------------------------------------------------------------------

     template <typename T, typename deleter_t = cpp_deleter_t>
     struct PointerNode : public Node
     {
	  PointerNode(const std::string& name, unsigned int D, long dims[])
	       : Node(name)
	       , ptr_(new cx_float[md_calc_size(D, dims)])
	       {
		    std::fill(dims_, dims_+DIMS_MAX, 1);
		    std::copy(dims, dims+D, dims_);
	       }

	  PointerNode(const std::string& name,
		      unsigned int D,
		      long dims[],
		      T* ptr)
	       : Node(name)
	       , ptr_(ptr)
	       {
		    std::fill(dims_, dims_+DIMS_MAX, 1);
		    std::copy(dims, dims+D, dims_);
	       }

	  virtual ~PointerNode()
	       {
		    SUPER_DEBUG_OUT("in: PointerNode::~PointerNode()");
		    SUPER_DEBUG_OUT("     deleting %s node",
				    name_.empty() ? "anonymous" : ("\"" + name_ + "\"").c_str());
	  	    reset();
	       }

	  virtual void reset()
	       {
		    Node::reset();
		    deleter_t::deallocate(ptr_);
		    ptr_ = NULL;
		    std::fill(dims_, dims_+DIMS_MAX, 1);
	       }

	  virtual void* data() const { return ptr_; }

	  T* ptr_;
     };

     // ------------------------------------------------------------------------

#ifdef BART_WITH_PYTHON
     struct PyPointerNode : public Node
     {
	  PyPointerNode(const std::string& name,
			PyArrayObject* ptr)
	       : Node(name)
	       , ptr_(ptr)
	       {}

	  virtual ~PyPointerNode()
	       {
		    SUPER_DEBUG_OUT("in: PyPointerNode::~PyPointerNode()");
		    SUPER_DEBUG_OUT("     deleting %s node",
				    name_.empty() ? "anonymous" : ("\"" + name_ + "\"").c_str());
	  	    reset();
	       }

	  virtual void reset()
	       {
		    Node::reset();
 		    // PyArray_XDECREF(ptr_); // FIXME: this should really be uncommented... but right now it segfaults with python3
		    ptr_ = NULL;
		    std::fill(dims_, dims_+DIMS_MAX, 1);
		    dirty_ = false;
	       }

	  virtual const long* dims()
	       {
		    SUPER_DEBUG_OUT("PyPointerNode::dims()");
		    unsigned int D(PyArray_NDIM(ptr_));
		    npy_intp* npy_dims(PyArray_SHAPE(ptr_));
		    std::fill(dims_, dims_+DIMS_MAX, 1);
		    std::copy(npy_dims, npy_dims +  + std::min(D, DIMS_MAX), dims_);
		    return dims_;
	       }
	  virtual void* data() const { return PyArray_DATA(ptr_); }

	  PyArrayObject* ptr_;
     };
#endif /* BART_WITH_PYTHON */

     // ------------------------------------------------------------------------

     void call_delete(Node* node)
     {
	  delete node;
     }
     
     class NameEqual
     {
     public:
	  NameEqual(const std::string& name)
	       : name_(name)
	       {}
	  virtual ~NameEqual() {}

	  bool operator() (const Node* node) const
	       {
		    return name_ == node->name();
	       }
     private:
	  std::string name_;
     };

     class PtrDataEqual
     {
     public:
	  PtrDataEqual(const void* ptr)
	       : ptr_(ptr)
	       {}
	  virtual ~PtrDataEqual() {}

	  bool operator() (const Node* node) const
	       {
		    return ptr_ != NULL && ptr_ == node->data();
	       }
     private:
	  const void* ptr_;
     };
     
     // ========================================================================

     class MemoryHandler
     {
	  typedef std::vector<Node*> vector_t;
     
     public:
	  MemoryHandler() {}

	  ~MemoryHandler() { clear(); }

	  template <typename T>
	  T* allocate_mem_cfl(const std::string& name, unsigned int D, long dims[])
	       {
		    SUPER_DEBUG_OUT("in: MemoryHandler::allocate_mem_cfl<T>(\"%s\", %d, ...)", name.c_str(), D);
		    
		    vector_t::iterator it(std::find_if(list_.begin(),
						       list_.end(),
						       NameEqual(name)));
		    if (it != list_.end()) {
			 SUPER_DEBUG_OUT("     found \"%s\" already in the database!", name.c_str());
			 
			 /* We are attempting to create a new memory CFL
			  * but found that another one with the same name
			  * already exists.
			  * Now either:
			  *   - the mem CFL has previously been marked as input,
			  *     in which case we mark the existing one as dirty
			  *   - it is still marked as output, in which case we
			  *     delete the existing one
			  *     
			  */
			 if ((*it)->data_dir() == INPUT) {
			      BART_OUT("MEMCFL: marking first occurrence of %s as DIRTY!\n", (*it)->name().c_str());
			      (*it)->dirty() = true;
			 }
			 else {
			      BART_OUT("MEMCFL: deleting first occurrence of %s\n", (*it)->name().c_str());
			      remove_node_(it);
			 }
		    }
		    
		    SUPER_DEBUG_OUT("     allocating PointerNode<T>");
		    list_.push_back(new PointerNode<T>(name, D, dims));
		    SUPER_DEBUG_OUT("     returning from MemoryHandler::allocate_mem_cfl<T>");
		    return reinterpret_cast<T*>(list_.back()->data());
	       }
	  template <typename T>
	  T* allocate_mem_cfl(unsigned int D, long dims[])
	       {
		    SUPER_DEBUG_OUT("in: MemoryHandler::allocate_mem_cfl<T>(%d, ...)", D);
		    SUPER_DEBUG_OUT("     allocating PointerNode<T>");
		    list_.push_back(new PointerNode<T>("", D, dims));
		    SUPER_DEBUG_OUT("     returning from MemoryHandler::allocate_mem_cfl<T>");
		    return reinterpret_cast<T*>(list_.back()->data());
	       }

	  template <typename T, typename deleter_t>
	  void register_mem_cfl(const std::string& name,
				unsigned int D,
				long dims[],
				T* ptr,
				deleter_t)
	       {
		    SUPER_DEBUG_OUT("in: MemoryHandler::register_mem_cfl<T>(\"%s\", ...)", name.c_str());
		    
		    vector_t::iterator it(std::find_if(list_.begin(),
						       list_.end(),
						       NameEqual(name)));
		    if (it != list_.end()) {
			 SUPER_DEBUG_OUT("In-mem CFL: found existing data with the same name, deleting old data");
			 remove_node_(it);
		    }

		    it = std::find_if(list_.begin(),
				      list_.end(),
				      PtrDataEqual(ptr));
		    if (it != list_.end()) {
			 error("In-mem CFL: attempting to register ptr for %s, "
			       "but ptr has alread been registered for %s!\n",
			       name.c_str(),
			       (*it)->name().c_str());
		    }

		    // Need to call io_register_input here since no calls to
		    // either create_cfl() or load_cfl() lead to here...
		    io_register_input(name.c_str());
		    list_.push_back(new PointerNode<T, deleter_t>(name, D, dims, ptr));
		    list_.back()->data_dir() = INPUT;
	       }

#ifdef BART_WITH_PYTHON
	  void register_mem_cfl(const std::string& name, PyArrayObject* npy_data)
	       {
		    SUPER_DEBUG_OUT("in: MemoryHandler::register_mem_cfl(\"%s\", npy_data)", name.c_str());
		    
		    vector_t::iterator it(std::find_if(list_.begin(),
						       list_.end(),
						       NameEqual(name)));
		    if (it != list_.end()) {
			 SUPER_DEBUG_OUT("In-mem CFL: found existing data with the same name, deleting old data");
			 remove_node_(it);
		    }

		    it = std::find_if(list_.begin(),
				      list_.end(),
				      PtrDataEqual(npy_data));
		    if (it != list_.end()) {
			 error("In-mem CFL: attempting to register ptr for %s, "
			       "but ptr has alread been registered for %s!\n",
			       name.c_str(),
			       (*it)->name().c_str());
		    }

		    // Need to call io_register_input here since no calls to
		    // either create_cfl() or load_cfl() lead to here...
		    io_register_input(name.c_str());
		    list_.push_back(new PyPointerNode(name, npy_data));
		    list_.back()->data_dir() = INPUT;
	       }
#endif /* BART_WITH_PYTHON */

	  cx_float* load_mem_cfl(const std::string& name,
				 unsigned int D,
				 long dims[])
	       {
		    SUPER_DEBUG_OUT("in: MemoryHandler::load_mem_cfl<T> (\"%s\", ...)", name.c_str());
		    
		    vector_t::iterator it(std::find_if(list_.begin(),
						       list_.end(),
						       NameEqual(name)));
		    if (it != list_.end()) {
			 SUPER_DEBUG_OUT("     found it! copying dimensions");

			 const long* d = (*it)->dims();
			 std::copy(d, d + std::min(D, DIMS_MAX), dims);
			 SUPER_DEBUG_OUT("     marking it as input");
			 (*it)->data_dir() = INPUT;

			 SUPER_DEBUG_OUT("     returning from MemoryHandler::load_mem_cfl<T>");
			 return reinterpret_cast<cx_float*>((*it)->data());
		    }
		    else {
			 return NULL;
		    }
	       }

	  template <typename T>
	  bool is_mem_cfl(T* ptr)
	       {
		    return std::find_if(list_.begin(),
					list_.end(),
					PtrDataEqual(ptr)) != list_.end();
	       }

	  template <typename T>
	  bool try_delete_mem_cfl(T* ptr)
	       {
		    SUPER_DEBUG_OUT("in: MemoryHandler::try_delete_mem_cfl<T>");
		    
		    vector_t::iterator it(std::find_if(list_.begin(),
						       list_.end(),
						       PtrDataEqual(ptr)));
		    if (it != list_.end()) {
			 SUPER_DEBUG_OUT("     found data (%s)!", (*it)->name().empty() ? "anonymous" : ("\"" + (*it)->name() + "\"").c_str());
			 
			 if ((*it)->dirty()) {
			      SUPER_DEBUG_OUT("     node is dirty, deallocating!");
			      remove_node_(it);
			 }
			 else {
			      SUPER_DEBUG_OUT("     node is ok, calling io_unregister(...) and clear_flags(...)");
			      io_unregister((*it)->name().c_str());
			      (*it)->clear_flags();
			 }
			 return true;
		    }
		    SUPER_DEBUG_OUT("     data *not* found!");
		    return false;
	       }
     
	  // Attempt to deallocate memory based on address or name
	  template <typename T>
	  bool remove(T* ptr)
	       {
		    return remove_(PtrDataEqual(ptr));
	       }
	  bool remove(const std::string& name)
	       {
		    return remove_(NameEqual(name));
	       }

	  void clear()
	       {
		    std::for_each(list_.begin(), list_.end(), call_delete);
		    list_.clear();
	       }

     private:
	  void remove_node_(vector_t::iterator it)
	       {
		    delete (*it);
		    *it = NULL;
		    list_.erase(it);
	       }
	  
	  // Return true if deletion occurred, false otherwise
	  template <typename unop_t>
	  bool remove_(unop_t op) 
	       {
		    vector_t::iterator it = std::remove_if(list_.begin(),
							   list_.end(),
							   op);
		    if (it != list_.end()) {
			 std::for_each(it, list_.end(), call_delete);
			 list_.erase(it, list_.end());
			 return true;
		    }
		    return false;
	       }

     
	  vector_t list_;

	  // Not implemented!
	  MemoryHandler(const MemoryHandler&);
	  MemoryHandler& operator=(const MemoryHandler&);
     };
} // namespace internal_

typedef internal_::c_deleter_t c_delete;
typedef internal_::cpp_deleter_t cpp_delete;
typedef internal_::noop_deleter_t noop_delete;

static internal_::MemoryHandler mem_handler;

// =============================================================================

void* create_mem_cfl(const char* name, unsigned int D, const long dims[])
{
     SUPER_DEBUG_OUT("in: create_mem_cfl");
     
     if (D > DIMS_MAX) {
	  BART_WARN("create_mem_cfl: D > DIMS_MAX: %d > %d!\n", D, DIMS_MAX);
     }
     
     return mem_handler.allocate_mem_cfl<cx_float>(name,
						   D,
						   const_cast<long*>(dims));
}

void* create_anon_mem_cfl(unsigned int D, const long dims[])
{
     SUPER_DEBUG_OUT("in: create_anon_mem_cfl");
     
     if (D > DIMS_MAX) {
	  BART_WARN("create_anon_mem_cfl: D > DIMS_MAX: %d > %d!\n", D, DIMS_MAX);
     }
     
     return mem_handler.allocate_mem_cfl<cx_float>(D, const_cast<long*>(dims));
}

void* load_mem_cfl(const char* name, unsigned int D, long dims[])
{
     SUPER_DEBUG_OUT("in: load_mem_cfl");

     if (D > DIMS_MAX) {
	  BART_WARN("load_mem_cfl: D > DIMS_MAX: %d > %d!\n", D, DIMS_MAX);
     }
     
     cx_float* ret = mem_handler.load_mem_cfl(name, D, dims);

     if (ret == NULL) {
#ifndef BART_WITH_PYTHON
	  BART_ERR("failed loading memory cfl file \"%s\"", name);
#else
	  PyErr_Format(PyExc_RuntimeError, "failed loading memory cfl file \"%s\"", name);
#endif /* !BART_WITH_PYTHON */
     }
     return ret;
}

template <typename deleter_t>
void register_mem_cfl_impl(const char* name,
			   unsigned int D,
			   const long dims[],
			   void* data)
{
     if (D > DIMS_MAX) {
	  BART_WARN("register_mem_cfl_impl: D > DIMS_MAX: %d > %d!\n", D, DIMS_MAX);
     }

     /* NB: due to the use of noop_delete, the memory will not be freed upon
      *     destruction
      */
     mem_handler.register_mem_cfl(name,
				  D,
				  const_cast<long*>(dims),
				  reinterpret_cast<cx_float*>(data),
				  deleter_t());
}

void register_mem_cfl_non_managed(const char* name,
				  unsigned int D,
				  const long dims[],
				  void* data)
     
{
     SUPER_DEBUG_OUT("in: register_mem_cfl_non_managed");

     register_mem_cfl_impl<noop_delete>(name, D, dims, data);
}

void register_mem_cfl_malloc(const char* name,
			     unsigned int D,
			     const long dims[],
			     void* data)
{
     SUPER_DEBUG_OUT("in: register_mem_cfl_malloc");
     
     register_mem_cfl_impl<c_delete>(name, D, dims, data);
}
void register_mem_cfl_new(const char* name,
			  unsigned int D,
			  const long dims[],
			  void* data)
{
     SUPER_DEBUG_OUT("in: register_mem_cfl_new");
     
     register_mem_cfl_impl<cpp_delete>(name, D, dims, data);
}

_Bool is_mem_cfl(const cx_float* ptr)
{
     return mem_handler.is_mem_cfl(ptr);
}

_Bool try_delete_mem_cfl(const _Complex float* ptr)
{
     return mem_handler.try_delete_mem_cfl(ptr);
}

_Bool deallocate_mem_cfl_name(const char* name)
{
     return mem_handler.remove(name);
}

_Bool deallocate_mem_cfl_ptr(const cx_float* ptr)
{
     return mem_handler.remove(ptr);
}

void deallocate_all_mem_cfl()
{
     SUPER_DEBUG_OUT("in: deallocate_all_mem_cfl");
     mem_handler.clear();
}

// =============================================================================

#ifdef BART_WITH_PYTHON
extern "C" _Bool register_mem_cfl_python(const char* name, PyArrayObject* npy_data)
{
     SUPER_DEBUG_OUT("in: register_mem_cfl_python");
     unsigned int D(PyArray_NDIM(npy_data));
     if (D > DIMS_MAX) {
	  BART_WARN("register_mem_cfl_python: D > DIMS_MAX: %d > %d!\n", D, DIMS_MAX);
     }

     if (PyArray_INCREF(npy_data)) {
	  PyErr_SetString(PyExc_RuntimeError,
			  "failed to increase reference count of npy_data");
	  return false;
     }
     
     mem_handler.register_mem_cfl(name, npy_data);
     return true;
}
#endif /* BART_WITH_PYTHON */
