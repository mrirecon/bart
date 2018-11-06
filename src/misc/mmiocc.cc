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
#  include "pybind11/pybind11.h"
#  include "pybind11/numpy.h"

namespace py = pybind11;
using numpy_farray_t = py::array_t<std::complex<float>,
				   py::array::f_style
				   | py::array::forcecast>;
using numpy_array_t = py::array_t<std::complex<float>,
				  py::array::c_style
				  | py::array::forcecast>;
#endif /* BART_WITH_PYTHON */

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdbool.h>

#include <algorithm>
#include <string>
#include <vector>
#include <stdexcept>
#include <memory>
#include <type_traits>

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
     
     class Node
     {
     public:
	  Node(const std::string& name)
	       : name_(name)
	       , dirty_(name.empty() ? true : false)
	       , direction_(OUTPUT)
	       {
		    std::fill(dims_, dims_+DIMS_MAX, 1);
	       }

	  Node(const Node&) = delete;
	  Node(Node&&) = delete;
	  Node& operator=(const Node&) = delete;
	  Node& operator=(Node&&) = delete;
	  
	  virtual ~Node() {}

	  virtual std::string name() const { return name_; }
	  virtual const long* dims() { return dims_; }
	  virtual void* data() = 0;
	  virtual const void* data() const = 0;

	  virtual DATA_DIRECTION& data_dir() { return direction_; }
	  virtual bool& dirty() { return dirty_; }

	  virtual void clear_flags()
	       {
		    dirty_ = false;
		    direction_ = OUTPUT;
	       }

     protected:
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
     class PointerNode : public Node
     {
     public:
	  PointerNode(const std::string& name, unsigned int D, long dims[])
	       : PointerNode(name, D, dims, new cx_float[md_calc_size(D, dims)])
	       {}

	  PointerNode(const std::string& name,
		      unsigned int D,
		      long dims[],
		      T* ptr)
	       : Node(name)
	       , ptr_(ptr)
	       {
		    std::copy(dims, dims+D, dims_);
	       }

	  virtual ~PointerNode()
	       {
		    debug_printf(DP_DEBUG2, "in: PointerNode::~PointerNode()\n");
		    debug_printf(DP_DEBUG2, "     deleting %s node\n",
				 name_.empty() ? "anonymous" : ("\"" + name_ + "\"").c_str());
		    deleter_t::deallocate(ptr_);
		    ptr_ = nullptr;
	       }

	  virtual void* data() override { return ptr_; }
	  virtual const void* data() const override { return ptr_; }

     private:
	  T* ptr_;
     };

     // ------------------------------------------------------------------------

#ifdef BART_WITH_PYTHON
// https://stackoverflow.com/questions/2828738/c-warning-declared-with-greater-visibility-than-the-type-of-its-field
#pragma GCC visibility push(hidden)
     class PyPointerNode : public Node
     {
     public:
	  PyPointerNode(const std::string& name,
			const numpy_farray_t& array)
	       : Node(name)
	       , array_(array)
	       {}

	  virtual ~PyPointerNode()
	       {
		    debug_printf(DP_DEBUG2, "in: PyPointerNode::~PyPointerNode()\n");
		    debug_printf(DP_DEBUG2, "     deleting %s node\n",
				 name_.empty() ? "anonymous" : ("\"" + name_ + "\"").c_str());
	       }

	  virtual const long* dims() override
	       {
	  	    const auto* npy_dims(array_.shape());
	  	    debug_printf(DP_DEBUG2, "     reading dimensions from Python object\n");
	  	    std::fill(dims_, dims_+DIMS_MAX, 1);
	  	    std::copy(npy_dims,
	  		      npy_dims + std::min(array_.ndim(), py::ssize_t(DIMS_MAX)),
	  		      dims_);
		    return dims_;
	       }
	  virtual void* data() override { return array_.mutable_data(); }
	  virtual const void* data() const override { return array_.data(); }

     private:
	  /*
	   * Since we copied the input array into this PyPointerNode, its
	   * reference count was automatically increased by pybind11
	   * so that even if the user deletes it, the memory stays valid
	   */
	  numpy_farray_t array_;
     };
#pragma GCC visibility pop
#endif /* BART_WITH_PYTHON */

     // ------------------------------------------------------------------------

     class NameEqual
     {
     public:
	  NameEqual(const std::string& name)
	       : name_(name)
	       {}

	  template <typename node_t>
	  auto operator() (const node_t& node) const
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

	  template <typename node_t>
	  auto operator() (const node_t& node) const
	       {
		    return (ptr_ != nullptr) && (ptr_ == node->data());
	       }
     private:
	  const void* ptr_;
     };
     
     // ========================================================================

     class MemoryHandler
     {
	  typedef std::vector<std::unique_ptr<Node>> vector_t;
     
     public:
	  MemoryHandler() = default;
	  ~MemoryHandler() = default;
	  MemoryHandler(const MemoryHandler&) = delete;
	  MemoryHandler& operator=(const MemoryHandler&) = delete;

	  template <typename T>
	  T* allocate_mem_cfl(const std::string& name, unsigned int D, long dims[])
	       {
		    debug_printf(DP_DEBUG2, "in: MemoryHandler::allocate_mem_cfl<T>(\"%s\", %d, ...)\n", name.c_str(), D);
		    
		    const auto it(std::find_if(list_.begin(),
					       list_.end(),
					       NameEqual(name)));

		    bool is_dirty(false);
		    if (it != list_.end()) {
			 debug_printf(DP_DEBUG2, "     found \"%s\" already in the database!\n", name.c_str());
			 
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
			      is_dirty = true;
			      std::iter_swap(it, list_.end()-1); // move it to the end
			 }
			 else {
			      BART_OUT("MEMCFL: deleting first occurrence of %s\n", (*it)->name().c_str());
			      remove_node_(it);
			 }
		    }
		    
		    debug_printf(DP_DEBUG2, "     allocating PointerNode<T>\n");
		    list_.emplace_back(std::make_unique<PointerNode<T>>(name, D, dims));
		    auto* data = reinterpret_cast<T*>(list_.back()->data());
		    if (is_dirty) {
		    	 /*
		    	  * Make sure the dirty node is after the one we just added
		    	  * NB: cannot use 'it' here as a re-allocation might have
		    	  *     happened...
		    	  */
		    	 std::iter_swap(list_.end()-2, list_.end()-1);
		    }
		    debug_printf(DP_DEBUG2, "     returning from MemoryHandler::allocate_mem_cfl<T>\n");
		    return data;
	       }
	  template <typename T>
	  T* allocate_mem_cfl(unsigned int D, long dims[])
	       {
		    debug_printf(DP_DEBUG2, "in: MemoryHandler::allocate_mem_cfl<T>(%d, ...)\n", D);
		    debug_printf(DP_DEBUG2, "     allocating PointerNode<T>\n");
		    list_.emplace_back(std::make_unique<PointerNode<T>>("", D, dims));
		    debug_printf(DP_DEBUG2, "     returning from MemoryHandler::allocate_mem_cfl<T>\n");
		    return reinterpret_cast<T*>(list_.back()->data());
	       }

	  template <typename T, typename deleter_t>
	  void register_mem_cfl(const std::string& name,
				unsigned int D,
				long dims[],
				T* ptr,
				deleter_t)
	       {
		    debug_printf(DP_DEBUG2, "in: MemoryHandler::register_mem_cfl<T>(\"%s\", ...)\n", name.c_str());
		    
		    auto it(std::find_if(list_.begin(),
					 list_.end(),
					 NameEqual(name)));
		    if (it != list_.end()) {
			 debug_printf(DP_DEBUG2, "In-mem CFL: found existing data with the same name, deleting old data\n");
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
		    list_.emplace_back(std::make_unique<PointerNode<T, deleter_t>>(name, D, dims, ptr));
		    list_.back()->data_dir() = INPUT;
	       }

#ifdef BART_WITH_PYTHON
	  void register_mem_cfl(const std::string& name,
				const numpy_farray_t& array)
	       {
		    debug_printf(DP_DEBUG2, "in: MemoryHandler::register_mem_cfl(\"%s\", npy_data)\n", name.c_str());
		    
		    auto it(std::find_if(list_.begin(),
					 list_.end(),
					 NameEqual(name)));
		    if (it != list_.end()) {
			 debug_printf(DP_DEBUG2, "In-mem CFL: found existing data with the same name, deleting old data\n");
			 remove_node_(it);
		    }

		    it = std::find_if(list_.begin(),
				      list_.end(),
				      PtrDataEqual(array.data()));
		    if (it != list_.end()) {
			 char buf[1024] = { '\0' };
			 snprintf(buf, 1024,
				  "In-mem CFL: attempting to register ptr for %s, "
				  "but ptr has alread been registered for %s!\n",
				  name.c_str(), (*it)->name().c_str());
			 throw std::runtime_error(buf);
		    }

		    // Need to call io_register_input here since no calls to
		    // either create_cfl() or load_cfl() lead to here...
		    io_register_input(name.c_str());
		    list_.emplace_back(std::make_unique<PyPointerNode>(name, array));
		    list_.back()->data_dir() = INPUT;
	       }
#endif /* BART_WITH_PYTHON */

	  cx_float* load_mem_cfl(const std::string& name,
				 unsigned int D,
				 long dims[])
	       {
		    debug_printf(DP_DEBUG2, "in: MemoryHandler::load_mem_cfl<T> (\"%s\", ...)\n", name.c_str());
		    
		    const auto it(std::find_if(list_.begin(),
					       list_.end(),
					       NameEqual(name)));
		    if (it != list_.end()) {
			 debug_printf(DP_DEBUG2, "     found it! copying dimensions\n");

			 auto* d = (*it)->dims();
			 std::copy(d, d + std::min(D, DIMS_MAX), dims);
			 debug_printf(DP_DEBUG2, "     marking it as input\n");
			 (*it)->data_dir() = INPUT;

			 debug_printf(DP_DEBUG2, "     returning from MemoryHandler::load_mem_cfl<T>\n");
			 return reinterpret_cast<cx_float*>((*it)->data());
		    }
		    else {
			 return nullptr;
		    }
	       }

	  template <typename T>
	  auto is_mem_cfl(T* ptr)
	       {
		    return std::find_if(list_.cbegin(),
					list_.cend(),
					PtrDataEqual(ptr)) != list_.cend();
	       }

	  template <typename T>
	  auto try_delete_mem_cfl(T* ptr)
	       {
		    debug_printf(DP_DEBUG2, "in: MemoryHandler::try_delete_mem_cfl<T>\n");
		    
		    const auto it(std::find_if(list_.begin(),
					       list_.end(),
					       PtrDataEqual(ptr)));
		    if (it != list_.end()) {
			 debug_printf(DP_DEBUG2, "     found data (%s)!\n", (*it)->name().empty() ? "anonymous" : ("\"" + (*it)->name() + "\"").c_str());
			 
			 if ((*it)->dirty()) {
			      debug_printf(DP_DEBUG2, "     node is dirty, deallocating!\n");
			      remove_node_(it);
			 }
			 else {
			      debug_printf(DP_DEBUG2, "     node is ok, calling io_unregister(...) and clear_flags(...)\n");
			      io_unregister((*it)->name().c_str());
			      (*it)->clear_flags();
			 }
			 return true;
		    }
		    debug_printf(DP_DEBUG2, "     data *not* found!\n");
		    return false;
	       }
     
	  // Attempt to deallocate memory based on address or name
	  template <typename T>
	  std::enable_if_t<
	       !std::is_same<
		    char,
		    std::remove_reference_t<std::remove_cv_t<T>>>::value,
	       bool>
	  remove(T* ptr)
	       {
		    return remove_(PtrDataEqual(ptr));
	       }
	  bool remove(const std::string& name)
	       {
		    return remove_(NameEqual(name));
	       }

	  void clear()
	       {
		    list_.clear();
	       }

     private:
	  void remove_node_(vector_t::iterator it)
	       {
		    list_.erase(it);
	       }
	  
	  // Return true if deletion occurred, false otherwise
	  template <typename unop_t>
	  bool remove_(unop_t op) 
	       {
		    auto it = std::remove_if(list_.begin(), list_.end(), op);
		    if (it != list_.end()) {
			 list_.erase(it, list_.end());
			 return true;
		    }
		    return false;
	       }

     
	  vector_t list_;
     };
} // namespace internal_

typedef internal_::c_deleter_t c_delete;
typedef internal_::cpp_deleter_t cpp_delete;
typedef internal_::noop_deleter_t noop_delete;

static internal_::MemoryHandler mem_handler;

// =============================================================================

void* create_mem_cfl(const char* name, unsigned int D, const long dims[])
{
     debug_printf(DP_DEBUG2, "in: create_mem_cfl\n");
     
     if (D > DIMS_MAX) {
	  BART_WARN("create_mem_cfl: D > DIMS_MAX: %d > %d!\n", D, DIMS_MAX);
     }
     
     return mem_handler.allocate_mem_cfl<cx_float>(name,
						   D,
						   const_cast<long*>(dims));
}

void* create_anon_mem_cfl(unsigned int D, const long dims[])
{
     debug_printf(DP_DEBUG2, "in: create_anon_mem_cfl\n");
     
     if (D > DIMS_MAX) {
	  BART_WARN("create_anon_mem_cfl: D > DIMS_MAX: %d > %d!\n", D, DIMS_MAX);
     }
     
     return mem_handler.allocate_mem_cfl<cx_float>(D, const_cast<long*>(dims));
}

void* load_mem_cfl(const char* name, unsigned int D, long dims[])
{
     debug_printf(DP_DEBUG2, "in: load_mem_cfl\n");

     if (D > DIMS_MAX) {
	  BART_WARN("load_mem_cfl: D > DIMS_MAX: %d > %d!\n", D, DIMS_MAX);
     }
     
     cx_float* ret = mem_handler.load_mem_cfl(name, D, dims);

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
     debug_printf(DP_DEBUG2, "in: register_mem_cfl_non_managed\n");

     /* NB: due to the use of noop_delete, the memory will not be freed upon
      *     destruction
      */
     register_mem_cfl_impl<noop_delete>(name, D, dims, data);
}

void register_mem_cfl_malloc(const char* name,
			     unsigned int D,
			     const long dims[],
			     void* data)
{
     debug_printf(DP_DEBUG2, "in: register_mem_cfl_malloc\n");
     
     register_mem_cfl_impl<c_delete>(name, D, dims, data);
}
void register_mem_cfl_new(const char* name,
			  unsigned int D,
			  const long dims[],
			  void* data)
{
     debug_printf(DP_DEBUG2, "in: register_mem_cfl_new\n");
     
     register_mem_cfl_impl<cpp_delete>(name, D, dims, data);
}

_Bool is_mem_cfl(const cx_float* ptr)
{
     return mem_handler.is_mem_cfl(ptr);
}

_Bool try_delete_mem_cfl(const cx_float* ptr)
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
     debug_printf(DP_DEBUG2, "in: deallocate_all_mem_cfl\n");
     mem_handler.clear();
}

// =============================================================================

#ifdef BART_WITH_PYTHON
bool register_mem_cfl_python(const char* name, const numpy_farray_t& array)
{
     debug_printf(DP_DEBUG2, "in: register_mem_cfl_python\n");
     if (array.ndim() > DIMS_MAX) {
	  BART_WARN("register_mem_cfl_python: D > DIMS_MAX: %d > %d!\n",
		    array.ndim(),
		    DIMS_MAX);
     }

     // This call may throw!
     mem_handler.register_mem_cfl(name, array);
     return true;
}
#endif /* BART_WITH_PYTHON */
