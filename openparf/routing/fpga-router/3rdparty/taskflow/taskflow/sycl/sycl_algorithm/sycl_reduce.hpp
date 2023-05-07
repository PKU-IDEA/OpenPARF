#pragma once

#include "../sycl_flow.hpp"

namespace tf {

// Function: reduce
template <typename I, typename T, typename C>
syclTask syclFlow::reduce(I first, I last, T* res, C&& op) {

  // TODO: special case N == 0?
  size_t N = std::distance(first, last);
  size_t B = _default_group_size(N);

  auto node = _graph.emplace_back(
  
  [=, op=std::forward<C>(op)](sycl::handler& handler) mutable {
      
    // create a shared memory
    sycl::accessor<
      T, 1, sycl::access::mode::read_write, sycl::access::target::local
    > shm(sycl::range<1>(B), handler);
    
    // perform parallel reduction
    handler.parallel_for(
      sycl::nd_range<1>{sycl::range<1>(B), sycl::range<1>(B)},
      [=] (sycl::nd_item<1> item) { 
  
      size_t tid = item.get_global_id(0);

      if(tid >= N) {
        return;
      }

      shm[tid] = *(first+tid);
      
      for(size_t i=tid+B; i<N; i+=B) {
        shm[tid] = op(shm[tid], *(first+i));
      }

      item.barrier(sycl::access::fence_space::local_space);
  
      for(size_t s = B / 2; s > 0; s >>= 1) {
        if(tid < s && tid + s < N) {
          shm[tid] = op(shm[tid], shm[tid+s]);
        }
        item.barrier(sycl::access::fence_space::local_space);
      }

      if(tid == 0) {
        *res = op(*res, shm[0]);
      }
    });
  });

  return syclTask(node);
}

// Function: uninitialized_reduce
template <typename I, typename T, typename C>
syclTask syclFlow::uninitialized_reduce(I first, I last, T* res, C&& op) {

  // TODO: special case N == 0?
  size_t N = std::distance(first, last);
  size_t B = _default_group_size(N);

  auto node = _graph.emplace_back(
  
  [=, op=std::forward<C>(op)](sycl::handler& handler) mutable {
      
    // create a shared memory
    sycl::accessor<
      T, 1, sycl::access::mode::read_write, sycl::access::target::local
    > shm(sycl::range<1>(B), handler);
    
    // perform parallel reduction
    handler.parallel_for(
      sycl::nd_range<1>{sycl::range<1>(B), sycl::range<1>(B)},
      [=] (sycl::nd_item<1> item) { 
  
      size_t tid = item.get_local_id(0);

      if(tid >= N) {
        return;
      }

      shm[tid] = *(first+tid);
      
      for(size_t i=tid+B; i<N; i+=B) {
        shm[tid] = op(shm[tid], *(first+i));
      }

      item.barrier(sycl::access::fence_space::local_space);
  
      for(size_t s = B / 2; s > 0; s >>= 1) {
        if(tid < s && tid + s < N) {
          shm[tid] = op(shm[tid], shm[tid+s]);
        }
        item.barrier(sycl::access::fence_space::local_space);
      }

      if(tid == 0) {
        *res = shm[0];
      }
    });
  });

  return syclTask(node);
}

// ----------------------------------------------------------------------------
// rebind methods
// ----------------------------------------------------------------------------

// Function: reduce
template <typename I, typename T, typename C>
void syclFlow::rebind_reduce(syclTask task, I first, I last, T* res, C&& op) {

  // TODO: special case N == 0?
  size_t N = std::distance(first, last);
  size_t B = _default_group_size(N);

  task._node->_func = 
  [=, op=std::forward<C>(op)](sycl::handler& handler) mutable {
      
    // create a shared memory
    sycl::accessor<
      T, 1, sycl::access::mode::read_write, sycl::access::target::local
    > shm(sycl::range<1>(B), handler);
    
    // perform parallel reduction
    handler.parallel_for(
      sycl::nd_range<1>{sycl::range<1>(B), sycl::range<1>(B)},
      [=] (sycl::nd_item<1> item) { 
  
      size_t tid = item.get_global_id(0);

      if(tid >= N) {
        return;
      }

      shm[tid] = *(first+tid);
      
      for(size_t i=tid+B; i<N; i+=B) {
        shm[tid] = op(shm[tid], *(first+i));
      }

      item.barrier(sycl::access::fence_space::local_space);
  
      for(size_t s = B / 2; s > 0; s >>= 1) {
        if(tid < s && tid + s < N) {
          shm[tid] = op(shm[tid], shm[tid+s]);
        }
        item.barrier(sycl::access::fence_space::local_space);
      }

      if(tid == 0) {
        *res = op(*res, shm[0]);
      }
    });
  };
}

// Function: uninitialized_reduce
template <typename I, typename T, typename C>
void syclFlow::rebind_uninitialized_reduce(
  syclTask task, I first, I last, T* res, C&& op
) {

  // TODO: special case N == 0?
  size_t N = std::distance(first, last);
  size_t B = _default_group_size(N);

  task._node->_func = 
  [=, op=std::forward<C>(op)](sycl::handler& handler) mutable {
      
    // create a shared memory
    sycl::accessor<
      T, 1, sycl::access::mode::read_write, sycl::access::target::local
    > shm(sycl::range<1>(B), handler);
    
    // perform parallel reduction
    handler.parallel_for(
      sycl::nd_range<1>{sycl::range<1>(B), sycl::range<1>(B)},
      [=] (sycl::nd_item<1> item) { 
  
      size_t tid = item.get_local_id(0);

      if(tid >= N) {
        return;
      }

      shm[tid] = *(first+tid);
      
      for(size_t i=tid+B; i<N; i+=B) {
        shm[tid] = op(shm[tid], *(first+i));
      }

      item.barrier(sycl::access::fence_space::local_space);
  
      for(size_t s = B / 2; s > 0; s >>= 1) {
        if(tid < s && tid + s < N) {
          shm[tid] = op(shm[tid], shm[tid+s]);
        }
        item.barrier(sycl::access::fence_space::local_space);
      }

      if(tid == 0) {
        *res = shm[0];
      }
    });
  };
}


}  // end of namespace tf -----------------------------------------------------


