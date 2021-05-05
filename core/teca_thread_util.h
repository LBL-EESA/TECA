#ifndef teca_thread_utils_h
#define teca_thread_utils_h

/// @file

#include "teca_common.h"
#include "teca_mpi.h"
#include <deque>

/// Codes for dealing with threading
namespace teca_thread_util
{
/** load balances threads across an MPI communication space such that on the
 * individual nodes physical cores each receive the same number of threads.
 * This is an MPI collective call.  Building the affinity map relies on
 * features available only in _GNU_SOURCE.  On systems where these features are
 * unavailable, when automated detection of the number of threads is requested,
 * the call will fail and the n_threads will be set to 1,
 *
 * @param[in] comm an MPI communcation space to load balance threads across.
 *                 the communicator is used to coordinate affinity mapping such that
 *                 each rank can allocate a number of threads bound to unique cores.
 *
 * @param[in] base_core_id identifies the core in use by this MPI rank's main
 *                         thread. if -1 is passed this will be automatically
 *                         determined.
 *
 * @param[in] n_requested the number of requested threads per rank. Passing a value of
 *                        -1 results in use of all the cores on the node such that each
 *                        physical core is assigned exactly 1 thread. Note that for
 *                        performance reasons hyperthreads are not used here. The
 *                        suggested number of threads is retruned in n_threads, and the
 *                        returned affinity map specifies which core the thread should
 *                        be bound to to acheive this. Passing n_requested >= 1
 *                        specifies a run time override. This indicates that caller
 *                        wants to use a specific number of threads, rather than one per
 *                        physical core. In this case the affinity map is also
 *                        constructed.
 *
 * @param[in] bind if true extra work is done to determine an affinity map such
 *                 that each thread can be bound to a unique core on the node.
 *
 * @param[in] verbose prints a report decribing the affinity map.
 *
 * @param[in,out] n_threads if n_requested is -1, this will be set to the number of threads
 *                          one can use such that there is one thread per phycial core
 *                          taking into account all ranks running on the node. if
 *                          n_requested is >= 1 n_threads will be set to n_requested. This
 *                          allows a run time override for cases when the caller knows how
 *                          she wants to schedule things. if an error occurs and n_requested
 *                          is -1 this will be set to 1.
 *
 * @param[out] affinity an affinity map, describing for each of n_threads,
 *                      a core id that the thread can be bound to. if n_requested is -1
 *                      then the map will conatin an entry for each of n_threads where
 *                      each of the threads is assigned a unique phyical core.  when
 *                      n_requested is >= 1 the map contains an enrty for each of the
 *                      n_requested threads such that when more threads are requested
 *                      than cores each core is assigned approximately the same number of
 *                      threads.
 *
 * @returns 0 on success
 */
int thread_parameters(MPI_Comm comm, int base_core_id, int n_requested,
    bool bind, bool verbose, int &n_threads, std::deque<int> &affinity);
};

#endif
