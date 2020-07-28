/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/framework/collective.h"

#include "absl/strings/escaping.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

namespace {
// A RegistrationInfo object stores a collective implementation registration
// details.  `factory` is used to create instances of the collective
// implementation.
struct RegistrationInfo {
  // This constructor also creates, and stores in `param_resolver_instance`,
  // what is effectively a static instance of the collective implementation.
  // During param resolution of collective ops we return this static instance.
  // The actual op execution gets a fresh instance using `factory`.
  RegistrationInfo(const string& n, CollectiveRegistry::Factory f)
      : name(n),
        factory(std::move(f)),
        param_resolver_instance(this->factory()) {}
  string name;
  CollectiveRegistry::Factory factory;
  CollectiveImplementationInterface* param_resolver_instance;
};

std::vector<RegistrationInfo>* MutableCollectiveRegistry() {
  static std::vector<RegistrationInfo>* registry =
      new std::vector<RegistrationInfo>;
  return registry;
}
}  // namespace

string CollGroupRuntimeDetails::ToString() const {
  return strings::StrCat("CollGroupRuntimeDetails {communicator_key=",
                         absl::CEscape(communicator_key), "}");
}

string CollGroupParams::ToString() const {
  return strings::StrCat(
      "CollGroupParams {group_key=", group_key, " group_size=", group_size,
      " device_type=", device_type.type_string(), " num_tasks=", num_tasks,
      " runtime_details=", runtime_details.ToString(), "}");
}

CollInstanceParams& CollInstanceParams::operator=(
    const CollInstanceParams& other) {
  if (this != &other) {
    instance_key = other.instance_key;
    type = other.type;
    data_type = other.data_type;
    shape = other.shape;
    device_names.clear();
    device_names.assign(other.device_names.begin(), other.device_names.end());
    task_names.assign(other.task_names.begin(), other.task_names.end());
    same_num_devices_per_task = other.same_num_devices_per_task;
    num_devices_per_task = other.num_devices_per_task;
    gpu_ring_order = other.gpu_ring_order;
    impl_details.subdiv_offsets.assign(
        other.impl_details.subdiv_offsets.begin(),
        other.impl_details.subdiv_offsets.end());
    impl_details.subdiv_permutations.clear();
    for (auto p : other.impl_details.subdiv_permutations) {
      impl_details.subdiv_permutations.push_back(
          std::vector<int>(p.begin(), p.end()));
    }
    impl_details.subdiv_source_rank.assign(
        other.impl_details.subdiv_source_rank.begin(),
        other.impl_details.subdiv_source_rank.end());
    impl_details.dependencies = other.impl_details.dependencies;
  }
  return *this;
}

string CollInstanceParams::ToString() const {
  string v =
      strings::StrCat("CollInstanceParams { instance_key=", instance_key,
                      " type=", type, " data_type=", DataTypeString(data_type),
                      " shape=", shape.DebugString(), " devices {");
  for (const auto& d : device_names) {
    strings::StrAppend(&v, d, ",");
  }
  strings::StrAppend(&v, "} task_names={");
  for (const auto& n : task_names) {
    strings::StrAppend(&v, n, ", ");
  }
  strings::StrAppend(&v, "} num_devices_per_task={");
  for (const auto& dpt : num_devices_per_task) {
    strings::StrAppend(&v, dpt.first, ": ", dpt.second, ", ");
  }
  strings::StrAppend(&v, "}, collective_name=", impl_details.collective_name,
                     ", subdiv_offsets={");
  strings::StrAppend(&v, "}, subdiv_offsets={");
  for (const auto& d : impl_details.subdiv_offsets) {
    strings::StrAppend(&v, d, ",");
  }
  strings::StrAppend(&v, "}, subdiv_perms={");
  for (const auto& p : impl_details.subdiv_permutations) {
    strings::StrAppend(&v, "{");
    for (const auto& i : p) {
      strings::StrAppend(&v, i, ",");
    }
    strings::StrAppend(&v, "}");  // one subdiv
  }
  if (!impl_details.subdiv_source_rank.empty()) {
    strings::StrAppend(&v, " subdiv_source_rank={");
    for (const auto& r : impl_details.subdiv_source_rank) {
      strings::StrAppend(&v, r, ",");
    }
    strings::StrAppend(&v, "}");
  }
  strings::StrAppend(&v, "}");  // all subdivs
  return v;
}

string CollTaskParams::ToString() const {
  string v = strings::StrCat("CollTaskParams {is_local={");
  for (const auto& b : is_local) {
    strings::StrAppend(&v, static_cast<int>(b), ",");
  }
  strings::StrAppend(&v, "}}");
  return v;
}

string CollectiveParams::ToString() const {
  string v = strings::StrCat("CollectiveParams ", name, " {", group.ToString());
  strings::StrAppend(&v, " ", instance.ToString());
  strings::StrAppend(&v, " ", task.ToString());
  strings::StrAppend(&v, " default_rank=", default_rank,
                     " is_source=", is_source, " source_rank=", source_rank,
                     " subdiv_rank={");
  for (const auto& r : subdiv_rank) {
    strings::StrAppend(&v, r, ",");
  }
  strings::StrAppend(&v, "}}");
  return v;
}

/*static*/ OpKernelContext::Params* CollectiveExecutor::CtxParams(
    OpKernelContext* ctx) {
  return ctx->params_;
}

CollectiveContext::CollectiveContext(CollectiveExecutor* col_exec,
                                     const DeviceMgr* dev_mgr,
                                     OpKernelContext* ctx,
                                     OpKernelContext::Params* op_params,
                                     const CollectiveParams& col_params,
                                     const string& exec_key, int64 step_id,
                                     const Tensor* input, Tensor* output)
    : col_exec(col_exec),
      dev_mgr(dev_mgr),
      op_ctx(ctx),
      op_params(op_params),
      col_params(col_params),
      exec_key(exec_key),
      step_id(step_id),
      input(input),
      output(output),
      device(nullptr),
      device_name(col_params.instance.device_names[col_params.default_rank]) {}

/*static*/
int64 CollectiveExecutor::kInvalidId = -1;

/*static*/
Status CollectiveRegistry::Lookup(
    const string& collective_name,
    CollectiveImplementationInterface** implementation) {
  return LookupHelper(collective_name, implementation, false);
}

/*static*/
Status CollectiveRegistry::LookupParamResolverInstance(
    const string& collective_name,
    CollectiveImplementationInterface** implementation) {
  return LookupHelper(collective_name, implementation, true);
}

/*static*/
void CollectiveRegistry::GetAll(
    std::vector<CollectiveImplementationInterface*>* implementations) {
  std::vector<RegistrationInfo>* registry = MutableCollectiveRegistry();
  for (const RegistrationInfo& reg_info : *registry)
    implementations->emplace_back(reg_info.factory());
}

/*static*/
Status CollectiveRegistry::Register(const string& collective_name,
                                    Factory factory) {
  std::vector<RegistrationInfo>* registry = MutableCollectiveRegistry();
  for (const RegistrationInfo& reg_info : *registry) {
    if (reg_info.name == collective_name)
      return errors::Internal("Already registered collective ",
                              collective_name);
  }
  registry->emplace_back(collective_name, std::move(factory));
  return Status::OK();
}

/*static*/
Status CollectiveRegistry::LookupHelper(
    const string& collective_name,
    CollectiveImplementationInterface** implementation, bool param_resolver) {
  std::vector<RegistrationInfo>* registry = MutableCollectiveRegistry();
  for (const RegistrationInfo& reg_info : *registry) {
    if (reg_info.name == collective_name) {
      if (param_resolver) {
        *implementation = reg_info.param_resolver_instance;
      } else {
        *implementation = reg_info.factory();
      }
      return Status::OK();
    }
  }
  return errors::Internal(
      "CollectiveRegistry::Lookup did not find collective implementation ",
      collective_name);
}

}  // namespace tensorflow
/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_CORE_FRAMEWORK_COLLECTIVE_H_
#define TENSORFLOW_CORE_FRAMEWORK_COLLECTIVE_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class BufRendezvous;
class CancellationManager;
class CompleteGroupRequest;
class CompleteGroupResponse;
class CompleteInstanceRequest;
class CompleteInstanceResponse;
class Device;
class DeviceMgr;
class GetStepSequenceRequest;
class GetStepSequenceResponse;
class Tensor;

// Types of supported collective operations.
enum CollectiveType {
  REDUCTION_COLLECTIVE = 0,
  BROADCAST_COLLECTIVE,
  GATHER_COLLECTIVE,
  UNDEFINED_COLLECTIVE,
};

// Some collective op implementations require runtime group configuration from
// the OpKernel.  Currently, this struct is used to set communicator key for
// NCCL-based collective implementation.
struct CollGroupRuntimeDetails {
  string communicator_key;  // for communicator-based techniques e.g. NCCL
  string ToString() const;
};

// Data common to all members of a device group.
// All members share the same device set but its order is
// particular to an instance so it is stored there.
struct CollGroupParams {
  int32 group_key;
  int32 group_size;
  DeviceType device_type;
  int32 num_tasks;  // number of distinct tasks in group
  CollGroupRuntimeDetails runtime_details;
  string ToString() const;
  CollGroupParams()
      : group_key(0), group_size(0), device_type(DEVICE_CPU), num_tasks(0) {}
};

// The best implementation of a collective op depends on many factors
// including the number of devices involved, the topology of
// interconnects between them and the sizes of inputs.  This structure
// is used in generating and representing data movement choreography
// for each specific algorithm, hence it does not have a single, fixed
// interpretation.  On first execution the runtime will update this
// structure with decisions that will guide all subsequent executions.
struct CollImplDetails {
  string collective_name;
  std::vector<std::vector<int>> subdiv_permutations;
  std::vector<int> subdiv_offsets;
  std::vector<int> subdiv_source_rank;  // rank of source in each subdiv
  std::vector<int32>
      dependencies;           // collective instances on which this node depends
  string communication_hint;  // user-supplied hint for implementation choice,
                              // e.g. ring or nccl
  float timeout_seconds;      // If non zero, set a completion timeout for the
                              // collective op to detect staleness.
};

// Data common to all members of a collective instance.
struct CollInstanceParams {
  // Identifies all participating graph nodes.
  int32 instance_key = -1;
  CollectiveType type = UNDEFINED_COLLECTIVE;
  DataType data_type = DT_FLOAT;
  TensorShape shape = {0};
  // Fully qualified name of device for each member, in default rank order.
  std::vector<string> device_names;
  // Task name prefix of corresponding device name.
  std::vector<string> task_names;
  // True if every task has the same number of devices.
  bool same_num_devices_per_task = false;
  // Task -> number of devices on that task.
  std::unordered_map<string, int32> num_devices_per_task;
  // If passed in to GPUOptions in ConfigProto, defines a good ring order for
  // GPUs.  Assumes same GPU configuration at each worker.
  string gpu_ring_order = "";
  CollImplDetails impl_details;
  string ToString() const;
  CollInstanceParams& operator=(const struct CollInstanceParams& other);
};

// Data common to all instance members in the same task.
struct CollTaskParams {
  // True for devices that are local to the process, i.e. no RPC needed.
  std::vector<bool> is_local;
  string ToString() const;
};

// Unique to a single CollectiveOp node.
struct CollectiveParams {
  CollGroupParams group;
  CollInstanceParams instance;
  CollTaskParams task;

  string name = "";        // node name used only for log or error messages
  int default_rank = -1;   // index of this op within device_names
  bool is_source = false;  // broadcast only
  int source_rank = -1;    // broadcast only
  // Rank of this device in each subdivision permutation.
  std::vector<int> subdiv_rank;
  std::unique_ptr<OpKernel> merge_op;  // reduction only
  std::unique_ptr<OpKernel> final_op;  // reduction only
  string ToString() const;
};

class CollectiveExecutor;

// Interface that provides resolution of device localities.
class DeviceResolverInterface {
 public:
  virtual ~DeviceResolverInterface() {}

  // Collects DeviceAttributes protobufs from all of the devices identified
  // in 'col_params'.
  virtual void GetAllDeviceAttributesAsync(
      const std::vector<string>& devices, const std::vector<string>& tasks,
      std::vector<DeviceAttributes>* attributes,
      const StatusCallback& done) = 0;

  // Populate *attributes with the DeviceAttributes of the specified
  // device.
  virtual void GetDeviceAttributesAsync(const string& device,
                                        const string& task,
                                        DeviceAttributes* attributes,
                                        const StatusCallback& done) = 0;

  // Clear the cache of device data belonging to the specified task.
  virtual void ClearTask(const string& task) = 0;

  // Clear the cache of all device data.
  virtual void ClearCache() = 0;
};

// Interface that provides resolution of shared CollectiveParams fields.
class ParamResolverInterface {
 public:
  virtual ~ParamResolverInterface() {}

  // Called by each collective op at first execution in order to fill out
  // the CollectiveParams structure with data gathered from the full
  // (maybe distributed) collection of peer nodes.
  virtual void CompleteParamsAsync(const string& device, CollectiveParams* cp,
                                   CancellationManager* cancel_mgr,
                                   const StatusCallback& done) = 0;

  // Used within a distributed implementation to discover/verify
  // data shared across a device group.
  virtual void CompleteGroupAsync(const CompleteGroupRequest* request,
                                  CompleteGroupResponse* response,
                                  CancellationManager* cancel_mgr,
                                  const StatusCallback& done) = 0;

  // Used within a distributed implementation to discover/verify data
  // shared across an instance group.
  virtual void CompleteInstanceAsync(const CompleteInstanceRequest* request,
                                     CompleteInstanceResponse* response,
                                     CancellationManager* cancel_mgr,
                                     const StatusCallback& done) = 0;
};

// Graphs which utilize Collective Ops in a common instance must
// execute with identical step_ids even if they are disjoint graphs
// run by otherwise independent tasks.  This interface supplies
// coordinated step_ids to use in such cases.
class StepSequenceInterface {
 public:
  virtual ~StepSequenceInterface() {}

  // Used with a distributed implementation to coordinate step_id
  // sequences across tasks.
  virtual void GetStepSequenceAsync(const GetStepSequenceRequest* request,
                                    GetStepSequenceResponse* response,
                                    const StatusCallback& done) = 0;

  // Refresh the local per-graph_key step_id sequence from collective
  // group leader, if applicable.
  virtual void RefreshStepIdSequenceAsync(int64 graph_key,
                                          const StatusCallback& done) = 0;

  // Returns the step_id that should be used for initiating a new execution
  // on the specified graph. May return the same step_id multiple times if
  // RetireStepId or RefreshStepIdReservation is not called.
  virtual int64 NextStepId(int64 graph_key) = 0;

  // Reports that execution of the given step has completed successfully.
  // Should be called immediately after a step completes with OK status,
  // prior to calling NextStepId().  If the step fails, don't call.
  virtual void RetireStepId(int64 graph_key, int64 step_id) = 0;
};

// Interface that provides access to per-step CollectiveExecutor
// instances and various distributed resolution capabilities.
class CollectiveExecutorMgrInterface : public StepSequenceInterface {
 public:
  virtual ~CollectiveExecutorMgrInterface() {}

  // Returns the step-specific CollectiveExecutor, creating if one does not
  // already exist.  The caller assumes ownership of one Ref on the object.
  virtual CollectiveExecutor* FindOrCreate(int64 step_id) = 0;

  // If there is a CollectiveExecutor for step_id, remove it from the
  // table.
  virtual void Cleanup(int64 step_id) = 0;

  virtual ParamResolverInterface* GetParamResolver() const = 0;

  virtual DeviceResolverInterface* GetDeviceResolver() const = 0;
};

// Interface that a Collective Op implementation uses to exchange data
// with peers.  Note that data exchange is currently limited to types
// for which DMAHelper::CanUseDMA() returns true, i.e.  dense numeric
// types.
class PeerAccessInterface {
 public:
  virtual ~PeerAccessInterface() {}

  virtual void RecvFromPeer(const string& peer_device, const string& peer_task,
                            bool peer_is_local, const string& key,
                            Device* to_device, DeviceContext* to_device_ctx,
                            const AllocatorAttributes& to_alloc_attr,
                            Tensor* to_tensor,
                            const DeviceLocality& client_locality,
                            int dev_to_dev_stream_index,
                            const StatusCallback& done) = 0;

  virtual void PostToPeer(const string& peer_device, const string& peer_task,
                          const string& key, Device* from_device,
                          DeviceContext* from_device_ctx,
                          const AllocatorAttributes& from_alloc_attr,
                          const Tensor* from_tensor,
                          const DeviceLocality& client_locality,
                          const StatusCallback& done) = 0;

  // Runs the potentially-blocking closure/expensive callback.
  virtual void RunClosure(std::function<void()> closure) = 0;
};

class PerStepCollectiveRemoteAccess;

// A step-specific object that can execute a collective operation completely
// described by a CollectiveParams object.
class CollectiveExecutor : public PeerAccessInterface, public core::RefCounted {
 public:
  virtual void StartAbort(const Status& s) {}

  virtual void ExecuteAsync(OpKernelContext* ctx,
                            const CollectiveParams& col_params,
                            const string& exec_key, StatusCallback done) {
    done(errors::Internal(
        "A collective Op has been called in a context in which "
        "a CollectiveExecutor has not been provided."));
  }

  virtual void CompleteParamsAsync(const string& device, CollectiveParams* cp,
                                   CancellationManager* cancel_mgr,
                                   StatusCallback done) {
    done(errors::Internal(
        "A collective Op has been called in a context in which "
        "a CollectiveExecutor has not been provided."));
  }

  virtual PerStepCollectiveRemoteAccess* remote_access() { return nullptr; }

  // `WaitForDependencies` and `Launched` are used for fine-grained control of
  // execution order between collective instances.  These functions are intended
  // to be called in `Run` function of collective implementations, and may be
  // used to make part, or whole, of the collective execution ordered with
  // respect to other collective instances.
  //
  // `WaitForDependencies` will block until it is safe to continue the callee's
  // execution, where safety is defined as: ordered with respect to the
  // collective instances defined in the callee's `wait_for` attribute.
  virtual void WaitForDependencies(const CollectiveParams& col_params) {}
  // `UnblockDependencies` unblocks the dependent collective instances by
  // recording that this caller's device has completed the critical portion of
  // the collective execution.
  virtual void UnblockDependencies(const CollectiveParams& col_params) {}

  // Used to designate an invalid group or instance key.
  static int64 kInvalidId;

  // Lexically scoped handle for Ref.
  class Handle {
   public:
    explicit Handle(CollectiveExecutor* ce, bool inherit_ref) : ce_(ce) {
      if (!inherit_ref) ce->Ref();
    }
    ~Handle() { ce_->Unref(); }
    CollectiveExecutor* get() const { return ce_; }

   private:
    CollectiveExecutor* ce_;
  };

 protected:
  explicit CollectiveExecutor(CollectiveExecutorMgrInterface* cem)
      : cem_(cem) {}

  // For use only by derived classes
  static OpKernelContext::Params* CtxParams(OpKernelContext* ctx);
  CollectiveExecutorMgrInterface* cem_;

  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveExecutor);
};

// Interface of a helper object that provides a CollectiveExecutor with
// all of the remote access it needs.
class CollectiveRemoteAccess : public PeerAccessInterface,
                               public DeviceResolverInterface {
 public:
  virtual ~CollectiveRemoteAccess() {}

  virtual BufRendezvous* buf_rendezvous() = 0;
};

// A per-step version of CollectiveRemoteAccess that cleans up outstanding
// communications in case step execution is abandoned.
class PerStepCollectiveRemoteAccess : public CollectiveRemoteAccess {
 public:
  virtual ~PerStepCollectiveRemoteAccess() {}
  virtual void StartAbort(const Status& s) = 0;
};

class CollectiveContext {
 public:
  CollectiveContext(CollectiveExecutor* col_exec, const DeviceMgr* dev_mgr,
                    OpKernelContext* ctx, OpKernelContext::Params* op_params,
                    const CollectiveParams& col_params, const string& exec_key,
                    int64 step_id, const Tensor* input, Tensor* output);

  virtual ~CollectiveContext() = default;

  CollectiveExecutor* col_exec;        // Not owned
  const DeviceMgr* dev_mgr;            // Not owned
  OpKernelContext* op_ctx;             // Not owned
  OpKernelContext::Params* op_params;  // Not owned
  const CollectiveParams& col_params;
  const string exec_key;
  const int64 step_id;
  const Tensor* input;  // Not owned
  Tensor* output;       // Not owned
  Device* device;       // The device for which this instance labors
  const string device_name;
  DeviceLocality device_locality;
};

// Interface of a Collective Op implementation.  Each specific CollectiveOp will
// implement this interface and register the implementation via the
// CollectiveRegistry detailed below.  See common_runtime/ring_reducer and
// common_runtime/hierarchical_tree_broadcaster for examples.
class CollectiveImplementationInterface {
 public:
  virtual ~CollectiveImplementationInterface() = default;

  // Initializes the portions of `col_params` specific to this
  // implementation.  Called exactly once for every Collective instance during
  // the CollectiveParams resolution process when the graph is first executed,
  // at the end of `CompleteInstanceLocal()`.
  // NOTE(ayushd): This is effectively a static function because it modifies the
  // `col_params` passed in and should not manipulate any data members.  However
  // because it is virtual and needs to be implemented by every derived class we
  // do not mark it as static.
  virtual Status InitializeCollectiveParams(CollectiveParams* col_params) = 0;

  // Prepares the CollectiveContext for executing this CollectiveImplementation.
  // Called from CollectiveExecutor right before calling Run().  The
  // CollectiveContext passed in must outlive the CollectiveImplementation
  // object.
  virtual Status InitializeCollectiveContext(
      std::shared_ptr<CollectiveContext> col_ctx) = 0;

  // Performs collective implementation specific group initialization.  The
  // intention is to do group-specific initialization of runtime details for the
  // collective implementation.  Currently used only to set `communicator_key`
  // in techniques which use a communicator for distributed collectives (NCCL).
  virtual Status InitializeCollectiveGroupRuntimeDetails(
      CollGroupRuntimeDetails* col_group_runtime_details) = 0;

  // Processes and moves data according to the logic of this Collective
  // implementation.  Relies on appropriate initialization of op-specific
  // CollectiveParams in InitializeCollectiveParams(), as well as appropriate
  // context initialization in InitializeCollectiveContext().
  virtual void Run(StatusCallback done) = 0;
};

// Static-methods only class for registering and looking up collective
// implementations.
class CollectiveRegistry {
 public:
  using Factory = std::function<CollectiveImplementationInterface*()>;
  // Looks up a previously registered CollectiveImplementation under
  // `collective_name`.  If found, creates an instance of the implementation and
  // assign to `implementation`.
  static Status Lookup(const string& collective_name,
                       CollectiveImplementationInterface** implementation);

  // Looks up a previously registered CollectiveImplementation under
  // `collective_name`.  If found, returns the static instance of this
  // implementation via `implementation`.  This instance should only be used to
  // call InitializateCollectiveParams.
  static Status LookupParamResolverInstance(
      const string& collective_name,
      CollectiveImplementationInterface** implementation);

  // Returns all registered collective implementations.
  static void GetAll(
      std::vector<CollectiveImplementationInterface*>* implementations);

 private:
  friend class CollectiveRegistration;
  // Registers a CollectiveImplementation with name `collective_name` and
  // factory `factory`.  The latter is a function used to create instances of
  // the CollectiveImplementation.  Also creates a static instance of the
  // implementation - this instance is used during param resolution and should
  // only be used to call InitializeCollectiveParams.
  static Status Register(const string& collective_name, Factory factory);

  static Status LookupHelper(const string& collective_name,
                             CollectiveImplementationInterface** implementation,
                             bool param_resolver);
};

// Class used to call CollectiveRegistry::Register.  This should only be used to
// create a global static object.
class CollectiveRegistration {
 public:
  CollectiveRegistration(const string& collective_name,
                         CollectiveRegistry::Factory factory) {
    TF_CHECK_OK(CollectiveRegistry::Register(collective_name, factory));
  }
};

#define REGISTER_COLLECTIVE(name, implementation)             \
  static CollectiveRegistration register_##name##_collective( \
      #name, []() { return new implementation; });

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_COLLECTIVE_H_
/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

namespace {
class CollectiveOpKernel : public AsyncOpKernel {
 public:
  explicit CollectiveOpKernel(OpKernelConstruction* c) : AsyncOpKernel(c) {}

  // A string encoding instance, frame and iter to be handed off to
  // the implementation for use in generating RecvBuf keys.
  string GetCollectiveKey(OpKernelContext* c) {
    return strings::StrCat(col_params_.instance.instance_key, ":",
                           c->frame_iter().frame_id, ":",
                           c->frame_iter().iter_id);
  }

  // Returns false if calling invocation of ComputeAsync should return
  // immediately.
  bool CanProceedWithCompute(OpKernelContext* c, CollectiveExecutor* col_exec,
                             const DoneCallback& done) {
    if (col_params_.group.group_size >
        col_params_.instance.device_names.size()) {
      // This is the first invocation: Finish initializing col_params_.
      // Schedule the `CompleteParamsAsync` call on a work queue that can handle
      // blocking work because it's not guaranteed that this call cannot block.
      c->collective_executor()->RunClosure([this, c, done, col_exec]() {
        VLOG(1) << "CollectiveOpKernel CompleteParams for collective "
                << col_params_.name << " device " << c->device()->name()
                << " group " << col_params_.group.group_key << " instance "
                << col_params_.instance.instance_key;
        col_exec->CompleteParamsAsync(
            c->device()->name(), &col_params_, c->cancellation_manager(),
            [this, c, done](const Status& s) {
              if (s.ok()) {
                col_params_.instance.impl_details.dependencies = dependencies_;
                ComputeAsync(c, done);
              } else {
                c->SetStatus(s);
                done();
              }
            });
      });
      return false;
    }
    return true;
  }

  CollectiveParams col_params_;
  std::vector<int32> dependencies_;
};

class CollectiveGatherOpKernel : public CollectiveOpKernel {
 public:
  explicit CollectiveGatherOpKernel(OpKernelConstruction* c)
      : CollectiveOpKernel(c) {
    col_params_.instance.type = GATHER_COLLECTIVE;
    OP_REQUIRES_OK(c, c->GetAttr("group_size", &col_params_.group.group_size));
    OP_REQUIRES(
        c, col_params_.group.group_size > 0,
        errors::InvalidArgument("group_size must be positive integer but got ",
                                col_params_.group.group_size));
    OP_REQUIRES_OK(c, c->GetAttr("group_key", &col_params_.group.group_key));
    OP_REQUIRES_OK(
        c, c->GetAttr("instance_key", &col_params_.instance.instance_key));
    OP_REQUIRES_OK(c, c->GetAttr("T", &col_params_.instance.data_type));
    OP_REQUIRES_OK(
        c, c->GetAttr("communication_hint",
                      &col_params_.instance.impl_details.communication_hint));
    OP_REQUIRES_OK(
        c, c->GetAttr("timeout_seconds",
                      &col_params_.instance.impl_details.timeout_seconds));
    const NodeDef& real_node = c->def();
    col_params_.name = strings::StrCat(real_node.name(), ": Gather");
    col_params_.group.device_type = c->device_type();
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    CollectiveExecutor* col_exec = c->collective_executor();
    OP_REQUIRES_ASYNC(
        c, col_exec,
        errors::Internal(
            "Failed to get CollectiveExecutor from OpKernelContext for Op ",
            col_params_.name),
        done);

    auto output_shape = c->input(0).shape();
    output_shape.set_dim(
        0, output_shape.dim_size(0) * col_params_.group.group_size);
    col_params_.instance.shape = output_shape;

    // Allocate output on the first pass through this function.  This must be
    // done immediately, while we're still in the executor thread.  Otherwise
    // the memory is not guaranteed to be unused by any concurrently executing
    // GPU kernel.
    if (c->mutable_output(0) == nullptr) {
      // Allocate the output tensor.
      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(
          c, c->allocate_output(0, col_params_.instance.shape, &output), done);
    }
    if (!CanProceedWithCompute(c, col_exec, done)) return;

    auto actual_done = [c, group_key = col_params_.group.group_key,
                        instance_key = col_params_.instance.instance_key,
                        done](const Status& s) {
      VLOG(1) << "CollectiveGatherOpKernel ExecuteAsync done for collective "
              << c->op_kernel().name() << " device " << c->device()->name()
              << " group " << group_key << " instance " << instance_key
              << " status " << s;
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };
    VLOG(1) << "CollectiveGatherOpKernel ExecuteAsync start for collective "
            << col_params_.name << " device " << c->device()->name()
            << " group " << col_params_.group.group_key << " instance "
            << col_params_.instance.instance_key;
    col_exec->ExecuteAsync(c, col_params_, GetCollectiveKey(c), actual_done);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveGatherOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("CollectiveGather").Device(DEVICE_CPU),
                        CollectiveGatherOpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveGather").Device(DEVICE_GPU),
                        CollectiveGatherOpKernel);

class CollectiveReduceOpKernel : public CollectiveOpKernel {
 public:
  explicit CollectiveReduceOpKernel(OpKernelConstruction* c)
      : CollectiveOpKernel(c) {
    col_params_.instance.type = REDUCTION_COLLECTIVE;
    OP_REQUIRES_OK(c, c->GetAttr("group_size", &col_params_.group.group_size));
    OP_REQUIRES(
        c, col_params_.group.group_size > 0,
        errors::InvalidArgument("group_size must be positive integer but got ",
                                col_params_.group.group_size));
    OP_REQUIRES_OK(c, c->GetAttr("group_key", &col_params_.group.group_key));
    OP_REQUIRES_OK(
        c, c->GetAttr("instance_key", &col_params_.instance.instance_key));
    OP_REQUIRES_OK(
        c, c->GetAttr("subdiv_offsets",
                      &col_params_.instance.impl_details.subdiv_offsets));
    string merge_op_name;
    OP_REQUIRES_OK(c, c->GetAttr("merge_op", &merge_op_name));
    if (merge_op_name == "Max") {
      merge_op_name = "Maximum";
    } else if (merge_op_name == "Min") {
      merge_op_name = "Minimum";
    }
    string final_op_name;
    OP_REQUIRES_OK(c, c->GetAttr("final_op", &final_op_name));
    OP_REQUIRES(c, final_op_name == "Id" || final_op_name == "Div",
                errors::InvalidArgument(
                    "final_op must be one of {\"Id\", \"Div\"} but got ",
                    final_op_name));
    OP_REQUIRES_OK(c, c->GetAttr("T", &col_params_.instance.data_type));
    OP_REQUIRES_OK(c, c->GetAttr("wait_for", &dependencies_));
    OP_REQUIRES_OK(
        c, c->GetAttr("communication_hint",
                      &col_params_.instance.impl_details.communication_hint));
    OP_REQUIRES_OK(
        c, c->GetAttr("timeout_seconds",
                      &col_params_.instance.impl_details.timeout_seconds));
    VLOG(2) << "CollectiveReduce instance " << col_params_.instance.instance_key
            << " merge_op " << merge_op_name << " final_op " << final_op_name
            << " communication_hint "
            << col_params_.instance.impl_details.communication_hint
            << " timeout " << col_params_.instance.impl_details.timeout_seconds;

    const NodeDef& real_node = c->def();
    col_params_.name = strings::StrCat(real_node.name(), ": Reduce(",
                                       merge_op_name, ",", final_op_name, ")");
    col_params_.group.device_type = c->device_type();

    // Find the OpKernels by name, type and device type.
    NodeDef sub_node;
    // The merge_op takes two inputs
    sub_node.add_input(real_node.input(0));
    sub_node.add_input(real_node.input(0));
    sub_node.set_device(real_node.device());
    SetAttrValue(col_params_.instance.data_type,
                 &(*sub_node.mutable_attr())["T"]);
    col_params_.merge_op = BuildOpKernel(c, merge_op_name, &sub_node);
    col_params_.final_op = BuildOpKernel(c, final_op_name, &sub_node);
  }

  std::unique_ptr<OpKernel> BuildOpKernel(OpKernelConstruction* c,
                                          const string& name,
                                          NodeDef* sub_node) {
    std::unique_ptr<OpKernel> k;
    if (name.empty() || name == "Id") return k;
    sub_node->set_name(name);
    sub_node->set_op(name);
    Status status;
    k = CreateOpKernel(c->device_type(), c->device(),
                       c->device()->GetAllocator(AllocatorAttributes()),
                       *sub_node, c->graph_def_version(), &status);
    if (!status.ok()) {
      c->CtxFailureWithWarning(errors::Internal("Failed to build OpKernel for ",
                                                name, " : ",
                                                status.error_message()));
    }
    return k;
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    CollectiveExecutor* col_exec = c->collective_executor();
    OP_REQUIRES_ASYNC(
        c, col_exec,
        errors::Internal(
            "Failed to get CollectiveExecutor from OpKernelContext for Op ",
            col_params_.name),
        done);
    // Allocate output on the first pass through this function.  This must be
    // done immediately, while we're still in the executor thread.  Otherwise
    // the memory is not guaranteed to be unused by any concurrently executing
    // GPU kernel.
    if (c->mutable_output(0) == nullptr) {
      // Allocate the output tensor, trying to reuse the input.
      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(c,
                           c->forward_input_or_allocate_output(
                               {0}, 0, c->input(0).shape(), &output),
                           done);
      col_params_.instance.shape = c->input(0).shape();
    }
    if (!CanProceedWithCompute(c, col_exec, done)) return;

    auto actual_done = [c, group_key = col_params_.group.group_key,
                        instance_key = col_params_.instance.instance_key,
                        done](const Status& s) {
      VLOG(1) << "CollectiveReduceOpKernel ExecuteAsync done for collective "
              << c->op_kernel().name() << " device " << c->device()->name()
              << " group " << group_key << " instance " << instance_key
              << " status " << s;
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };
    VLOG(1) << "CollectiveReduceOpKernel ExecuteAsync start for collective "
            << col_params_.name << " device " << c->device()->name()
            << " group " << col_params_.group.group_key << " instance "
            << col_params_.instance.instance_key;
    col_exec->ExecuteAsync(c, col_params_, GetCollectiveKey(c), actual_done);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveReduceOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("CollectiveReduce").Device(DEVICE_CPU),
                        CollectiveReduceOpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveReduce").Device(DEVICE_GPU),
                        CollectiveReduceOpKernel);

class CollectiveBcastSendOpKernel : public CollectiveOpKernel {
 public:
  explicit CollectiveBcastSendOpKernel(OpKernelConstruction* c)
      : CollectiveOpKernel(c) {
    col_params_.instance.type = BROADCAST_COLLECTIVE;
    OP_REQUIRES_OK(c, c->GetAttr("group_size", &col_params_.group.group_size));
    OP_REQUIRES(
        c, col_params_.group.group_size > 0,
        errors::InvalidArgument("group_size must be positive integer but got ",
                                col_params_.group.group_size));
    OP_REQUIRES_OK(c, c->GetAttr("group_key", &col_params_.group.group_key));
    OP_REQUIRES_OK(
        c, c->GetAttr("instance_key", &col_params_.instance.instance_key));
    OP_REQUIRES_OK(c, c->GetAttr("T", &col_params_.instance.data_type));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &col_params_.instance.shape));
    OP_REQUIRES_OK(
        c, c->GetAttr("communication_hint",
                      &col_params_.instance.impl_details.communication_hint));
    OP_REQUIRES_OK(
        c, c->GetAttr("timeout_seconds",
                      &col_params_.instance.impl_details.timeout_seconds));
    col_params_.is_source = true;
    col_params_.instance.impl_details.subdiv_offsets = {0};

    col_params_.name =
        strings::StrCat(name(), ": Broadcast(", col_params_.is_source, ")");
    col_params_.group.device_type = c->device_type();
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    CollectiveExecutor* col_exec = c->collective_executor();
    OP_REQUIRES_ASYNC(
        c, col_exec,
        errors::Internal(
            "Failed to get CollectiveExecutor from OpKernelContext for Op ",
            col_params_.name),
        done);
    // Allocate output on the first pass through this function.  This must be
    // done immediately, while we're still in the executor thread.  Otherwise
    // the memory is not guaranteed to be unused by any concurrently executing
    // GPU kernel.
    if (c->mutable_output(0) == nullptr) {
      // Allocate the output tensor, trying to reuse the input.
      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(c,
                           c->forward_input_or_allocate_output(
                               {0}, 0, col_params_.instance.shape, &output),
                           done);
    }
    if (!CanProceedWithCompute(c, col_exec, done)) return;
    OP_REQUIRES_ASYNC(
        c, col_params_.instance.shape.IsSameSize(c->input(0).shape()),
        errors::Internal("Declared shape of op ", col_params_.name,
                         " does not match shape of input"),
        done);

    auto actual_done = [c, group_key = col_params_.group.group_key,
                        instance_key = col_params_.instance.instance_key,
                        done](const Status& s) {
      VLOG(1) << "CollectiveBcastSendOpKernel ExecuteAsync done for collective "
              << c->op_kernel().name() << " device " << c->device()->name()
              << " group " << group_key << " instance " << instance_key
              << " status " << s;
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };
    VLOG(1) << "CollectiveBcastSendOpKernel ExecuteAsync start for collective "
            << col_params_.name << " device " << c->device()->name()
            << " group " << col_params_.group.group_key << " instance "
            << col_params_.instance.instance_key;
    col_exec->ExecuteAsync(c, col_params_, GetCollectiveKey(c), actual_done);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveBcastSendOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("CollectiveBcastSend").Device(DEVICE_CPU),
                        CollectiveBcastSendOpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveBcastSend").Device(DEVICE_GPU),
                        CollectiveBcastSendOpKernel);

class CollectiveBcastRecvOpKernel : public CollectiveOpKernel {
 public:
  explicit CollectiveBcastRecvOpKernel(OpKernelConstruction* c)
      : CollectiveOpKernel(c) {
    col_params_.instance.type = BROADCAST_COLLECTIVE;
    OP_REQUIRES_OK(c, c->GetAttr("group_size", &col_params_.group.group_size));
    OP_REQUIRES(
        c, col_params_.group.group_size > 0,
        errors::InvalidArgument("group_size must be positive integer but got ",
                                col_params_.group.group_size));
    OP_REQUIRES_OK(c, c->GetAttr("group_key", &col_params_.group.group_key));
    OP_REQUIRES_OK(
        c, c->GetAttr("instance_key", &col_params_.instance.instance_key));
    OP_REQUIRES_OK(c, c->GetAttr("T", &col_params_.instance.data_type));
    OP_REQUIRES_OK(c, c->GetAttr("shape", &col_params_.instance.shape));
    OP_REQUIRES_OK(
        c, c->GetAttr("communication_hint",
                      &col_params_.instance.impl_details.communication_hint));
    OP_REQUIRES_OK(
        c, c->GetAttr("timeout_seconds",
                      &col_params_.instance.impl_details.timeout_seconds));
    col_params_.is_source = false;
    col_params_.instance.impl_details.subdiv_offsets = {0};

    col_params_.name =
        strings::StrCat(name(), ": Broadcast(", col_params_.is_source, ")");
    col_params_.group.device_type = c->device_type();
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
    CollectiveExecutor* col_exec = c->collective_executor();
    OP_REQUIRES_ASYNC(
        c, col_exec,
        errors::Internal(
            "Failed to get CollectiveExecutor from OpKernelContext for Op ",
            col_params_.name),
        done);
    // Allocate output on the first pass through this function.  This must be
    // done immediately, while we're still in the executor thread.  Otherwise
    // the memory is not guaranteed to be unused by any concurrently executing
    // GPU kernel.
    if (c->mutable_output(0) == nullptr) {
      // No input, so must allocate output.
      Tensor* output = nullptr;
      OP_REQUIRES_OK_ASYNC(
          c, c->allocate_output(0, col_params_.instance.shape, &output), done);
    }
    if (!CanProceedWithCompute(c, col_exec, done)) return;

    auto actual_done = [c, group_key = col_params_.group.group_key,
                        instance_key = col_params_.instance.instance_key,
                        done](const Status& s) {
      VLOG(1) << "CollectiveBcastRecvOpKernel ExecuteAsync done for collective "
              << c->op_kernel().name() << " device " << c->device()->name()
              << " group " << group_key << " instance_key " << instance_key
              << " status  " << s;
      OP_REQUIRES_OK_ASYNC(c, s, done);
      done();
    };
    VLOG(1) << "CollectiveBcastRecvOpKernel ExecuteAsync start for collective "
            << col_params_.name << " device " << c->device()->name()
            << " group " << col_params_.group.group_key << " instance "
            << col_params_.instance.instance_key;
    col_exec->ExecuteAsync(c, col_params_, GetCollectiveKey(c), actual_done);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CollectiveBcastRecvOpKernel);
};

REGISTER_KERNEL_BUILDER(Name("CollectiveBcastRecv").Device(DEVICE_CPU),
                        CollectiveBcastRecvOpKernel);
REGISTER_KERNEL_BUILDER(Name("CollectiveBcastRecv").Device(DEVICE_GPU),
                        CollectiveBcastRecvOpKernel);

}  // namespace
}  // namespace tensorflow
/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("CollectiveReduce")
    .Input("input: T")
    .Output("data: T")
    .Attr("T: {float, float16, float64, int32, int64}")
    .Attr("group_size: int")
    .Attr("group_key: int")
    .Attr("instance_key: int")
    .Attr("merge_op: {'Min', 'Max', 'Mul', 'Add'}")
    .Attr("final_op: {'Id', 'Div'}")
    .Attr("subdiv_offsets: list(int)")
    .Attr("wait_for: list(int) = []")
    .Attr("communication_hint: string = 'auto'")
    .Attr("timeout_seconds: float = 0")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("CollectiveGather")
    .Input("input: T")
    .Output("data: T")
    .Attr("T: {float, float16, float64, int32, int64}")
    .Attr("group_size: int")
    .Attr("group_key: int")
    .Attr("instance_key: int")
    .Attr("shape: shape")
    .Attr("communication_hint: string = 'auto'")
    .Attr("timeout_seconds: float = 0")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // Scalar input is not supported.
      shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &unused));

      shape_inference::ShapeHandle in_subshape;
      TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 1, &in_subshape));

      auto input_first_dim_value = c->Value(c->Dim(c->input(0), 0));

      // This output should have the same shape as its input except the first
      // dimension should be multiplied by group size.
      shape_inference::ShapeHandle output_first_dim_as_shape;
      if (input_first_dim_value ==
          shape_inference::InferenceContext::kUnknownDim) {
        output_first_dim_as_shape =
            c->Vector(shape_inference::InferenceContext::kUnknownDim);
      } else {
        int group_size;
        TF_CHECK_OK(c->GetAttr("group_size", &group_size));
        std::vector<shape_inference::DimensionHandle> output_first_dim;
        output_first_dim.push_back(
            c->MakeDim(group_size * input_first_dim_value));
        output_first_dim_as_shape = c->MakeShape(output_first_dim);
      }

      shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(
          c->Concatenate(output_first_dim_as_shape, in_subshape, &out));
      c->set_output(0, out);
      return Status::OK();
    });

REGISTER_OP("CollectiveBcastSend")
    .Input("input: T")
    .Output("data: T")
    .Attr("T: {bool, float, float16, float64, int32, int64}")
    .Attr("group_size: int")
    .Attr("group_key: int")
    .Attr("instance_key: int")
    .Attr("shape: shape")
    .Attr("communication_hint: string = 'auto'")
    .Attr("timeout_seconds: float = 0")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);

REGISTER_OP("CollectiveBcastRecv")
    .Output("data: T")
    .Attr("T: {bool, float, float16, float64, int32, int64}")
    .Attr("group_size: int")
    .Attr("group_key: int")
    .Attr("instance_key: int")
    .Attr("shape: shape")
    .Attr("communication_hint: string = 'auto'")
    .Attr("timeout_seconds: float = 0")
    .SetIsStateful()
    .SetShapeFn(shape_inference::ExplicitShape);

}  // namespace tensorflow
