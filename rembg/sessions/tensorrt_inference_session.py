import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # This automatically initializes CUDA driver
import numpy as np
from dataclasses import dataclass


@dataclass
class TensorSpec:
    """Tensor specification for TensorRT tensors."""
    name: str
    dtype: any
    shape: tuple


class TensorRTInferenceSession:
    """This is an Inferece session for TensorRT compatiple with onnxruntime calls."""

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # Mapping from TensorRT DataType to NumPy dtype
    TRT_TO_NP_DTYPE = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT8: np.int8,
        trt.DataType.INT32: np.int32,
        trt.DataType.BOOL: np.bool_,
        trt.DataType.UINT8: np.uint8,
    }

    def __init__(self, engine_path: str, verbose: bool = False):
        """Initializes inference session with tensorRT engine path."""
        self.verbose = verbose
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self._get_engine_info()
        self._allocate_memory()

    def get_inputs(self):
        """Get input specifications."""
        return self.input_specs

    def load_engine(self, engine_path: str):
        """Loads TensorRT engine."""
        with open(engine_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _get_engine_info(self):
        """Get input and output tensor information."""
        self.input_specs = []
        self.output_specs = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            tensor_spec = TensorSpec(name=name, dtype=dtype, shape=shape)

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_specs.append(tensor_spec)
            else:
                self.output_specs.append(tensor_spec)

        if self.verbose:
            print(f"📋 Engine Info:")
            print(f"   Inputs: {len(self.input_specs)}")
            for spec in self.input_specs:
                print(f"     - {spec.name}: {spec.shape} ({spec.dtype})")
            print(f"   Outputs: {len(self.output_specs)}")
            for spec in self.output_specs:
                print(f"     - {spec.name}: {spec.shape} ({spec.dtype})")

    def _create_buffer_for_spec(self, spec: TensorSpec) -> tuple[dict, int]:
        """Create host and device buffers for a TensorSpec.

        Returns a tuple (buffer_dict, binding_int) where buffer_dict contains
        the host memory, device memory, name and shape, and binding_int is
        the integer device pointer used for TensorRT bindings.
        """
        shape = trt.volume(spec.shape)
        host_mem = cuda.pagelocked_empty(shape, dtype=self.TRT_TO_NP_DTYPE[spec.dtype])
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        buffer = {
            "host": host_mem,
            "device": device_mem,
            "spec": spec,
        }

        return buffer, int(device_mem)

    def _allocate_memory(self):
        """Allocate GPU and host memory for inference."""
        self.inputs = []
        self.outputs = []
        self.bindings = []

        # Allocate input and output memory
        for spec in self.input_specs:
            buf, binding = self._create_buffer_for_spec(spec)
            self.inputs.append(buf)
            self.bindings.append(binding)

        for spec in self.output_specs:
            buf, binding = self._create_buffer_for_spec(spec)
            self.outputs.append(buf)
            self.bindings.append(binding)

        # Create CUDA stream
        self.stream = cuda.Stream()

    def run(self, output_names, input_feed, run_options=None):
        """
        Compute the predictions.

        :param output_names: name of the outputs
        :param input_feed: dictionary ``{ input_name: input_value }``
        :param run_options: Not used, just to be compatible with onnx
        :return: list of results, every result is either a numpy array,
            a sparse tensor, a list or a dictionary.

        ::

            sess.run([output_name], {input_name: x})
        """
        # Copy input data to host memory
        name = self.inputs[0]["spec"].name
        np.copyto(self.inputs[0]["host"], input_feed[name].ravel())

        # Transfer input data to GPU
        cuda.memcpy_htod_async(
            self.inputs[0]["device"], self.inputs[0]["host"], self.stream
        )

        # Set tensor addresses
        for i, inp in enumerate(self.inputs):
            self.context.set_tensor_address(inp["spec"].name, inp["device"])

        for i, out in enumerate(self.outputs):
            self.context.set_tensor_address(out["spec"].name, out["device"])

        # Run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Transfer output data back to host
        cuda.memcpy_dtoh_async(
            self.outputs[0]["host"], self.outputs[0]["device"], self.stream
        )

        # Synchronize stream
        self.stream.synchronize()

        # Reshape output to original shape
        out = []
        out.append(self.outputs[0]["host"].reshape(self.outputs[0]["spec"].shape))

        return out
